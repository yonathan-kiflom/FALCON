from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

from .data import (
    CANONICAL_CATEGORY_IDS,
    CANONICAL_COMPONENT_ORDER,
    CANONICAL_LINK_ORDER,
    DATASET_FORMAT,
    DIRECTIONAL_RELATION_POLICY_MAXIMUM_OVERLAP,
    SCHEMA_VERSION,
    open_dataset,
)
from .partitions import build_partitions, load_partition, write_partitions

COMPONENTS = CANONICAL_COMPONENT_ORDER
LINKS = CANONICAL_LINK_ORDER
ALIASES = {
    "main charge": "explosive",
    "explosive charge": "explosive",
    "charge": "explosive",
}


def _json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read legacy JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Legacy JSON root must be an object: {path}")
    return value


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _name(value: str) -> str:
    normalized = value.strip().lower()
    return ALIASES.get(normalized, normalized)


def _number(value: Any, context: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{context} must be finite")
    return value


def _probability(value: Any, context: str) -> int | float:
    number = _number(value, context)
    if not 0.0 <= float(number) <= 1.0:
        raise ValueError(f"{context} must lie in [0, 1]")
    return number


def _optional_probability(value: Any, context: str) -> int | float | None:
    """Validate an available probability without inventing a missing label."""

    if value is None:
        return None
    return _probability(value, context)


def _validated_segmentation(
    value: Any,
    *,
    width: int,
    height: int,
    context: str,
) -> Any:
    """Validate and retain one immutable COCO instance mask.

    The migrated release records keep the original JSON representation so a
    detector proposal can be matched to the exact source instance mask rather
    than to any box with a positive overlap.
    """

    segmentation = copy.deepcopy(value)
    try:
        if isinstance(segmentation, list):
            if not segmentation:
                raise ValueError("polygon list is empty")
            encoded = mask_utils.frPyObjects(segmentation, height, width)
            encoded = mask_utils.merge(encoded)
        elif isinstance(segmentation, Mapping):
            size = segmentation.get("size")
            if (
                isinstance(size, str | bytes)
                or not isinstance(size, Sequence)
                or list(size) != [height, width]
            ):
                raise ValueError("RLE size disagrees with its source image")
            encoded = copy.deepcopy(dict(segmentation))
            if isinstance(encoded.get("counts"), str):
                encoded["counts"] = encoded["counts"].encode("ascii")
            elif isinstance(encoded.get("counts"), list):
                encoded = mask_utils.frPyObjects(encoded, height, width)
        else:
            raise ValueError("segmentation must be polygons or COCO RLE")
        decoded = np.asarray(mask_utils.decode(encoded))
    except Exception as exc:
        raise ValueError(f"{context} has malformed immutable segmentation") from exc
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width) or not bool(np.any(decoded)):
        raise ValueError(f"{context} immutable segmentation is empty or has the wrong shape")
    return segmentation


def _relative_filename(value: Any, context: str) -> str:
    """Accept normalized nested package members while rejecting path escape."""

    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{context} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{context} must be a normalized relative POSIX path")
    return value


def _image_dimensions(
    path: Path,
    declared: tuple[int, int],
    context: str,
) -> tuple[int, int]:
    """Return header dimensions and tolerate only the known legacy H/W swap."""

    try:
        with Image.open(path) as image:
            actual = (int(image.width), int(image.height))
    except OSError as exc:
        raise ValueError(f"Cannot read {context} image {path}: {exc}") from exc
    if actual[0] <= 0 or actual[1] <= 0:
        raise ValueError(f"{context} image has invalid dimensions: {path}")
    if declared not in (actual, actual[::-1]):
        raise ValueError(
            f"{context} dimensions {declared} disagree with image header {actual}: {path}"
        )
    return actual


def _component_names(values: Any, context: str, *, objects: bool) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise ValueError(f"{context} must be an array")
    seen: set[str] = set()
    for value in values:
        if objects:
            if not isinstance(value, Mapping) or not isinstance(value.get("category_name"), str):
                raise ValueError(f"{context} entries must contain category_name")
            raw_name = value["category_name"]
        else:
            if not isinstance(value, str):
                raise ValueError(f"{context} entries must be strings")
            raw_name = value
        name = _name(raw_name)
        if name not in COMPONENTS:
            raise ValueError(f"{context} contains unknown component {name!r}")
        if name in seen:
            if objects:
                # Multiple immutable instances of one category are valid.  The
                # state inventory is a set; component_counts below preserves
                # and verifies the instance multiplicity for current exports.
                continue
            raise ValueError(f"{context} contains duplicate component {name!r}")
        seen.add(name)
    return tuple(name for name in COMPONENTS if name in seen)


def _dataset_signature(path: Path) -> bool:
    return all(
        (path / split / "images").is_dir() and (path / split / "tasks").is_dir()
        for split in ("train", "test")
    )


def resolve_dataset_root(requested: Path) -> Path:
    """Resolve exactly one flat or one-level nested DismantledAnnotated root."""

    try:
        requested = requested.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Dataset path does not exist: {requested}") from exc
    if not requested.is_dir():
        raise ValueError(f"Dataset path is not a directory: {requested}")

    candidates: list[Path] = []
    possible = [requested]
    possible.extend(
        child for child in requested.iterdir() if child.is_dir() and not child.is_symlink()
    )
    for candidate in possible:
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(requested)
        except ValueError as exc:
            raise ValueError(
                f"Nested dataset candidate escapes requested root: {candidate}"
            ) from exc
        if _dataset_signature(resolved) and resolved not in candidates:
            candidates.append(resolved)

    if not candidates:
        raise ValueError(
            f"No Falcon dataset root under {requested}; expected train/images, "
            "train/tasks, test/images, and test/tasks"
        )
    if len(candidates) != 1:
        joined = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Ambiguous dataset root under {requested}: {joined}")
    return candidates[0]


def _legacy_source_id(item: Mapping[str, Any], base_ids: Mapping[str, int]) -> tuple[int, str]:
    explicit = item.get("source_image_id")
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
            raise ValueError("source_image_id must be a positive integer")
        if not item.get("is_variant") and explicit != int(item["image_id"]):
            raise ValueError("A base state source_image_id must equal image_id")
        return explicit, "explicit"
    if not item.get("is_variant"):
        return int(item["image_id"]), "native"
    file_name = _relative_filename(item.get("file_name"), "variant file_name")
    stem = file_name.split("_missing_", 1)[0]
    if stem not in base_ids:
        raise ValueError(f"Cannot recover source image for legacy variant {file_name}")
    return base_ids[stem], "legacy_filename"


def _score(answer: str) -> float | None:
    match = re.search(r"score\s*=\s*([0-9]+(?:\.[0-9]+)?)", answer)
    if match is None:
        return None
    value = float(match.group(1))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Functional completeness is outside [0, 1]: {answer!r}")
    return value


def _completeness_labels(tasks: Path) -> dict[int, float | None]:
    rows = _json(tasks / "grounding_semantic.json").get("grounding_semantic")
    if not isinstance(rows, list):
        raise ValueError("grounding_semantic.json lacks grounding_semantic array")
    result: dict[int, float | None] = {}
    supported_types = {"structural_completeness", "functional_completeness"}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("type") not in supported_types:
            continue
        image_id = int(row["image_id"])
        if image_id in result:
            raise ValueError(f"Duplicate structural completeness label for image {image_id}")
        explicit = row.get("component_coverage")
        result[image_id] = (
            _optional_probability(
                explicit,
                f"image {image_id} component_coverage",
            )
            if explicit is not None
            else _score(str(row.get("answer", "")))
        )
    return result


def _state_rows(
    split_root: Path,
    provenance: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if not provenance.strip():
        raise ValueError("Label provenance must be non-empty")
    tasks = split_root / "tasks"
    safety_payload = _json(tasks / "safety_relation_task.json")
    safety = safety_payload.get("safety_relation")
    safety_meta = safety_payload.get("meta")
    export_schema = (
        safety_meta.get("schema_version")
        if isinstance(safety_meta, Mapping)
        else None
    )
    current_schema = export_schema in {
        "falcon-dual-exports-v2",
        "falcon-dual-exports-v3",
    }
    nullable_directional_axes = export_schema == "falcon-dual-exports-v3"
    functional = _json(tasks / "functional_grounding.json")
    images = functional.get("images")
    if not isinstance(safety, list) or not isinstance(images, list):
        raise ValueError("Legacy safety_relation and functional image arrays are required")
    refseg_payload = _json(tasks / "refseg.json")
    refseg_annotations = refseg_payload.get("annotations")
    refseg_categories = refseg_payload.get("categories")
    if not isinstance(refseg_annotations, list) or not isinstance(refseg_categories, list):
        raise ValueError("refseg.json requires annotations and categories arrays")
    refseg_names = {
        int(category["id"]): _name(str(category["name"]))
        for category in refseg_categories
        if isinstance(category, Mapping)
    }
    refseg_by_id: dict[int, Mapping[str, Any]] = {}
    for annotation in refseg_annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("refseg annotations must be objects")
        annotation_id = int(annotation["id"])
        if annotation_id in refseg_by_id:
            raise ValueError(f"Duplicate RefSeg annotation ID {annotation_id}")
        refseg_by_id[annotation_id] = annotation

    sizes: dict[int, tuple[int, int]] = {}
    state_file_names: dict[int, str] = {}
    base_ids: dict[str, int] = {}
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError("functional_grounding images must be objects")
        image_id = int(image["id"])
        if image_id in sizes:
            raise ValueError(f"Duplicate functional image ID {image_id}")
        width, height = int(image["width"]), int(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions for functional image {image_id}")
        sizes[image_id] = (width, height)
        file_name = _relative_filename(
            image.get("file_name"), f"functional image {image_id}"
        )
        state_file_names[image_id] = file_name
        stem = Path(file_name).stem
        if "_missing_" not in stem:
            if stem in base_ids:
                raise ValueError(f"Duplicate base filename stem {stem!r}")
            base_ids[stem] = image_id

    base_presence: dict[int, set[str]] = {}
    for item in safety:
        if not isinstance(item, Mapping) or item.get("is_variant"):
            continue
        image_id = int(item["image_id"])
        if image_id in base_presence:
            raise ValueError(f"Duplicate base safety record for image {image_id}")
        base_presence[image_id] = set(
            _component_names(
                item.get("present_components", []),
                f"image {image_id} present_components",
                objects=True,
            )
        )

    completeness = _completeness_labels(tasks)
    canonical_pair_sets = {frozenset(pair) for pair in LINKS}
    validated_refseg: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    for item in safety:
        if not isinstance(item, Mapping):
            raise ValueError("safety_relation entries must be objects")
        image_id = int(item["image_id"])
        if image_id in by_id:
            raise ValueError(f"Duplicate safety record for image {image_id}")
        if image_id not in sizes:
            raise ValueError(f"Safety image {image_id} has no functional image metadata")
        source_id, source_method = _legacy_source_id(item, base_ids)
        file_name = _relative_filename(
            item.get("file_name"), f"safety image {image_id}"
        )
        if current_schema and file_name != state_file_names[image_id]:
            raise ValueError(
                f"Current safety image {image_id} file_name disagrees with functional metadata"
            )
        legacy_size = sizes[image_id]
        image_root = (
            split_root / "grounding_semantic" / "images"
            if item.get("is_variant")
            else split_root / "images"
        )
        expected_image_path = (
            f"grounding_semantic/images/{file_name}"
            if item.get("is_variant")
            else f"images/{file_name}"
        )
        if current_schema and item.get("image_path") != expected_image_path:
            raise ValueError(
                f"Current safety image {image_id} image_path must equal "
                f"{expected_image_path!r}"
            )
        width, height = _image_dimensions(
            image_root / file_name,
            legacy_size,
            f"safety image {image_id}",
        )

        present = _component_names(
            item.get("present_components", []),
            f"image {image_id} present_components",
            objects=True,
        )
        declared_missing = _component_names(
            item.get("missing_components", []),
            f"image {image_id} missing_components",
            objects=False,
        )
        expected_missing = tuple(name for name in COMPONENTS if name not in present)
        if current_schema and declared_missing != expected_missing:
            raise ValueError(
                f"Current image {image_id} missing_components must exactly equal "
                "the complement of present_components"
            )
        if not current_schema and not set(declared_missing).issubset(expected_missing):
            raise ValueError(f"Image {image_id} missing_components contradicts present_components")

        is_variant = bool(item.get("is_variant"))
        removed: tuple[str, ...] = ()
        declared_removed = _component_names(
            item.get("removed_components", []),
            f"image {image_id} removed_components",
            objects=False,
        )
        if is_variant:
            if source_id not in base_presence:
                raise ValueError(f"Variant {image_id} has no base safety record {source_id}")
            source_present = base_presence[source_id]
            if not set(present).issubset(source_present):
                raise ValueError(f"Variant {image_id} adds a component absent from its source")
            removed = tuple(
                name for name in COMPONENTS if name in source_present and name not in present
            )
            if not removed:
                raise ValueError(f"Variant {image_id} does not remove a source component")
            if current_schema and declared_removed != removed:
                raise ValueError(
                    f"Current variant {image_id} removed_components does not equal "
                    "the source-to-variant inventory delta"
                )
            if not current_schema and not set(removed).issubset(declared_missing):
                raise ValueError(
                    f"Variant {image_id} legacy missing_components omits a removed component"
                )
        elif current_schema and declared_removed:
            raise ValueError(
                f"Current base image {image_id} cannot declare removed_components"
            )

        link_values: dict[frozenset[str], int | float] = {}
        raw_safety_links = item.get("safety_links")
        if raw_safety_links is not None and not isinstance(raw_safety_links, list):
            raise ValueError(f"Image {image_id} safety_links must be an array")
        for link in raw_safety_links or ():
            if not isinstance(link, Mapping):
                raise ValueError(f"Image {image_id} safety link must be an object")
            source = link.get("src_component")
            target = link.get("dst_component")
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(f"Image {image_id} safety link names must be strings")
            pair = frozenset((_name(source), _name(target)))
            if pair not in canonical_pair_sets:
                raise ValueError(f"Image {image_id} has unknown safety link {sorted(pair)!r}")
            if pair in link_values:
                raise ValueError(f"Image {image_id} has a duplicate safety link {sorted(pair)!r}")
            if not pair.issubset(present):
                raise ValueError(f"Image {image_id} links a missing component")
            link_values[pair] = _probability(
                link.get("safety_probability"),
                f"image {image_id} safety_probability",
            )

        present_items = item.get("present_components", [])
        observed_counts = {name: 0 for name in COMPONENTS}
        grounding = []
        for component in present_items:
            name = _name(component["category_name"])
            observed_counts[name] += 1
            bbox = component.get("bbox")
            if isinstance(bbox, str | bytes) or not isinstance(bbox, Sequence) or len(bbox) != 4:
                raise ValueError(f"Image {image_id} component bbox must contain four numbers")
            segment_id = int(component["segment_id"])
            raw_source_annotation_id = component.get("source_annotation_id")
            if current_schema and raw_source_annotation_id is None:
                raise ValueError(
                    f"Current image {image_id} component {segment_id} lacks "
                    "source_annotation_id"
                )
            source_annotation_id = int(
                segment_id
                if raw_source_annotation_id is None
                else raw_source_annotation_id
            )
            if current_schema and source_annotation_id != segment_id:
                raise ValueError(
                    f"Current image {image_id} segment/source annotation IDs disagree"
                )
            grounding_entry: dict[str, Any] = {
                "instance_id": segment_id,
                "source_annotation_id": source_annotation_id,
                "category": name,
                "bbox_xywh": [
                    _number(value, f"image {image_id} bbox") for value in bbox
                ],
            }
            source_annotation = refseg_by_id.get(source_annotation_id)
            if source_annotation is None:
                if current_schema:
                    raise ValueError(
                        f"Current image {image_id} component {segment_id} has no exact "
                        "RefSeg source annotation"
                    )
            else:
                if int(source_annotation["image_id"]) != source_id:
                    raise ValueError(
                        f"Image {image_id} component {segment_id} resolves to a RefSeg "
                        "annotation from another source image"
                    )
                source_category = refseg_names.get(int(source_annotation["category_id"]))
                if source_category != name:
                    raise ValueError(
                        f"Image {image_id} component {segment_id} category disagrees "
                        "with its RefSeg source annotation"
                    )
                source_bbox = source_annotation.get("bbox")
                if (
                    isinstance(source_bbox, str | bytes)
                    or not isinstance(source_bbox, Sequence)
                    or len(source_bbox) != 4
                    or any(
                        not math.isclose(float(left), float(right), abs_tol=1e-6)
                        for left, right in zip(bbox, source_bbox, strict=True)
                    )
                ):
                    raise ValueError(
                        f"Image {image_id} component {segment_id} bbox disagrees with "
                        "its RefSeg source annotation"
                    )
                if source_annotation_id not in validated_refseg:
                    validated_refseg[source_annotation_id] = _validated_segmentation(
                        source_annotation.get("segmentation"),
                        width=width,
                        height=height,
                        context=(
                            f"image {image_id} source annotation {source_annotation_id}"
                        ),
                    )
                grounding_entry["segmentation"] = copy.deepcopy(
                    validated_refseg[source_annotation_id]
                )
                grounding_entry["match_policy"] = "exact_mask_iou"
            grounding.append(grounding_entry)

        grounding_by_id = {
            int(entry["source_annotation_id"]): entry for entry in grounding
        }
        raw_candidates = item.get("candidate_relations", [])
        if isinstance(raw_candidates, str | bytes) or not isinstance(
            raw_candidates, Sequence
        ):
            raise ValueError(f"Image {image_id} candidate_relations must be an array")
        candidate_relations: list[dict[str, Any]] = []
        observed_candidate_pairs: set[frozenset[int]] = set()
        for index, candidate in enumerate(raw_candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"Image {image_id} candidate relation {index} must be an object"
                )
            if candidate.get("connection_observed") is not False:
                raise ValueError(
                    f"Image {image_id} candidate relation {index} must not assert a connection"
                )
            if current_schema and (
                candidate.get("telemetry_only") is not True
                or candidate.get("training_target") is not False
            ):
                raise ValueError(
                    f"Current image {image_id} candidate relation {index} is not "
                    "telemetry-only"
                )
            src_id = int(candidate["src_segment_id"])
            dst_id = int(candidate["dst_segment_id"])
            pair = frozenset((src_id, dst_id))
            if src_id == dst_id or pair in observed_candidate_pairs:
                raise ValueError(
                    f"Image {image_id} has a duplicate/self candidate relation"
                )
            observed_candidate_pairs.add(pair)
            src_grounding = grounding_by_id.get(src_id)
            dst_grounding = grounding_by_id.get(dst_id)
            if src_grounding is None or dst_grounding is None:
                raise ValueError(
                    f"Image {image_id} candidate relation refers to a missing instance"
                )
            if _name(str(candidate.get("src_component", ""))) != src_grounding["category"]:
                raise ValueError(
                    f"Image {image_id} candidate relation source category disagrees"
                )
            if _name(str(candidate.get("dst_component", ""))) != dst_grounding["category"]:
                raise ValueError(
                    f"Image {image_id} candidate relation target category disagrees"
                )
            horizontal_relation = candidate.get("horizontal_relation")
            vertical_relation = candidate.get("vertical_relation")
            if horizontal_relation not in {
                "left_of",
                "right_of",
                "horizontally_aligned",
                "left of",
                "right of",
                "horizontally aligned",
            } and not (nullable_directional_axes and horizontal_relation is None):
                raise ValueError(
                    f"Image {image_id} candidate relation has invalid horizontal geometry"
                )
            if vertical_relation not in {
                "above",
                "below",
                "vertically_aligned",
                "vertically aligned",
            } and not (nullable_directional_axes and vertical_relation is None):
                raise ValueError(
                    f"Image {image_id} candidate relation has invalid vertical geometry"
                )
            if nullable_directional_axes:
                directional_policy = candidate.get(
                    "directional_relation_policy_version"
                )
                directional_maximum = (
                    DIRECTIONAL_RELATION_POLICY_MAXIMUM_OVERLAP.get(
                        directional_policy
                    )
                    if isinstance(directional_policy, str)
                    else None
                )
                if (
                    directional_maximum is None
                    or candidate.get("maximum_axis_projection_overlap")
                    != directional_maximum
                ):
                    raise ValueError(
                        f"Image {image_id} candidate relation has invalid directional policy"
                    )
                for axis, relation in (
                    ("horizontal", horizontal_relation),
                    ("vertical", vertical_relation),
                ):
                    eligible = candidate.get(f"{axis}_relation_eligible")
                    claim_id = candidate.get(f"{axis}_relation_claim_id")
                    if (
                        not isinstance(eligible, bool)
                        or (relation is not None) is not eligible
                        or (isinstance(claim_id, str) and bool(claim_id)) is not eligible
                        or (not eligible and claim_id is not None)
                    ):
                        raise ValueError(
                            f"Image {image_id} candidate relation has inconsistent "
                            f"{axis} eligibility"
                        )
            candidate_relations.append(copy.deepcopy(dict(candidate)))
        if current_schema:
            expected_candidate_pairs = {
                frozenset((left, right))
                for offset, left in enumerate(grounding_by_id)
                for right in list(grounding_by_id)[offset + 1 :]
            }
            if observed_candidate_pairs != expected_candidate_pairs:
                raise ValueError(
                    f"Current image {image_id} candidate relations do not cover every "
                    "present instance pair"
                )

        if current_schema:
            raw_counts = item.get("component_counts")
            if raw_counts is not None and (
                not isinstance(raw_counts, Mapping)
                or set(raw_counts) != set(COMPONENTS)
            ):
                raise ValueError(
                    f"Current image {image_id} component_counts must cover the canonical ontology"
                )
            if raw_counts is not None:
                checked_counts: dict[str, int] = {}
                for name in COMPONENTS:
                    value = raw_counts[name]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(
                            f"Current image {image_id} component_counts[{name!r}] "
                            "must be a non-negative integer"
                        )
                    checked_counts[name] = value
                if checked_counts != observed_counts:
                    raise ValueError(
                        f"Current image {image_id} component_counts disagrees with "
                        "present_components"
                    )

        risk = _optional_probability(
            item.get("scene_risk"),
            f"image {image_id} scene_risk",
        )
        links = [link_values.get(frozenset(pair)) for pair in LINKS]
        record = {
            "image_id": image_id,
            "source_image_id": source_id,
            "kind": "counterfactual" if is_variant else "base",
            "file_name": file_name,
            "image_relpath": (
                f"grounding_semantic/images/{file_name}" if is_variant else f"images/{file_name}"
            ),
            "width": width,
            "height": height,
            "removed_components": list(removed),
            "present_components": list(present),
            "missing_components": list(expected_missing),
            "grounding": grounding,
            "candidate_relations": candidate_relations,
            "targets": {
                "presence": [int(name in present) for name in COMPONENTS],
                "completeness": completeness.get(image_id),
                "risk": risk,
                "links": links,
            },
            "provenance": {
                "risk": {
                    "source": provenance if risk is not None else "unavailable",
                    "version": "v1",
                },
                "links": {
                    "source": (
                        provenance
                        if any(value is not None for value in links)
                        else "unavailable"
                    ),
                    "version": "v1",
                },
                "source_image_id": source_method,
            },
        }
        rows.append(record)
        by_id[image_id] = record
    return rows, by_id


def _state_grounding_entry(
    state: Mapping[str, Any],
    source_annotation_id: int,
    context: str,
) -> dict[str, Any]:
    matches = [
        entry
        for entry in state.get("grounding", [])
        if int(entry.get("source_annotation_id", entry.get("instance_id", -1)))
        == source_annotation_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{context} does not resolve uniquely to state source annotation "
            f"{source_annotation_id}"
        )
    return copy.deepcopy(dict(matches[0]))


def _grounding(
    row: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for item in row.get("grounding", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("category_name"), str):
            raise ValueError("Legacy grounding entries require category_name")
        raw_annotation_id = item.get("source_annotation_id", item.get("segment_id"))
        if raw_annotation_id is not None:
            exact = _state_grounding_entry(
                state,
                int(raw_annotation_id),
                "grounding task row",
            )
            if exact["category"] != _name(item["category_name"]):
                raise ValueError("Grounding category disagrees with its source annotation")
            result.append(exact)
            continue
        if any("segmentation" in entry for entry in state.get("grounding", [])):
            raise ValueError(
                "Current grounding row lacks source_annotation_id/segment_id"
            )
        bbox = item.get("bbox")
        if isinstance(bbox, str | bytes) or not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise ValueError("Legacy grounding bbox must contain four numbers")
        result.append(
            {
                "category": _name(item["category_name"]),
                "bbox_xywh": [_number(value, "grounding bbox") for value in bbox],
            }
        )
    return result


def _example_rows(
    split_root: Path,
    states: Mapping[int, dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    tasks = split_root / "tasks"

    def emit(
        task: str,
        row: Mapping[str, Any],
        prompt: str,
        answer: str,
        grounding: Iterable[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        image_id = int(row["image_id"])
        if image_id not in states:
            raise ValueError(f"Task {task!r} references unknown image {image_id}")
        state = states[image_id]
        return {
            "id": f"{task}:{row.get('question_id', row.get('id', image_id))}:{image_id}",
            "image_id": image_id,
            "image_relpath": state["image_relpath"],
            "task": task,
            "prompt": prompt,
            "answer": answer,
            "grounding": list(grounding),
        }

    for row in _json(tasks / "captions.json").get("captions", []):
        yield emit("caption", row, "Describe the X-ray image.", str(row["caption"]))
    for row in _json(tasks / "vqa.json").get("vqa", []):
        yield emit(
            f"vqa/{row.get('type', 'other')}",
            row,
            str(row["question"]),
            str(row["answer"]),
        )
    for key, task in (
        ("grounding_semantic", "grounding"),
        ("functional_grounding", "functional_grounding"),
    ):
        for row in _json(tasks / f"{key}.json").get(key, []):
            task_name = task if key == "functional_grounding" else f"grounding/{row['type']}"
            yield emit(
                task_name,
                row,
                str(row["question"]),
                str(row["answer"]),
                _grounding(row, states[int(row["image_id"])]),
            )
    for filename, task in (("refseg.json", "refseg"), ("refpanoptic.json", "refpanoptic")):
        payload = _json(tasks / filename)
        names = {int(item["id"]): _name(item["name"]) for item in payload["categories"]}
        for annotation in payload["annotations"]:
            category = names[int(annotation["category_id"])]
            for sentence in annotation.get("sentences", []):
                sentence_id = sentence.get("sent_id")
                if sentence_id is None:
                    sentence_id = annotation.get("id", annotation.get("segment_id"))
                if sentence_id is None:
                    raise ValueError(f"{task} annotation has no sentence or segment identifier")
                row = {
                    "image_id": annotation["image_id"],
                    "id": sentence_id,
                }
                source_annotation_id = int(
                    annotation.get(
                        "source_annotation_id",
                        annotation.get("id", annotation.get("segment_id")),
                    )
                )
                state = states[int(annotation["image_id"])]
                if "segmentation" in annotation:
                    grounding_entry = {
                        "instance_id": int(
                            annotation.get("segment_id", source_annotation_id)
                        ),
                        "source_annotation_id": source_annotation_id,
                        "category": category,
                        "bbox_xywh": [
                            _number(value, "referring bbox")
                            for value in annotation["bbox"]
                        ],
                        "segmentation": _validated_segmentation(
                            annotation["segmentation"],
                            width=int(state["width"]),
                            height=int(state["height"]),
                            context=f"{task} annotation {source_annotation_id}",
                        ),
                        "match_policy": "exact_mask_iou",
                    }
                else:
                    grounding_entry = _state_grounding_entry(
                        state,
                        source_annotation_id,
                        f"{task} annotation",
                    )
                grounding = [grounding_entry]
                yield emit(task, row, str(sentence["raw"]), category, grounding)
    panoptic = _json(tasks / "panoptic.json")
    names = {int(item["id"]): _name(item["name"]) for item in panoptic["categories"]}
    for annotation in panoptic["annotations"]:
        state = states[int(annotation["image_id"])]
        grounding = [
            _state_grounding_entry(
                state,
                int(segment.get("source_annotation_id", segment["id"])),
                "panoptic segment",
            )
            for segment in annotation.get("segments_info", [])
        ]
        for segment, exact in zip(
            annotation.get("segments_info", []), grounding, strict=True
        ):
            if exact["category"] != names[int(segment["category_id"])]:
                raise ValueError("Panoptic category disagrees with its source annotation")
        answer = ", ".join(item["category"] for item in grounding) or "none"
        yield emit(
            "panoptic",
            annotation,
            "Segment every dismantled threat component.",
            answer,
            grounding,
        )
    for item in states.values():
        risk = item["targets"]["risk"]
        links = item["targets"]["links"]
        available_links = any(value is not None for value in links)
        answer_payload: dict[str, Any] = {
            "presence": item["targets"]["presence"],
        }
        if risk is not None:
            answer_payload["risk"] = risk
        if risk is not None or available_links:
            answer_payload["links"] = links
        grounding = copy.deepcopy(item["grounding"])
        candidate_relations = []
        for relation in item.get("candidate_relations", []):
            # Current dual-model exports deliberately mark pair geometry as
            # audit telemetry, not a learner/evaluation target.  Preserve it
            # in states.jsonl, but never serialize it into a supervised answer.
            if (
                relation.get("telemetry_only") is True
                and relation.get("training_target") is False
            ):
                continue
            exported_relation = {
                "src_component": relation["src_component"],
                "dst_component": relation["dst_component"],
                "src_segment_id": relation["src_segment_id"],
                "dst_segment_id": relation["dst_segment_id"],
            }
            for field in ("horizontal_relation", "vertical_relation"):
                if relation.get(field) is not None:
                    exported_relation[field] = relation[field]
            if not {
                "horizontal_relation",
                "vertical_relation",
            } & exported_relation.keys():
                continue
            candidate_relations.append(exported_relation)
        if candidate_relations:
            answer_payload["candidate_relations"] = candidate_relations
        answer = json.dumps(answer_payload, separators=(",", ":"), allow_nan=False)
        yield emit(
            "safety",
            {"image_id": item["image_id"], "id": item["image_id"]},
            (
                "Report annotated component presence, functional links, and scene risk."
                if risk is not None or available_links
                else (
                    "Report the annotated component presence and observable spatial "
                    "geometry. Risk and physical links are unavailable."
                    if candidate_relations
                    else (
                        "Report the annotated component presence. Risk, physical links, "
                        "and spatial telemetry are unavailable as targets."
                    )
                )
            ),
            answer,
            grounding,
        )


def migrate_dataset(root: Path, provenance: str) -> Path:
    root = resolve_dataset_root(root)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "category_ids": dict(CANONICAL_CATEGORY_IDS),
        "component_order": list(COMPONENTS),
        "link_order": [list(pair) for pair in LINKS],
        "splits": {},
    }
    for split in ("train", "test"):
        split_root = root / split
        states, by_id = _state_rows(split_root, provenance)
        _write_jsonl(split_root / "annotations/states.jsonl", states)
        manifest["splits"][split] = {
            "root": split,
            "images": "images",
            "counterfactual_images": "grounding_semantic/images",
            "states": "annotations/states.jsonl",
        }
        if split == "train":
            _write_jsonl(
                split_root / "tasks/examples.jsonl",
                _example_rows(split_root, by_id),
            )
            manifest["splits"][split]["tasks"] = "tasks/examples.jsonl"
    with (root / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    # Fail the migration if its result cannot be consumed through the public
    # API or if any explicit image path is absent/unsafe.
    dataset = open_dataset(root)
    for split in ("train", "test"):
        for record in dataset.records(split):
            dataset.resolve_image(split, record)
    return root


def _resolve_coco(root: Path, split: str, archive_root: Path | None = None) -> Path:
    filename = f"coco_annotations_{split}.json"
    candidates = [root / split / "tasks" / filename, root / filename]
    if archive_root is not None and archive_root != root:
        candidates.append(archive_root / filename)
    existing = [path.resolve(strict=True) for path in candidates if path.is_file()]
    if not existing:
        joined = " or ".join(str(path) for path in candidates)
        raise ValueError(f"Missing {split} COCO annotations; expected {joined}")
    if len(existing) != 1:
        joined = ", ".join(str(path) for path in candidates if path.is_file())
        raise ValueError(f"Ambiguous {split} COCO annotations: {joined}")
    try:
        existing[0].relative_to(archive_root or root)
    except ValueError as exc:
        raise ValueError(f"{split} COCO annotation path escapes dataset root") from exc
    return existing[0]


def _validate_coco(payload: Mapping[str, Any], name: str) -> tuple[set[int], set[str]]:
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list) or not images:
        raise ValueError(f"{name} COCO must contain non-empty images and annotations arrays")
    image_ids: set[int] = set()
    filenames: set[str] = set()
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError(f"{name} COCO image entries must be objects")
        image_id = int(image["id"])
        filename = _relative_filename(
            image.get("file_name"), f"{name} COCO file_name"
        )
        if image_id in image_ids or filename in filenames:
            raise ValueError(f"{name} COCO has duplicate image IDs or filenames")
        image_ids.add(image_id)
        filenames.add(filename)
    annotation_ids: set[int] = set()
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError(f"{name} COCO annotations must be objects")
        annotation_id = int(annotation["id"])
        image_id = int(annotation["image_id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"{name} COCO has duplicate annotation ID {annotation_id}")
        if image_id not in image_ids:
            raise ValueError(f"{name} COCO annotation references missing image {image_id}")
        annotation_ids.add(annotation_id)
    return image_ids, filenames


def _normalize_coco_geometry(
    payload: Mapping[str, Any],
    image_root: Path,
    name: str,
) -> None:
    """Correct legacy swapped dimensions and validate every detector box."""

    dimensions: dict[int, tuple[int, int]] = {}
    for image in payload["images"]:
        image_id = int(image["id"])
        declared = (int(image["width"]), int(image["height"]))
        actual = _image_dimensions(
            image_root / image["file_name"],
            declared,
            f"{name} COCO image {image_id}",
        )
        image["width"], image["height"] = actual
        dimensions[image_id] = actual

    for annotation in payload["annotations"]:
        annotation_id = int(annotation["id"])
        bbox = annotation.get("bbox")
        if isinstance(bbox, str | bytes) or not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise ValueError(f"{name} COCO annotation {annotation_id} has an invalid bbox")
        x, y, width, height = (
            float(_number(value, f"{name} COCO annotation {annotation_id} bbox"))
            for value in bbox
        )
        image_width, image_height = dimensions[int(annotation["image_id"])]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"{name} COCO annotation {annotation_id} has invalid bbox geometry")
        if x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
            raise ValueError(f"{name} COCO annotation {annotation_id} bbox exceeds its image")


def _validation_ids(images: Sequence[Mapping[str, Any]], valid_percent: int) -> set[int]:
    if not 1 <= valid_percent <= 99:
        raise ValueError("valid_percent must be between 1 and 99")
    if len(images) < 2:
        raise ValueError("At least two training images are required for a non-leaking split")
    ranked = sorted(int(image["id"]) for image in images)
    random.Random(0).shuffle(ranked)
    count = max(1, min(len(ranked) - 1, (len(ranked) * valid_percent + 50) // 100))
    return set(ranked[:count])


def _class_agnostic(payload: Mapping[str, Any], image_ids: set[int]) -> dict[str, Any]:
    return {
        "images": [image for image in payload["images"] if int(image["id"]) in image_ids],
        "annotations": [
            dict(annotation, category_id=1)
            for annotation in payload["annotations"]
            if int(annotation["image_id"]) in image_ids
        ],
        "categories": [{"id": 1, "name": "component", "supercategory": "threat"}],
        "licenses": payload.get("licenses", []),
        "info": payload.get("info", {}),
    }


def _link(source: Path, target: Path, mode: str) -> None:
    if not source.is_file():
        raise ValueError(f"Detector source image is missing: {source}")
    if target.exists() or target.is_symlink():
        raise ValueError(f"Refusing to reuse detector output path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve(strict=True))
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unknown link mode {mode!r}")


def prepare_partitions(
    root: Path,
    output: Path,
    *,
    validation_percent: int = 5,
    calibration_percent: int = 5,
    seed: int = 0,
):
    """Write parent-grouped train, validation, and calibration partitions."""

    dataset = open_dataset(root)
    if getattr(dataset, "format", None) != DATASET_FORMAT:
        raise ValueError("partition preparation requires a falcon-x dataset")
    plan = build_partitions(
        dataset,
        validation_percent=validation_percent,
        calibration_percent=calibration_percent,
        seed=seed,
    )
    write_partitions(plan, output)
    return plan


def _native_detector_output(
    dataset,
    output: Path,
    *,
    valid_percent: int,
    calibration_percent: int,
    seed: int,
    link_mode: str,
    partition: Path | None,
    image_scope: str,
) -> dict[str, set[int]]:
    if image_scope not in ("base", "all"):
        raise ValueError("image_scope must be 'base' or 'all'")
    if link_mode not in ("hardlink", "symlink", "copy"):
        raise ValueError("link_mode must be 'hardlink', 'symlink', or 'copy'")
    for split in ("train", "test"):
        # Validate image headers and instance masks before creating the export.
        report = dataset.validate(split, tasks=(), full=True)
        if not report["ok"]:
            raise ValueError(f"Native {split} full validation failed: {report['errors'][:3]}")

    partition_plan = None
    source_partition = None
    if partition is None:
        partition_plan = build_partitions(
            dataset,
            validation_percent=valid_percent,
            calibration_percent=calibration_percent,
            seed=seed,
        )
    else:
        source_partition = partition.expanduser()
        if source_partition.is_symlink():
            raise ValueError("Partition input must not be a symlink")
        source_partition = source_partition.resolve(strict=True)
        # Validate every role before creating the detector export directory.
        for role in ("train", "validation", "calibration"):
            load_partition(dataset, source_partition, role=role)

    output = output.expanduser().resolve(strict=False)
    try:
        output.relative_to(dataset.root)
    except ValueError:
        pass
    else:
        raise ValueError("Detector output must be outside the immutable dataset root")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Detector output is not a directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"Detector output must be empty: {output}")
    else:
        output.mkdir(parents=True)

    embedded_partition = output / "partitions.json"
    if partition_plan is not None:
        write_partitions(partition_plan, embedded_partition)
    else:
        assert source_partition is not None
        shutil.copy2(source_partition, embedded_partition)

    selections = {
        role: load_partition(dataset, embedded_partition, role=role)
        for role in ("train", "validation", "calibration")
    }
    train_images = {row["id"]: row for row in dataset.images("train")}
    test_images = {row["id"]: row for row in dataset.images("test")}
    role_specs = (
        ("train", "train", selections["train"].image_ids, train_images),
        ("valid", "train", selections["validation"].image_ids, train_images),
        (
            "calibration",
            "train",
            selections["calibration"].image_ids,
            train_images,
        ),
        ("test", "test", frozenset(test_images), test_images),
    )

    next_image_id = 1
    next_annotation_id = 1
    result: dict[str, set[int]] = {}
    mapping_rows: list[dict[str, Any]] = []
    report_roles: dict[str, Any] = {}
    all_native_selected: dict[tuple[str, int | str], str] = {}
    for output_role, source_split, selected_ids, image_index in role_specs:
        if image_scope == "base":
            selected_ids = frozenset(
                image_id
                for image_id in selected_ids
                if "source_image_id" not in image_index[image_id]
            )
        ordered_ids = sorted(
            selected_ids,
            key=lambda value: (0, str(value)) if isinstance(value, int) else (1, value),
        )
        coco_images = []
        coco_annotations = []
        role_ids: set[int] = set()
        empty_images = 0
        split_dir = output / output_role
        split_dir.mkdir()
        for native_image_id in ordered_ids:
            state = image_index[native_image_id]
            native_key = source_split, native_image_id
            if native_key in all_native_selected:
                raise AssertionError(
                    f"native image {native_key!r} leaked into {all_native_selected[native_key]!r} "
                    f"and {output_role!r}"
                )
            all_native_selected[native_key] = output_role
            coco_image_id = next_image_id
            next_image_id += 1
            role_ids.add(coco_image_id)
            source_path = dataset.resolve_image(source_split, native_image_id)
            with Image.open(source_path) as opened:
                actual_size = opened.size
            declared_size = state["width"], state["height"]
            if actual_size != declared_size:
                raise ValueError(
                    f"Native image {native_key!r} header {actual_size} != {declared_size}"
                )
            target_name = f"{coco_image_id:08d}_{source_path.name}"
            _link(source_path, split_dir / target_name, link_mode)
            coco_images.append(
                {
                    "id": coco_image_id,
                    "file_name": target_name,
                    "width": state["width"],
                    "height": state["height"],
                }
            )
            source_id = state.get("source_image_id", state["id"])
            mapping_rows.append(
                {
                    "kind": "image",
                    "coco_id": coco_image_id,
                    "source_split": source_split,
                    "partition_role": output_role,
                    "native_id": native_image_id,
                    "source_image_id": source_id,
                    "native_file_name": state["file_name"],
                    "coco_file_name": target_name,
                }
            )
            present = dataset.present_objects(source_split, native_image_id)
            if not present:
                empty_images += 1
            for obj in present:
                coco_annotations.append(
                    {
                        "id": next_annotation_id,
                        "image_id": coco_image_id,
                        "category_id": 1,
                        "bbox": copy.deepcopy(obj["bbox"]),
                        "area": obj["area"],
                        "iscrowd": obj.get("iscrowd", 0),
                        "segmentation": copy.deepcopy(obj["segmentation"]),
                    }
                )
                mapping_rows.append(
                    {
                        "kind": "annotation",
                        "coco_id": next_annotation_id,
                        "coco_image_id": coco_image_id,
                        "source_split": source_split,
                        "partition_role": output_role,
                        "native_image_id": native_image_id,
                        "source_annotation_id": obj["id"],
                        "source_category_id": obj["category_id"],
                        "canonical_category_id": dataset.canonical_category_id(
                            obj["category_id"]
                        ),
                    }
                )
                next_annotation_id += 1
        payload = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [
                {"id": 1, "name": "component", "supercategory": "threat"}
            ],
            "info": {
                "description": "Falcon class-agnostic detector data",
                "source_format": DATASET_FORMAT,
                "image_scope": image_scope,
                "partition_role": output_role,
            },
            "licenses": [],
        }
        with (split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
        result[output_role] = role_ids
        report_roles[output_role] = {
            "source_split": source_split,
            "images": len(coco_images),
            "annotations": len(coco_annotations),
            "empty_images": empty_images,
        }

    _write_jsonl(output / "id_mapping.jsonl", mapping_rows)
    detector_report = {
        "format": "falcon-detector-export-v1",
        "dataset": dataset.metadata_summary(),
        "partition": embedded_partition.name,
        "selections": {role: selection.to_dict() for role, selection in selections.items()},
        "image_scope": image_scope,
        "link_mode": link_mode,
        "paper_stage1_image_composition": image_scope == "base",
        "class_agnostic": True,
        "roles": report_roles,
        "mapping_rows": len(mapping_rows),
    }
    with (output / "detector_export.json").open("w", encoding="utf-8") as stream:
        json.dump(detector_report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def prepare_detector(
    root: Path,
    output: Path,
    valid_percent: int,
    link_mode: str,
    *,
    calibration_percent: int = 5,
    seed: int = 0,
    partition: Path | None = None,
    image_scope: str = "base",
) -> dict[str, set[int]]:
    archive_root = root.expanduser().resolve(strict=True)
    native_manifest = archive_root / "dataset.json"
    if native_manifest.is_file():
        document = _json(native_manifest)
        if "format" in document:
            dataset = open_dataset(archive_root)
            if getattr(dataset, "format", None) != DATASET_FORMAT:
                raise ValueError(f"Unsupported native dataset format {document.get('format')!r}")
            return _native_detector_output(
                dataset,
                output,
                valid_percent=valid_percent,
                calibration_percent=calibration_percent,
                seed=seed,
                link_mode=link_mode,
                partition=partition,
                image_scope=image_scope,
            )
    if partition is not None:
        raise ValueError("--partition is supported only for falcon-x detector data")
    if image_scope != "base":
        raise ValueError("--image-scope is supported only for falcon-x detector data")
    root = resolve_dataset_root(archive_root)
    train_path = _resolve_coco(root, "train", archive_root)
    test_path = _resolve_coco(root, "test", archive_root)
    if train_path == test_path:
        raise ValueError("Train and test COCO annotations resolve to the same file")
    train = _json(train_path)
    test = _json(test_path)
    train_all_ids, train_filenames = _validate_coco(train, "train")
    test_ids, test_filenames = _validate_coco(test, "test")
    _normalize_coco_geometry(train, root / "train" / "images", "train")
    _normalize_coco_geometry(test, root / "test" / "images", "test")
    duplicate_filenames = train_filenames.intersection(test_filenames)
    if duplicate_filenames:
        raise ValueError(
            f"Train/test COCO files share image filenames: {sorted(duplicate_filenames)!r}"
        )

    valid_ids = _validation_ids(train["images"], valid_percent)
    train_ids = train_all_ids.difference(valid_ids)
    if train_ids.intersection(valid_ids) or train_ids.union(valid_ids) != train_all_ids:
        raise AssertionError("Internal train/validation split invariant failed")

    output = output.expanduser().resolve(strict=False)
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Detector output is not a directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"Detector output must be empty: {output}")
    else:
        output.mkdir(parents=True)

    partitions = (
        ("train", train, train_ids, root / "train" / "images"),
        ("valid", train, valid_ids, root / "train" / "images"),
        ("test", test, test_ids, root / "test" / "images"),
    )
    result: dict[str, set[int]] = {}
    for split, payload, image_ids, image_root in partitions:
        split_payload = _class_agnostic(payload, image_ids)
        split_dir = output / split
        split_dir.mkdir(parents=True)
        for image in split_payload["images"]:
            _link(
                image_root / image["file_name"],
                split_dir / image["file_name"],
                link_mode,
            )
        with (split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
            json.dump(split_payload, handle, separators=(",", ":"), allow_nan=False)
        result[split] = set(image_ids)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Falcon-X metadata or RF-DETR data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--data-dir", required=True)
    dataset.add_argument("--label-provenance", default="deterministic_policy_v1")
    tasks = subparsers.add_parser("tasks", help="Add component tasks and repair panoptic metadata")
    tasks.add_argument("--data-dir", required=True)
    tasks.add_argument("--output-dir", required=True)
    tasks.add_argument("--link-mode", choices=("copy", "hardlink"), default="copy")
    partitions = subparsers.add_parser("partitions")
    partitions.add_argument("--data-dir", required=True)
    partitions.add_argument("--output", required=True)
    partitions.add_argument("--validation-percent", type=int, default=5)
    partitions.add_argument("--calibration-percent", type=int, default=5)
    partitions.add_argument("--seed", type=int, default=42)
    detector = subparsers.add_parser("detector")
    detector.add_argument("--data-dir", required=True)
    detector.add_argument("--output-dir", required=True)
    detector.add_argument("--valid-percent", type=int, default=5)
    detector.add_argument("--calibration-percent", type=int, default=5)
    detector.add_argument("--seed", type=int, default=42)
    detector.add_argument("--partition")
    detector.add_argument(
        "--image-scope",
        choices=("base", "all"),
        default="base",
        help="Use paper-profile base images or all original/counterfactual states",
    )
    detector.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="copy",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.data_dir)
    if args.command == "dataset":
        migrate_dataset(root, args.label_provenance)
    elif args.command == "tasks":
        from .task_preparation import prepare_tasks

        result = prepare_tasks(root, Path(args.output_dir), link_mode=args.link_mode)
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "partitions":
        prepare_partitions(
            root,
            Path(args.output),
            validation_percent=args.validation_percent,
            calibration_percent=args.calibration_percent,
            seed=args.seed,
        )
    else:
        prepare_detector(
            root,
            Path(args.output_dir),
            args.valid_percent,
            args.link_mode,
            calibration_percent=args.calibration_percent,
            seed=args.seed,
            partition=Path(args.partition) if args.partition else None,
            image_scope=args.image_scope,
        )


if __name__ == "__main__":
    main()
