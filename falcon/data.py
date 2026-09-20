"""Falcon-X dataset contract and safe record access.

The dataset manifest is the only root-level configuration required by data
preparation, training, and inference. Detector predictions are not part of the
dataset contract because RF-DETR runs live in the same process.

Risk and link targets are authoritative dataset values.  This module validates
and returns them without deriving defaults from component presence.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dataset import FalconXDataset

SCHEMA_VERSION = "falcon-x/v1"
DATASET_FORMAT = "falcon-x"
DATASET_FORMATS = (DATASET_FORMAT,)
CANONICAL_COMPONENT_ORDER = ("detonator", "explosive", "battery")
CANONICAL_LINK_ORDER = (
    ("battery", "detonator"),
    ("battery", "explosive"),
    ("detonator", "explosive"),
)
CANONICAL_CATEGORY_IDS = {
    "detonator": 1,
    "explosive": 2,
    "battery": 3,
}
DIRECTIONAL_RELATION_POLICY_MAXIMUM_OVERLAP = {
    # Retain v1 so already-published Falcon datasets remain readable.
    "falcon-directional-relation-eligibility-v1": 0.5,
    # v2 withholds the five-point gray zone below the old 0.50 boundary.
    "falcon-directional-relation-eligibility-v2": 0.45,
}

PathSource = str | os.PathLike


class DatasetContractError(ValueError):
    """Raised when a manifest, record, or path violates the Falcon-X contract."""


class FalconXDatasetError(DatasetContractError):
    """Raised when falcon-x annotations violate the dataset contract."""

    def __init__(self, message: str, *, code: str = "dataset_contract") -> None:
        super().__init__(message)
        self.code = code


def _annotation_sources(
    annotations: PathSource | Iterable[PathSource] | None,
) -> tuple[PathSource, ...]:
    sources = annotations
    if sources is None:
        return ()
    if isinstance(sources, str | os.PathLike):
        return (sources,)
    return tuple(sources)


def _context(field_name: str, source: str | None) -> str:
    if source:
        return f"{field_name} ({source})"
    return field_name


def _require_mapping(
    value: Any, field_name: str, source: str | None = None
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetContractError(f"{_context(field_name, source)} must be an object")
    return value


def _require_sequence(
    value: Any, field_name: str, source: str | None = None
) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise DatasetContractError(f"{_context(field_name, source)} must be an array")
    return value


def _require_number(value: Any, field_name: str, source: str | None = None) -> Any:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DatasetContractError(
            f"{_context(field_name, source)} must be a finite number"
        )
    if not math.isfinite(float(value)):
        raise DatasetContractError(
            f"{_context(field_name, source)} must be a finite number"
        )
    return value


def _require_probability(value: Any, field_name: str, source: str | None = None) -> Any:
    number = _require_number(value, field_name, source)
    if not 0.0 <= float(number) <= 1.0:
        raise DatasetContractError(f"{_context(field_name, source)} must lie in [0, 1]")
    return value


def _safe_relative_posix(value: Any, field_name: str, source: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetContractError(
            f"{_context(field_name, source)} must be a non-empty POSIX path"
        )
    if "\\" in value or "\x00" in value:
        raise DatasetContractError(
            f"{_context(field_name, source)} must be a canonical POSIX path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise DatasetContractError(f"{_context(field_name, source)} must be relative")
    if any(part in ("", ".", "..") for part in path.parts):
        raise DatasetContractError(
            f"{_context(field_name, source)} contains an unsafe segment"
        )
    normalized = path.as_posix()
    if normalized != value:
        raise DatasetContractError(
            f"{_context(field_name, source)} must be normalized as {normalized!r}"
        )
    return normalized


def _resolve_beneath(
    root: Path, relpath: str, require_exists: bool, field_name: str
) -> Path:
    try:
        root_resolved = root.resolve(strict=require_exists)
        candidate = (root_resolved / Path(*PurePosixPath(relpath).parts)).resolve(
            strict=require_exists
        )
    except OSError as exc:
        raise DatasetContractError(
            f"{field_name} does not exist or cannot be resolved"
        ) from exc
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise DatasetContractError(f"{field_name} escapes dataset root") from exc
    return candidate


@dataclass(frozen=True)
class SplitSpec:
    """Relative paths for one split, rooted at the dataset manifest."""

    name: str
    root: str
    images: str
    states: str
    counterfactual_images: str = "grounding_semantic/images"
    instances: str | None = None
    panoptic_json: str | None = None
    panoptic_dir: str | None = None
    tasks: str | None = None


@dataclass(frozen=True)
class DatasetManifest:
    """Validated Falcon-X root schema."""

    path: Path
    schema_version: str
    category_ids: Mapping[str, int]
    component_order: tuple[str, ...]
    link_order: tuple[tuple[str, str], ...]
    splits: Mapping[str, SplitSpec]

    @property
    def root(self) -> Path:
        return self.path.parent

    def split(self, name: str) -> SplitSpec:
        try:
            return self.splits[name]
        except KeyError as exc:
            raise DatasetContractError(f"unknown dataset split {name!r}") from exc


def _validate_image_prefix(spec: SplitSpec, kind: str, image_relpath: str) -> None:
    if kind not in ("base", "counterfactual"):
        raise DatasetContractError("kind must be 'base' or 'counterfactual'")
    expected_root = spec.images if kind == "base" else spec.counterfactual_images
    rel_parts = PurePosixPath(image_relpath).parts
    prefix_parts = PurePosixPath(expected_root).parts
    if (
        len(rel_parts) <= len(prefix_parts)
        or rel_parts[: len(prefix_parts)] != prefix_parts
    ):
        raise DatasetContractError(
            f"{kind} image path must be explicitly beneath {expected_root}/"
        )


@dataclass(frozen=True)
class ImageRecord:
    """One authoritative base or counterfactual image-state record.

    ``targets`` is retained as loaded.  In particular, ``risk`` and ``links``
    are never inferred, rounded, sorted, or filled from other fields.
    """

    image_id: int | str
    source_image_id: int | str
    kind: str
    file_name: str
    image_relpath: str
    width: int
    height: int
    removed_components: tuple[str, ...]
    present_components: tuple[str, ...]
    missing_components: tuple[str, ...]
    grounding: tuple[Mapping[str, Any], ...]
    candidate_relations: tuple[Mapping[str, Any], ...]
    targets: Mapping[str, Any]
    provenance: Mapping[str, Any]
    _raw: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def risk(self) -> Any:
        return self.targets["risk"]

    @property
    def links(self) -> tuple[Any, ...]:
        return tuple(self.targets["links"])

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the input JSON object without synthesized labels."""

        return copy.deepcopy(dict(self._raw))


RecordSource = ImageRecord | Mapping[str, Any]


def _load_document(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetContractError(
            f"cannot read dataset manifest {path}: {exc}"
        ) from exc

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            value = json.loads(text)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise DatasetContractError(
                    f"PyYAML is required to read {path}; JSON manifests need no optional dependency"
                ) from exc
            value = yaml.safe_load(text)
        else:
            raise DatasetContractError(
                "dataset manifest must end in .json, .yaml, or .yml"
            )
    except DatasetContractError:
        raise
    except Exception as exc:
        raise DatasetContractError(f"invalid dataset manifest {path}: {exc}") from exc
    return _require_mapping(value, "manifest", str(path))


def _parse_split(name: str, value: Any, source: str) -> SplitSpec:
    data = _require_mapping(value, f"splits.{name}", source)
    for required in ("root", "images", "states"):
        if required not in data:
            raise DatasetContractError(
                f"splits.{name}.{required} is required ({source})"
            )

    def path_field(field_name: str, default: str | None = None) -> str | None:
        raw = data.get(field_name, default)
        if raw is None:
            return None
        return _safe_relative_posix(raw, f"splits.{name}.{field_name}", source)

    return SplitSpec(
        name=name,
        root=path_field("root"),  # type: ignore[arg-type]
        images=path_field("images"),  # type: ignore[arg-type]
        states=path_field("states"),  # type: ignore[arg-type]
        counterfactual_images=path_field(
            "counterfactual_images", "grounding_semantic/images"
        ),  # type: ignore[arg-type]
        instances=path_field("instances"),
        panoptic_json=path_field("panoptic_json"),
        panoptic_dir=path_field("panoptic_dir"),
        tasks=path_field("tasks"),
    )


def load_manifest(path: PathSource) -> DatasetManifest:
    """Load and strictly validate a Falcon-X YAML or JSON manifest."""

    manifest_path = Path(path).expanduser().resolve(strict=True)
    data = _load_document(manifest_path)
    source = str(manifest_path)

    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise DatasetContractError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {schema_version!r} ({source})"
        )

    category_ids_raw = _require_mapping(
        data.get("category_ids"), "category_ids", source
    )
    category_ids: dict[str, int] = {}
    for name, value in category_ids_raw.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise DatasetContractError(
                f"category_ids must map names to integer IDs ({source})"
            )
        category_ids[name] = value
    if category_ids != CANONICAL_CATEGORY_IDS:
        raise DatasetContractError(
            f"category_ids must equal canonical mapping {CANONICAL_CATEGORY_IDS!r} ({source})"
        )

    component_raw = _require_sequence(
        data.get("component_order"), "component_order", source
    )
    component_order = tuple(component_raw)
    if component_order != CANONICAL_COMPONENT_ORDER:
        raise DatasetContractError(
            f"component_order must equal {CANONICAL_COMPONENT_ORDER!r} ({source})"
        )

    link_raw = _require_sequence(data.get("link_order"), "link_order", source)
    link_order = tuple(
        tuple(_require_sequence(pair, "link_order[]", source)) for pair in link_raw
    )
    if link_order != CANONICAL_LINK_ORDER:
        raise DatasetContractError(
            f"link_order must equal {CANONICAL_LINK_ORDER!r} ({source})"
        )

    split_raw = _require_mapping(data.get("splits"), "splits", source)
    if not split_raw:
        raise DatasetContractError(f"splits must not be empty ({source})")
    splits: dict[str, SplitSpec] = {}
    for name, value in split_raw.items():
        if not isinstance(name, str) or not name:
            raise DatasetContractError(
                f"split names must be non-empty strings ({source})"
            )
        splits[name] = _parse_split(name, value, source)

    return DatasetManifest(
        path=manifest_path,
        schema_version=schema_version,
        category_ids=dict(category_ids),
        component_order=component_order,
        link_order=link_order,  # type: ignore[arg-type]
        splits=dict(splits),
    )


def _component_tuple(
    value: Any, field_name: str, source: str | None
) -> tuple[str, ...]:
    sequence = _require_sequence(value, field_name, source)
    result = tuple(sequence)
    if any(not isinstance(item, str) for item in result):
        raise DatasetContractError(
            f"{_context(field_name, source)} must contain component names"
        )
    if len(result) != len(set(result)):
        raise DatasetContractError(
            f"{_context(field_name, source)} contains duplicate components"
        )
    unknown = set(result).difference(CANONICAL_COMPONENT_ORDER)
    if unknown:
        raise DatasetContractError(
            f"{_context(field_name, source)} contains unknown components {sorted(unknown)!r}"
        )
    return result


def _valid_identifier(value: Any, field_name: str, source: str | None) -> int | str:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise DatasetContractError(
            f"{_context(field_name, source)} must be an integer or non-empty string"
        )
    if isinstance(value, str) and not value:
        raise DatasetContractError(f"{_context(field_name, source)} must not be empty")
    return value


def parse_image_record(
    value: Mapping[str, Any],
    manifest: DatasetManifest,
    split: str,
    source: str | None = None,
) -> ImageRecord:
    """Validate one JSON object and expose its labels without deriving defaults."""

    data = _require_mapping(value, "record", source)
    split_spec = manifest.split(split)
    required = (
        "image_id",
        "source_image_id",
        "kind",
        "file_name",
        "image_relpath",
        "width",
        "height",
        "removed_components",
        "present_components",
        "missing_components",
        "grounding",
        "candidate_relations",
        "targets",
        "provenance",
    )
    missing_fields = [name for name in required if name not in data]
    if missing_fields:
        raise DatasetContractError(
            "record is missing required fields {}{}".format(
                ", ".join(missing_fields), f" ({source})" if source else ""
            )
        )

    image_id = _valid_identifier(data["image_id"], "image_id", source)
    source_image_id = _valid_identifier(
        data["source_image_id"], "source_image_id", source
    )
    kind = data["kind"]
    if kind not in ("base", "counterfactual"):
        raise DatasetContractError(
            "{} must be 'base' or 'counterfactual'".format(_context("kind", source))
        )

    file_name = _safe_relative_posix(data["file_name"], "file_name", source)
    image_relpath = _safe_relative_posix(data["image_relpath"], "image_relpath", source)
    file_parts = PurePosixPath(file_name).parts
    image_parts = PurePosixPath(image_relpath).parts
    if image_parts[-len(file_parts) :] != file_parts:
        raise DatasetContractError(
            "{} must end with file_name".format(_context("image_relpath", source))
        )
    _validate_image_prefix(split_spec, kind, image_relpath)

    width = data["width"]
    height = data["height"]
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise DatasetContractError(
            "{} must be a positive integer".format(_context("width", source))
        )
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise DatasetContractError(
            "{} must be a positive integer".format(_context("height", source))
        )

    removed = _component_tuple(data["removed_components"], "removed_components", source)
    present = _component_tuple(data["present_components"], "present_components", source)
    missing = _component_tuple(data["missing_components"], "missing_components", source)
    if set(present).intersection(missing) or set(present).union(missing) != set(
        manifest.component_order
    ):
        raise DatasetContractError(
            "present_components and missing_components must partition component_order"
        )
    if not set(removed).issubset(missing):
        raise DatasetContractError(
            "removed_components must be a subset of missing_components"
        )
    if kind == "base":
        if image_id != source_image_id:
            raise DatasetContractError(
                "base records must use their own image_id as source_image_id"
            )
        if removed:
            raise DatasetContractError("base records cannot have removed_components")
    elif not removed:
        raise DatasetContractError(
            "counterfactual records must identify at least one removed component"
        )

    grounding_raw = _require_sequence(data["grounding"], "grounding", source)
    grounding = []
    grounding_ids: set[int | str] = set()
    grounding_source_ids: set[int | str] = set()
    for index, item in enumerate(grounding_raw):
        entry = _require_mapping(item, f"grounding[{index}]", source)
        instance_id = _valid_identifier(
            entry.get("instance_id"),
            f"grounding[{index}].instance_id",
            source,
        )
        if instance_id in grounding_ids:
            raise DatasetContractError(f"grounding[{index}].instance_id is duplicated")
        grounding_ids.add(instance_id)
        source_annotation_id = _valid_identifier(
            entry.get("source_annotation_id", instance_id),
            f"grounding[{index}].source_annotation_id",
            source,
        )
        if source_annotation_id in grounding_source_ids:
            raise DatasetContractError(
                f"grounding[{index}].source_annotation_id is duplicated"
            )
        grounding_source_ids.add(source_annotation_id)
        category = entry.get("category")
        if category not in manifest.component_order:
            raise DatasetContractError(f"grounding[{index}].category is not canonical")
        if category not in present:
            raise DatasetContractError(
                f"grounding[{index}] refers to a missing component"
            )
        bbox = _require_sequence(
            entry.get("bbox_xywh"), f"grounding[{index}].bbox_xywh", source
        )
        if len(bbox) != 4:
            raise DatasetContractError(
                f"grounding[{index}].bbox_xywh must contain four values"
            )
        x, y, box_width, box_height = (
            float(_require_number(value, f"grounding[{index}].bbox_xywh", source))
            for value in bbox
        )
        if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
            raise DatasetContractError(
                f"grounding[{index}].bbox_xywh must have a nonnegative origin and positive size"
            )
        if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
            raise DatasetContractError(
                f"grounding[{index}].bbox_xywh exceeds image dimensions"
            )
        match_policy = entry.get("match_policy")
        if match_policy is not None and match_policy not in (
            "exact_mask_iou",
            "legacy_bbox_iou",
        ):
            raise DatasetContractError(
                f"grounding[{index}].match_policy is unsupported"
            )
        if match_policy == "exact_mask_iou" and "segmentation" not in entry:
            raise DatasetContractError(
                f"grounding[{index}] exact_mask_iou requires immutable segmentation"
            )
        segmentation = entry.get("segmentation")
        if segmentation is not None:
            if isinstance(segmentation, Mapping):
                size = segmentation.get("size")
                if (
                    isinstance(size, str | bytes)
                    or not isinstance(size, Sequence)
                    or tuple(size) != (height, width)
                    or not isinstance(segmentation.get("counts"), str | list)
                ):
                    raise DatasetContractError(
                        f"grounding[{index}].segmentation is malformed COCO RLE"
                    )
            elif isinstance(segmentation, Sequence) and not isinstance(
                segmentation, str | bytes
            ):
                if not segmentation:
                    raise DatasetContractError(
                        f"grounding[{index}].segmentation polygons are empty"
                    )
            else:
                raise DatasetContractError(
                    f"grounding[{index}].segmentation must be polygons or COCO RLE"
                )
        grounding.append(copy.deepcopy(dict(entry)))

    candidate_raw = _require_sequence(
        data["candidate_relations"], "candidate_relations", source
    )
    candidate_relations = []
    candidate_pairs: set[frozenset[int | str]] = set()
    grounding_by_source = {
        entry.get("source_annotation_id", entry["instance_id"]): entry
        for entry in grounding
    }
    for index, item in enumerate(candidate_raw):
        entry = _require_mapping(item, f"candidate_relations[{index}]", source)
        if entry.get("connection_observed") is not False:
            raise DatasetContractError(
                f"candidate_relations[{index}] must not assert a physical connection"
            )
        if (
            entry.get("telemetry_only") is not True
            or entry.get("training_target") is not False
        ):
            raise DatasetContractError(
                f"candidate_relations[{index}] must be marked telemetry-only"
            )
        src_id = _valid_identifier(
            entry.get("src_segment_id"),
            f"candidate_relations[{index}].src_segment_id",
            source,
        )
        dst_id = _valid_identifier(
            entry.get("dst_segment_id"),
            f"candidate_relations[{index}].dst_segment_id",
            source,
        )
        pair = frozenset((src_id, dst_id))
        if src_id == dst_id or pair in candidate_pairs:
            raise DatasetContractError(
                f"candidate_relations[{index}] is a duplicate or self relation"
            )
        candidate_pairs.add(pair)
        if src_id not in grounding_by_source or dst_id not in grounding_by_source:
            raise DatasetContractError(
                f"candidate_relations[{index}] refers to a missing grounded instance"
            )
        if entry.get("src_component") != grounding_by_source[src_id]["category"]:
            raise DatasetContractError(
                f"candidate_relations[{index}].src_component disagrees with grounding"
            )
        if entry.get("dst_component") != grounding_by_source[dst_id]["category"]:
            raise DatasetContractError(
                f"candidate_relations[{index}].dst_component disagrees with grounding"
            )
        directional_policy = entry.get("directional_relation_policy_version")
        directional_maximum = (
            DIRECTIONAL_RELATION_POLICY_MAXIMUM_OVERLAP.get(directional_policy)
            if isinstance(directional_policy, str)
            else None
        )
        if directional_policy is not None and directional_maximum is None:
            raise DatasetContractError(
                f"candidate_relations[{index}] has unknown directional policy"
            )
        nullable_directional_axes = directional_maximum is not None
        horizontal_relation = entry.get("horizontal_relation")
        vertical_relation = entry.get("vertical_relation")
        if horizontal_relation not in (
            "left_of",
            "right_of",
            "horizontally_aligned",
            "left of",
            "right of",
            "horizontally aligned",
        ) and not (nullable_directional_axes and horizontal_relation is None):
            raise DatasetContractError(
                f"candidate_relations[{index}].horizontal_relation is invalid"
            )
        if vertical_relation not in (
            "above",
            "below",
            "vertically_aligned",
            "vertically aligned",
        ) and not (nullable_directional_axes and vertical_relation is None):
            raise DatasetContractError(
                f"candidate_relations[{index}].vertical_relation is invalid"
            )
        if nullable_directional_axes:
            if entry.get("maximum_axis_projection_overlap") != directional_maximum:
                raise DatasetContractError(
                    f"candidate_relations[{index}] has invalid directional policy"
                )
            for axis, relation in (
                ("horizontal", horizontal_relation),
                ("vertical", vertical_relation),
            ):
                eligible = entry.get(f"{axis}_relation_eligible")
                claim_id = entry.get(f"{axis}_relation_claim_id")
                if (
                    not isinstance(eligible, bool)
                    or (relation is not None) is not eligible
                    or (isinstance(claim_id, str) and bool(claim_id)) is not eligible
                    or (not eligible and claim_id is not None)
                ):
                    raise DatasetContractError(
                        f"candidate_relations[{index}] has inconsistent {axis} eligibility"
                    )
        candidate_relations.append(copy.deepcopy(dict(entry)))

    targets_raw = _require_mapping(data["targets"], "targets", source)
    missing_targets = [
        name
        for name in ("presence", "completeness", "risk", "links")
        if name not in targets_raw
    ]
    if missing_targets:
        raise DatasetContractError(
            "targets is missing {}; risk and links have no fallback".format(
                ", ".join(missing_targets)
            )
        )
    presence_raw = _require_sequence(
        targets_raw["presence"], "targets.presence", source
    )
    presence = tuple(presence_raw)
    if len(presence) != len(manifest.component_order) or any(
        value not in (0, 1) for value in presence
    ):
        raise DatasetContractError("targets.presence must be a canonical 0/1 vector")
    expected_presence = tuple(
        1 if name in present else 0 for name in manifest.component_order
    )
    if presence != expected_presence:
        raise DatasetContractError("targets.presence disagrees with present_components")
    # Missing source labels remain explicit JSON null values. Consumers mask
    # them; this loader never replaces them with derived values.
    if targets_raw["completeness"] is not None:
        _require_probability(
            targets_raw["completeness"], "targets.completeness", source
        )
    if targets_raw["risk"] is not None:
        _require_probability(targets_raw["risk"], "targets.risk", source)
    links_raw = _require_sequence(targets_raw["links"], "targets.links", source)
    if len(links_raw) != len(manifest.link_order):
        raise DatasetContractError("targets.links must follow canonical link_order")
    for index, value_item in enumerate(links_raw):
        if value_item is not None:
            _require_probability(value_item, f"targets.links[{index}]", source)

    provenance_raw = _require_mapping(data["provenance"], "provenance", source)
    for label in ("risk", "links"):
        provenance_entry = _require_mapping(
            provenance_raw.get(label), f"provenance.{label}", source
        )
        provenance_source = provenance_entry.get("source")
        if not isinstance(provenance_source, str) or not provenance_source:
            raise DatasetContractError(
                f"provenance.{label}.source must be a non-empty string"
            )

    raw_copy = copy.deepcopy(dict(data))
    return ImageRecord(
        image_id=image_id,
        source_image_id=source_image_id,
        kind=kind,
        file_name=file_name,
        image_relpath=image_relpath,
        width=width,
        height=height,
        removed_components=removed,
        present_components=present,
        missing_components=missing,
        grounding=tuple(grounding),
        candidate_relations=tuple(candidate_relations),
        targets=copy.deepcopy(dict(targets_raw)),
        provenance=copy.deepcopy(dict(provenance_raw)),
        _raw=raw_copy,
    )


def _split_root(manifest: DatasetManifest, split: str, require_exists: bool) -> Path:
    spec = manifest.split(split)
    return _resolve_beneath(manifest.root, spec.root, require_exists, "split root")


def load_records(manifest: DatasetManifest, split: str) -> tuple[ImageRecord, ...]:
    """Load and cross-check all records for one split."""

    spec = manifest.split(split)
    split_root = _split_root(manifest, split, require_exists=True)
    states_path = _resolve_beneath(split_root, spec.states, True, "states path")
    if not states_path.is_file():
        raise DatasetContractError(f"states path is not a file: {states_path}")

    records = []
    identifiers: dict[int | str, ImageRecord] = {}
    relpaths = set()
    try:
        handle = states_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise DatasetContractError(
            f"cannot read states file {states_path}: {exc}"
        ) from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source = f"{states_path}:{line_number}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetContractError(
                    f"invalid JSON object at {source}: {exc}"
                ) from exc
            record = parse_image_record(raw, manifest, split, source=source)
            if record.image_id in identifiers:
                raise DatasetContractError(
                    f"duplicate image_id {record.image_id!r} at {source}"
                )
            if record.image_relpath in relpaths:
                raise DatasetContractError(
                    f"duplicate image_relpath {record.image_relpath!r} at {source}"
                )
            identifiers[record.image_id] = record
            relpaths.add(record.image_relpath)
            records.append(record)

    base_ids = {record.image_id for record in records if record.kind == "base"}
    for record in records:
        if record.kind == "counterfactual" and record.source_image_id not in base_ids:
            raise DatasetContractError(
                f"counterfactual {record.image_id!r} references missing base image "
                f"{record.source_image_id!r}"
            )
    return tuple(records)


def resolve_image_path(
    manifest: DatasetManifest,
    split: str,
    record: RecordSource,
    require_exists: bool = True,
) -> Path:
    """Resolve the record's explicit path without guessing or directory fallback."""

    spec = manifest.split(split)
    if not isinstance(record, ImageRecord):
        record = parse_image_record(record, manifest, split)
    else:
        _safe_relative_posix(record.image_relpath, "image_relpath")
        file_name = _safe_relative_posix(record.file_name, "file_name")
        file_parts = PurePosixPath(file_name).parts
        image_parts = PurePosixPath(record.image_relpath).parts
        if image_parts[-len(file_parts) :] != file_parts:
            raise DatasetContractError("image_relpath must end with file_name")
        _validate_image_prefix(spec, record.kind, record.image_relpath)

    split_root = _split_root(manifest, split, require_exists=require_exists)
    path = _resolve_beneath(
        split_root, record.image_relpath, require_exists, "image_relpath"
    )
    if require_exists and not path.is_file():
        raise DatasetContractError(f"image path is not a file: {path}")
    return path


class FalconDataset:
    """Small public facade shared by preparation, training, and inference."""

    def __init__(self, manifest: DatasetManifest):
        self.manifest = manifest
        self._records: dict[str, tuple[ImageRecord, ...]] = {}

    @classmethod
    def open(
        cls,
        root_or_manifest: PathSource,
        *,
        annotations: PathSource | Iterable[PathSource] | None = None,
    ) -> FalconDataset | FalconXDataset:
        return open_dataset(root_or_manifest, annotations=annotations)

    def split(self, name: str) -> SplitSpec:
        return self.manifest.split(name)

    def split_root(self, name: str) -> Path:
        return _split_root(self.manifest, name, require_exists=True)

    def records(self, name: str, refresh: bool = False) -> tuple[ImageRecord, ...]:
        if refresh or name not in self._records:
            self._records[name] = load_records(self.manifest, name)
        return self._records[name]

    def resolve_image(
        self,
        name: str,
        record: RecordSource,
        require_exists: bool = True,
    ) -> Path:
        return resolve_image_path(
            self.manifest, name, record, require_exists=require_exists
        )


def open_dataset(
    root_or_manifest: PathSource,
    *,
    annotations: PathSource | Iterable[PathSource] | None = None,
) -> FalconDataset | FalconXDataset:
    """Open falcon-x data with optional reviewed annotation files."""

    sources = _annotation_sources(annotations)

    path = Path(root_or_manifest).expanduser()
    if path.is_dir():
        candidates = [
            path / "dataset.yaml",
            path / "dataset.yml",
            path / "dataset.json",
        ]
        found = [candidate for candidate in candidates if candidate.is_file()]
        if len(found) != 1:
            raise DatasetContractError(
                "dataset root must contain exactly one of dataset.yaml, "
                "dataset.yml, or dataset.json"
            )
        path = found[0]
    resolved = path.resolve(strict=True)
    document = _load_document(resolved)
    declares_legacy = "schema_version" in document
    declares_native = "format" in document
    if declares_legacy and declares_native:
        raise DatasetContractError(
            "dataset manifest cannot mix schema_version and format markers"
        )
    if declares_native:
        value = document.get("format")
        if value not in DATASET_FORMATS:
            raise DatasetContractError(f"unsupported dataset format {value!r}")
        from .dataset import FalconXDataset

        return FalconXDataset(resolved, annotations=sources)
    if not declares_legacy:
        raise DatasetContractError(
            "dataset manifest must declare schema_version or format"
        )
    if sources:
        raise DatasetContractError(
            "additional annotations require the falcon-x dataset format"
        )
    manifest = load_manifest(resolved)
    return FalconDataset(manifest)


__all__ = [
    "CANONICAL_CATEGORY_IDS",
    "CANONICAL_COMPONENT_ORDER",
    "CANONICAL_LINK_ORDER",
    "DATASET_FORMAT",
    "DATASET_FORMATS",
    "DIRECTIONAL_RELATION_POLICY_MAXIMUM_OVERLAP",
    "DatasetContractError",
    "DatasetManifest",
    "FalconDataset",
    "FalconXDatasetError",
    "ImageRecord",
    "SCHEMA_VERSION",
    "SplitSpec",
    "load_manifest",
    "load_records",
    "open_dataset",
    "parse_image_record",
    "resolve_image_path",
]
