"""Prepare component tasks and repair metadata without changing image assets."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment

from .data import DATASET_FORMAT, open_dataset
from .dataset import FalconXDataset
from .evaluation.metrics import checked_segments, panoptic_ids
from .tasks import REQUIRED_COMPONENTS, component_membership_targets

ADDED_FAMILIES = (
    "missing_component_identification",
    "functional_completeness",
    "referring_functional_grounding",
    "referring_panoptic_segmentation",
)
_MIN_SEGMENT_CONTAINMENT = 0.90
_MIN_CATEGORY_ASSIGNMENT_MARGIN = 0.25
PANOPTIC_BINDING_POLICY = {
    "version": "instance-overlap-v1",
    "assignment": "one-to-one maximum total segment/object IoU",
    "minimum_segment_containment": _MIN_SEGMENT_CONTAINMENT,
    "boundary_tolerance_chebyshev_pixels": 1,
    "minimum_category_assignment_margin": _MIN_CATEGORY_ASSIGNMENT_MARGIN,
    "low_direct_containment_threshold": 0.85,
    "low_direct_containment_rule": (
        "segment count equals object count and every other assigned pair has "
        "at least 0.85 direct containment"
    ),
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _write_row(stream: Any, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _copy_checked(source: Path, destination: Path, mode: str) -> tuple[int, ...]:
    """Copy unchanged bytes and return the source's pre-copy file identity."""

    before = FalconXDataset._identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if mode == "hardlink":
        os.link(source, destination)
        expected = FalconXDataset._identity(source)
        # Creating a hardlink changes ctime, but not the content or inode.
        if expected[:4] != before[:4]:
            raise ValueError(f"Source file changed during preparation: {source}")
    else:
        shutil.copyfile(source, destination)
        expected = before
    destination_before = FalconXDataset._identity(destination)
    with source.open("rb") as original, destination.open("rb") as copied:
        while True:
            chunk = original.read(1024 * 1024)
            if chunk != copied.read(1024 * 1024):
                raise ValueError(f"Copied file differs from input: {source}")
            if not chunk:
                break
    if (
        FalconXDataset._identity(source) != expected
        or FalconXDataset._identity(destination) != destination_before
    ):
        raise ValueError(f"File changed during preparation: {source}")
    return before


def _repair_panoptic(
    dataset: FalconXDataset, split: str, image: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[int, ...]]:
    """Bind visible segments to instance categories, allowing mutual occlusion.

    COCO polygons and the existing PNG use slightly different boundaries. A
    one-to-one maximum-IoU assignment handles occlusion without reusing an
    instance for multiple segments. Each category must have a clear assignment
    margin and at least 90% containment, allowing a fixed one-pixel polygon
    boundary tolerance. Low-overlap remnants additionally require a complete
    one-to-one assignment whose other pairs have strong direct overlap.
    """
    path, metadata = dataset.resolve_panoptic(split, image["id"])
    before = FalconXDataset._identity(path)
    ids = panoptic_ids(path, image["height"], image["width"], "source")
    if FalconXDataset._identity(path) != before:
        raise ValueError(f"Panoptic PNG changed while decoding: {path}")
    objects = dataset.present_objects(split, image["id"])
    masks = [dataset.decode_object_mask(split, obj) for obj in objects]
    segment_ids = [int(value) for value in np.unique(ids) if value]
    if len(segment_ids) > len(objects) or not segment_ids or not objects:
        raise ValueError("Panoptic segments cannot be matched one-to-one to current objects")
    segments = [ids == segment_id for segment_id in segment_ids]
    areas = np.asarray([int(mask.sum()) for mask in segments])
    object_areas = np.asarray([int(mask.sum()) for mask in masks])
    intersections = np.asarray(
        [[int(np.count_nonzero(segment & mask)) for mask in masks] for segment in segments]
    )
    iou = intersections / (areas[:, None] + object_areas[None, :] - intersections)
    row_ids, object_ids = linear_sum_assignment(iou, maximize=True)
    assignment = dict(zip(row_ids.tolist(), object_ids.tolist(), strict=True))
    score = float(iou[row_ids, object_ids].sum())
    direct = {row: float(intersections[row, column] / areas[row])
              for row, column in assignment.items()}
    repaired = []
    bindings = []
    for row, segment_id in enumerate(segment_ids):
        column = assignment[row]
        category = objects[column]["category_id"]
        alternative = iou.copy()
        # Alternative assignments within the same category are irrelevant to a
        # category-conditioned panoptic query; physical instance IDs stay intact.
        for candidate, obj in enumerate(objects):
            if obj["category_id"] == category:
                alternative[row, candidate] = -1e6
        alternative_rows, alternative_columns = linear_sum_assignment(alternative, maximize=True)
        alternative_score = float(alternative[alternative_rows, alternative_columns].sum())
        margin = score - alternative_score
        containment = direct[row]
        boundary_containment = containment
        boundary_only = containment < _MIN_SEGMENT_CONTAINMENT
        if boundary_only:
            expanded = binary_dilation(masks[column], structure=np.ones((3, 3), dtype=bool))
            boundary_containment = float(np.count_nonzero(segments[row] & expanded) / areas[row])
        anchored = containment >= 0.85 or (
            len(segment_ids) == len(objects)
            and all(value >= 0.85 for other, value in direct.items() if other != row)
        )
        if (
            boundary_containment < _MIN_SEGMENT_CONTAINMENT
            or margin < _MIN_CATEGORY_ASSIGNMENT_MARGIN
            or not anchored
        ):
            raise ValueError(
                f"Ambiguous segment {segment_id}: direct_containment={containment:.6f}, "
                f"boundary_containment={boundary_containment:.6f}, anchored={anchored}, "
                f"category_assignment_margin={margin:.6f}, area={int(areas[row])}"
            )
        ys, xs = np.nonzero(segments[row])
        repaired.append(
            {
                "id": segment_id,
                "category_id": category,
                "iscrowd": 0,
                "area": int(areas[row]),
                "bbox": [
                    int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
                ],
            }
        )
        bindings.append(
            {
                "segment_id": segment_id,
                "annotation_id": objects[column]["id"],
                "direct_containment": containment,
                "boundary_containment": boundary_containment,
                "boundary_only": boundary_only,
                "iou": float(iou[row, column]),
                "category_assignment_margin": margin,
            }
        )
    checked_segments(ids, repaired, {obj["category_id"] for obj in objects}, "repaired")
    return {"file_name": metadata["file_name"], "segments_info": repaired}, bindings, before


def _new_tasks(dataset: FalconXDataset, split: str, image: dict[str, Any]):
    objects = dataset.present_objects(split, image["id"])
    missing, completeness = component_membership_targets(
        [dataset.category_name(obj["category_id"]) for obj in objects]
    )
    prefix = f"{split}:components-v1:{json.dumps(image['id'], separators=(',', ':'))}"
    common = {"image_id": image["id"]}
    yield {
        **common,
        "id": f"{prefix}:missing",
        "family": "missing_component_identification",
        "prompt": (
            "Which required component categories are absent from this image? "
            "The required categories are detonator, explosive, and battery. "
            "Return a JSON array of category names, or [] when none are missing."
        ),
        "answer": json.dumps(missing, separators=(",", ":")),
        "missing_categories": missing,
    }
    yield {
        **common,
        "id": f"{prefix}:complete",
        "family": "functional_completeness",
        "prompt": (
            "Does this image contain at least one detonator, one explosive, and one battery? "
            "Return 1 if all three categories are present, otherwise 0."
        ),
        "answer": str(completeness),
        "completeness": completeness,
    }
    yield {
        **common,
        "id": f"{prefix}:functional-grounding",
        "family": "referring_functional_grounding",
        "prompt": (
            "Segment all present components belonging to the battery, detonator, and "
            "explosive functional component set. Include incomplete sets."
        ),
        "answer": "<SEG>",
        "target_annotation_ids": [obj["id"] for obj in objects],
    }
    if "source_image_id" not in image:
        categories = {row["name"]: row["id"] for row in dataset.manifest["categories"]}
        for name in REQUIRED_COMPONENTS:
            category_id = categories[name]
            yield {
                **common,
                "id": f"{prefix}:referring-panoptic:{name}",
                "family": "referring_panoptic_segmentation",
                "prompt": (
                    f"Produce a panoptic segmentation of all {name} instances in this image. "
                    "Keep instances separate and treat everything else as background."
                ),
                "answer": "<PANOPTIC>",
                "category_id": category_id,
                "target_segment_ids": [
                    segment["id"] for segment in image["panoptic"]["segments_info"]
                    if segment["category_id"] == category_id
                ],
            }


def prepare_tasks(data_dir: Path, output_dir: Path, *, link_mode: str = "copy") -> dict[str, Any]:
    """Write a new falcon-x package; inputs and all image bytes remain unchanged."""
    if link_mode not in ("copy", "hardlink"):
        raise ValueError("link_mode must be copy or hardlink")
    dataset = open_dataset(data_dir)
    if not isinstance(dataset, FalconXDataset):
        raise ValueError("Task preparation requires a native falcon-x dataset")
    output = output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must not exist: {output}")
    output = output.resolve()
    if output.is_relative_to(dataset.root) or dataset.root.is_relative_to(output):
        raise ValueError("Input and output dataset directories must be separate")
    manifest = dataset.manifest
    for split, spec in manifest["splits"].items():
        if set(spec["families"]) & set(ADDED_FAMILIES):
            raise ValueError(f"Split {split} already contains added task families")
        validation = dataset.validate(split)
        if not validation["ok"]:
            raise ValueError(f"Input metadata is invalid: {validation['errors'][:3]}")
    source_metadata = dataset.metadata_summary()
    # Reject links, including unreferenced members, before copying the package.
    source_files = []
    for parent, directories, files in os.walk(dataset.root, followlinks=False):
        for name in (*directories, *files):
            if (Path(parent) / name).is_symlink():
                raise ValueError(f"Input package contains a symlink: {Path(parent) / name}")
        for name in files:
            path = Path(parent) / name
            if not path.is_file():
                raise ValueError(f"Input package member is not a regular file: {path}")
            source_files.append(path)
    repaired_images: dict[str, list[dict[str, Any]]] = {}
    panoptic_bindings = []
    panoptic_files = {}
    issues = []
    for split in manifest["splits"]:
        images = list(dataset.images(split))
        for image in images:
            if "source_image_id" in image:
                continue
            try:
                image["panoptic"], bindings, identity = _repair_panoptic(dataset, split, image)
                relative = Path(split) / image["panoptic"]["file_name"]
                panoptic_files[relative] = identity
                panoptic_bindings.append({"split": split, "image_id": image["id"],
                                          "file_name": relative.as_posix(), "segments": bindings})
            except ValueError as exc:
                issues.append({"split": split, "image_id": image["id"], "error": str(exc)})
        repaired_images[split] = images
        print(f"[{split}] checked panoptic category bindings", flush=True)
    dataset._assert_metadata_unchanged()
    output.mkdir(parents=True, exist_ok=False)
    if issues:
        report = {"complete": False, "source": str(dataset.root), "dataset": source_metadata,
                  "panoptic_binding_policy": PANOPTIC_BINDING_POLICY, "errors": issues}
        _write_json(output / "preparation_report.json", report)
        raise ValueError(
            f"Panoptic category bindings require review for {len(issues)} image(s); "
            f"see {output / 'preparation_report.json'}. No dataset was published."
        )
    replaced = {Path("dataset.json"), Path("README.md")}
    replaced.update(Path(spec[field]) for spec in manifest["splits"].values()
                    for field in ("images", "tasks"))
    copied = 0
    for source in sorted(source_files):
        relative = source.relative_to(dataset.root)
        if relative in replaced:
            # These small originals record precisely which metadata was repaired.
            destination = output / "provenance" / "original" / relative
            _copy_checked(source, destination, "copy")
            continue
        mode = "copy" if source.suffix in (".json", ".jsonl", ".md") else link_mode
        identity = _copy_checked(source, output / relative, mode)
        if relative in panoptic_files and panoptic_files[relative] != identity:
            raise ValueError(f"Panoptic PNG changed after category binding: {source}")
        copied += 1
    for split, spec in manifest["splits"].items():
        image_path = output / spec["images"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with image_path.open("x", encoding="utf-8") as stream:
            for image in repaired_images[split]:
                _write_row(stream, image)
        families: Counter[str] = Counter()
        identifiers = set()
        task_path = output / spec["tasks"]
        task_path.parent.mkdir(parents=True, exist_ok=True)
        with task_path.open("x", encoding="utf-8") as stream:
            for task in dataset.iter_tasks(split):
                identifiers.add(task["id"])
                families[task["family"]] += 1
                _write_row(stream, task)
            for image in repaired_images[split]:
                for task in _new_tasks(dataset, split, image):
                    if task["id"] in identifiers:
                        raise ValueError(f"Duplicate generated task ID: {task['id']}")
                    identifiers.add(task["id"])
                    families[task["family"]] += 1
                    _write_row(stream, task)
        spec["families"] = dict(sorted(families.items()))
        spec["tasks_count"] = sum(families.values())
        print(f"[{split}] prepared {spec['tasks_count']} tasks", flush=True)
    manifest["format"] = DATASET_FORMAT
    if "train_family_weights" in manifest:
        # Retain old relative weights; allocate equal initial weight to each new family.
        weights = {name: float(value) for name, value in manifest["train_family_weights"].items()}
        weights.update({family: 10.0 for family in ADDED_FAMILIES})
        total = sum(weights.values())
        manifest["train_family_weights"] = {name: 100 * value / total
                                            for name, value in weights.items()}
    manifest["task_definitions"] = {
        "version": "component-tasks-v1",
        "required_categories": list(REQUIRED_COMPONENTS),
        "functional_completeness": "1 iff each required category is present; otherwise 0",
        "referring_functional_grounding": "union of all present required-component masks",
        "referring_panoptic_segmentation": "category-conditioned visible segments, separate IDs",
        "source": str(dataset.root),
        "panoptic_binding_policy": PANOPTIC_BINDING_POLICY,
    }
    dataset._assert_metadata_unchanged()
    _write_json(output / "provenance" / "panoptic_bindings.json", panoptic_bindings)
    _write_json(output / "dataset.json", manifest)
    prepared = open_dataset(output)
    validation = {}
    for split in manifest["splits"]:
        print(f"[{split}] validating all task targets and unchanged image assets", flush=True)
        validation[split] = prepared.validate(split, full=True)
    report = {
        "complete": all(value["ok"] for value in validation.values()),
        "source": str(dataset.root),
        "source_dataset": source_metadata,
        "output": str(output),
        "output_dataset": prepared.metadata_summary(),
        "copied_files": copied,
        "verified_image_files": sum(len(images) for images in repaired_images.values()),
        "verified_panoptic_files": len(panoptic_files),
        "images_byte_for_byte_unchanged": True,
        "panoptic_binding_policy": PANOPTIC_BINDING_POLICY,
        "boundary_tolerance_segments": sum(
            segment["boundary_only"] for image in panoptic_bindings for segment in image["segments"]
        ),
        "validation": validation,
    }
    _write_json(output / "preparation_report.json", report)
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(
            "# falcon-x\n\n"
            "Original images, counterfactual images, PNG masks, and existing tasks are retained.\n"
            "Component absence and completeness use current object membership. Functional\n"
            "grounding selects all present components; it does not assert physical connectivity.\n"
            "Referring-panoptic queries select visible instances of a named category.\n\n"
            "Panoptic categories were matched to instance masks; areas and boxes were recomputed\n"
            "from unchanged PNG pixels. See preparation_report.json and provenance/.\n"
        )
    if not report["complete"]:
        raise ValueError(f"Output validation failed; see {output / 'preparation_report.json'}")
    return report
