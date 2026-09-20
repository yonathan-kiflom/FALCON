"""Pure CPU metric primitives for task-level evaluation."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

from ..artifacts import panoptic_ids_from_image
from .schemas import ModelOutputError


def normalize_text(value: str) -> str:
    """Conservative normalization used for exact-match diagnostics."""

    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(' \t\r\n.,;:!?"')


def _unique_label(text: str, patterns: Mapping[str, Sequence[str]]) -> str | None:
    normalized = normalize_text(text).replace("_", " ").replace("-", " ")
    found: set[str] = set()
    for label, aliases in patterns.items():
        if any(
            re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized)
            for alias in aliases
        ):
            found.add(label)
    return next(iter(found)) if len(found) == 1 else None


_UNCERTAIN_OR_NEGATED = re.compile(
    r"(?<!\w)(?:not|isn['’]?t|aren['’]?t|maybe|perhaps|possibly|probably|"
    r"uncertain|unsure|might|could|seems?|appears?)(?!\w)"
)


_QUESTION_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "width_height": {
        "wider": (
            "wider than tall",
            "wider than it is tall",
            "width greater than height",
        ),
        "taller": (
            "taller than wide",
            "taller than it is wide",
            "height greater than width",
        ),
        "equal": (
            "equal width and height",
            "same width and height",
            "equally wide and tall",
        ),
    },
    "absolute_position": {
        "top-left": ("top left", "upper left"),
        "top-center": ("top center", "upper center", "top middle", "upper middle"),
        "top-right": ("top right", "upper right"),
        "middle-left": ("middle left", "center left"),
        "center": ("center", "middle center", "central"),
        "middle-right": ("middle right", "center right"),
        "bottom-left": ("bottom left", "lower left"),
        "bottom-center": (
            "bottom center",
            "lower center",
            "bottom middle",
            "lower middle",
        ),
        "bottom-right": ("bottom right", "lower right"),
    },
    "horizontal_relation": {
        "left": ("left of", "to the left"),
        "right": ("right of", "to the right"),
        "aligned": ("horizontally aligned", "same horizontal level"),
    },
    "vertical_relation": {
        "above": ("above", "over"),
        "below": ("below", "under"),
        "aligned": ("same vertical level", "vertically aligned"),
    },
    "relative_area": {
        "more": (
            "more image area",
            "larger image area",
            "greater image area",
            "larger area",
        ),
        "less": ("less image area", "smaller image area", "smaller area"),
        "equal": ("same image area", "equal image area", "same area"),
    },
}


def parse_vqa_label(question_type: str, answer: str) -> str | None:
    patterns = _QUESTION_PATTERNS.get(question_type)
    if patterns is None or not isinstance(answer, str):
        return None
    normalized = normalize_text(answer).replace("_", " ").replace("-", " ")
    if _UNCERTAIN_OR_NEGATED.search(normalized):
        return None
    if question_type == "absolute_position":
        # ``center`` is nested in top-center/bottom-center.  Resolve the eight
        # qualified cells first and use the generic center label only if none
        # of those phrases occurs.
        specific = {key: value for key, value in patterns.items() if key != "center"}
        label = _unique_label(normalized, specific)
        if label is not None:
            residual = normalized
            for aliases in specific.values():
                for alias in aliases:
                    residual = re.sub(
                        rf"(?<!\w){re.escape(alias)}(?!\w)", " ", residual
                    )
            if _unique_label(residual, {"center": patterns["center"]}) is not None:
                return None
            return label
        if any(
            candidate is not None
            for candidate in (
                _unique_label(normalized, {key: value})
                for key, value in specific.items()
            )
        ):
            return None
        return _unique_label(normalized, {"center": patterns["center"]})
    return _unique_label(normalized, patterns)


_YES = frozenset({"yes", "yes it is", "present", "true"})
_NO = frozenset({"no", "no it is not", "absent", "false"})


def parse_yes_no(answer: str) -> bool | None:
    if not isinstance(answer, str):
        return None
    normalized = normalize_text(answer)
    if normalized in _YES:
        return True
    if normalized in _NO:
        return False
    return None


def parse_category_set(answer: str, categories: set[str]) -> frozenset[str] | None:
    """Parse a JSON set represented by unique canonical category names."""

    if not isinstance(answer, str):
        return None
    try:
        values = json.loads(answer)
    except (json.JSONDecodeError, ValueError):
        return None
    if (
        not isinstance(values, list)
        or any(
            not isinstance(value, str) or value not in categories for value in values
        )
        or len(values) != len(set(values))
    ):
        return None
    return frozenset(values)


def parse_unit_interval(answer: str) -> float | None:
    """Accept a finite JSON number in [0, 1], not prose or a boolean."""

    if not isinstance(answer, str):
        return None
    try:
        value = json.loads(answer)
    except (json.JSONDecodeError, ValueError):
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= 1
        or not math.isfinite(value)
    ):
        return None
    return float(value)


def binary_classification(
    labels: Sequence[bool], predictions: Sequence[bool]
) -> dict[str, float]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("binary metric inputs must have equal non-zero length")
    tp = sum(
        label and prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    pairs = list(zip(labels, predictions, strict=True))
    tn = sum(not label and not prediction for label, prediction in pairs)
    fp = sum(not label and prediction for label, prediction in pairs)
    fn = sum(label and not prediction for label, prediction in pairs)

    def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if denominator == 0 else 2 * true_positive / denominator

    positive_f1 = f1(tp, fp, fn)
    negative_f1 = f1(tn, fn, fp)
    return {
        "accuracy": (tp + tn) / len(labels),
        "positive_f1": positive_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2,
    }


def _as_binary_mask(decoded: Any, height: int, width: int, name: str) -> np.ndarray:
    array = np.asarray(decoded)
    if array.ndim == 3:
        array = np.any(array, axis=2)
    if array.shape != (height, width):
        raise ModelOutputError(
            f"{name} mask has shape {array.shape}, expected {(height, width)}"
        )
    return array.astype(bool, copy=False)


def decode_coco_mask(
    segmentation: Any, height: int, width: int, name: str
) -> np.ndarray:
    """Decode COCO polygons or RLE and enforce the task image dimensions."""

    try:
        if isinstance(segmentation, Mapping):
            size = segmentation.get("size")
            counts = segmentation.get("counts")
            if size != [height, width] and tuple(size or ()) != (height, width):
                raise ModelOutputError(
                    f"{name} RLE size {size!r} does not match {(height, width)}"
                )
            encoded = (
                mask_util.frPyObjects(dict(segmentation), height, width)
                if isinstance(counts, list)
                else dict(segmentation)
            )
            return _as_binary_mask(mask_util.decode(encoded), height, width, name)
        if isinstance(segmentation, Sequence) and not isinstance(
            segmentation, str | bytes
        ):
            if not segmentation:
                return np.zeros((height, width), dtype=bool)
            encoded = mask_util.frPyObjects(list(segmentation), height, width)
            return _as_binary_mask(mask_util.decode(encoded), height, width, name)
    except ModelOutputError:
        raise
    except Exception as exc:
        raise ModelOutputError(f"cannot decode {name} COCO mask: {exc}") from exc
    raise ModelOutputError(f"{name} must be COCO polygon or RLE data")


def union_object_masks(
    objects: Sequence[Mapping[str, Any]], height: int, width: int
) -> np.ndarray:
    result = np.zeros((height, width), dtype=bool)
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping) or "segmentation" not in obj:
            raise ValueError(f"target object {index} lacks segmentation")
        result |= decode_coco_mask(
            obj["segmentation"], height, width, f"target object {index}"
        )
    return result


def union_prediction_regions(regions: Any, height: int, width: int) -> np.ndarray:
    if isinstance(regions, str | bytes) or not isinstance(regions, Sequence):
        raise ModelOutputError("regions must be an array of COCO RLE objects")
    result = np.zeros((height, width), dtype=bool)
    for index, region in enumerate(regions):
        if not isinstance(region, Mapping):
            raise ModelOutputError(f"regions[{index}] must be a COCO RLE object")
        # RLE preserves the exact rasterization of saved predictions.
        if set(region) != {"size", "counts"}:
            raise ModelOutputError(
                f"regions[{index}] must contain only size and counts"
            )
        result |= decode_coco_mask(region, height, width, f"regions[{index}]")
    return result


def mask_intersection_union(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[int, int]:
    return int(np.count_nonzero(prediction & target)), int(
        np.count_nonzero(prediction | target)
    )


def iou_threshold_accuracies(values: Sequence[float]) -> dict[str, float]:
    """Return the fraction of queries meeting each IoU threshold.

    Callers include invalid queries with IoU zero. Empty support has no accuracy
    and produces no fields.
    """

    if not values:
        return {}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("query IoUs must be finite values in [0, 1]")
    return {
        "iou_accuracy_at_0_5": sum(value >= 0.5 for value in values) / len(values),
        "iou_accuracy_at_0_75": sum(value >= 0.75 for value in values) / len(values),
    }


def panoptic_ids(path: Path, height: int, width: int, name: str) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            if opened.size != (width, height):
                raise ModelOutputError(
                    f"{name} panoptic map has shape {(opened.height, opened.width)}, "
                    f"expected {(height, width)}"
                )
            return panoptic_ids_from_image(opened)
    except (OSError, ValueError) as exc:
        raise ModelOutputError(
            f"cannot decode {name} panoptic PNG {path}: {exc}"
        ) from exc


def safe_artifact_path(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ModelOutputError(f"{name} file_name must be a non-empty relative path")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ModelOutputError(f"{name} file_name escapes the prediction directory")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ModelOutputError(
            f"{name} file_name escapes the prediction directory"
        ) from exc
    if not path.is_file():
        raise ModelOutputError(f"{name} artifact does not exist: {path}")
    return path


def checked_segments(
    id_map: np.ndarray,
    segments_info: Any,
    category_ids: set[int],
    name: str,
) -> dict[int, dict[str, int]]:
    """Validate exact raster/metadata closure and recompute segment geometry."""

    if isinstance(segments_info, str | bytes) or not isinstance(
        segments_info, Sequence
    ):
        raise ModelOutputError(f"{name} segments_info must be an array")
    result: dict[int, dict[str, int]] = {}
    for index, segment in enumerate(segments_info):
        if not isinstance(segment, Mapping):
            raise ModelOutputError(f"{name} segments_info[{index}] must be an object")
        identifier = segment.get("id")
        category_id = segment.get("category_id")
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier <= 0
        ):
            raise ModelOutputError(f"{name} segments_info[{index}] has invalid id")
        if identifier in result:
            raise ModelOutputError(f"{name} has duplicate segment id {identifier}")
        if (
            isinstance(category_id, bool)
            or not isinstance(category_id, int)
            or category_id not in category_ids
        ):
            raise ModelOutputError(
                f"{name} segment {identifier} has unknown category_id {category_id!r}"
            )
        if segment.get("iscrowd", 0) not in (0, False):
            raise ModelOutputError(
                f"{name} segment {identifier} uses unsupported crowd semantics"
            )
        mask = id_map == identifier
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ModelOutputError(f"{name} segment {identifier} has no pixels")
        area = int(len(xs))
        bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        raw_area = segment.get("area")
        if (
            isinstance(raw_area, bool)
            or not isinstance(raw_area, int | float)
            or not math.isfinite(raw_area)
            or raw_area != area
        ):
            raise ModelOutputError(
                f"{name} segment {identifier} area disagrees with native pixels"
            )
        raw_bbox = segment.get("bbox")
        if (
            isinstance(raw_bbox, str | bytes)
            or not isinstance(raw_bbox, Sequence)
            or len(raw_bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                for value in raw_bbox
            )
            or list(raw_bbox) != bbox
        ):
            raise ModelOutputError(
                f"{name} segment {identifier} bbox disagrees with native pixels"
            )
        result[identifier] = {"category_id": int(category_id), "area": area}
    pixel_ids = {int(value) for value in np.unique(id_map) if int(value) != 0}
    if pixel_ids != set(result):
        missing = sorted(pixel_ids.difference(result))
        absent = sorted(set(result).difference(pixel_ids))
        raise ModelOutputError(
            f"{name} native pixel/segments_info IDs disagree; undeclared={missing}, absent={absent}"
        )
    return result


def semantic_map(
    id_map: np.ndarray, segments: Mapping[int, Mapping[str, int]]
) -> np.ndarray:
    result = np.zeros(id_map.shape, dtype=np.int64)
    for identifier, segment in segments.items():
        result[id_map == identifier] = int(segment["category_id"])
    return result
