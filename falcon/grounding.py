"""Grounding targets and prediction resolution for Falcon region tokens.

Ground-truth masks are accepted only by :func:`match_proposals`, which is a
training/scoring diagnostic.  Prediction resolvers consume only generated text
and the detector's already-selected proposals.  This separation prevents a
ground-truth mask from changing the proposal set seen by the model.

The module intentionally keeps torch and scipy imports inside the matcher so
transport parsing and inference-result serialization remain lightweight.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

SEG_MARKER = "<SEG>"
PANOPTIC_MARKER = "<PANOPTIC>"
TRANSPORT_VERSION = "falcon-region-transport/v1"
MATCHING_POLICY_VERSION = "falcon-grounding-match/cardinality-iou-v1"
_REGION = re.compile(r"region_([0-9]{3})\Z")


class GroundingTransportError(ValueError):
    """Raised when generated grounding transport is malformed or ambiguous."""


class GroundingResolutionError(ValueError):
    """Raised when valid transport cannot be resolved against detector output."""


class UnmatchedGroundingError(RuntimeError):
    """Raised when a positive target lacks a complete detector assignment."""


class GroundingMatchStatus(str, Enum):
    """Exhaustive distinction between negatives and positive match outcomes."""

    TRUE_NEGATIVE = "true_negative"
    FULL_MATCH = "full_match"
    PARTIAL_MATCH = "partial_match"
    UNMATCHED_POSITIVE = "unmatched_positive"


@dataclass(frozen=True)
class GroundTruthInstance:
    """One immutable ground-truth instance supplied to the diagnostic matcher."""

    instance_id: int | str
    category_id: int
    mask: Any

    def __post_init__(self) -> None:
        if isinstance(self.instance_id, bool) or not isinstance(self.instance_id, int | str):
            raise TypeError("instance_id must be an integer or string")
        if isinstance(self.instance_id, str) and not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if isinstance(self.category_id, bool) or not isinstance(self.category_id, int):
            raise TypeError("category_id must be an integer")
        if self.category_id < 1:
            raise ValueError("category_id must be positive")


@dataclass(frozen=True)
class ProposalMatch:
    """One threshold-qualified one-to-one GT/proposal assignment."""

    gt_index: int
    proposal_index: int
    iou: float
    instance_id: int | str
    category_id: int


@dataclass(frozen=True)
class GroundingMatch:
    """Complete matching diagnostic; never conflates a miss with a negative."""

    status: GroundingMatchStatus
    ground_truth_count: int
    proposal_count: int
    matches: tuple[ProposalMatch, ...]
    unmatched_gt_indices: tuple[int, ...]
    unmatched_proposal_indices: tuple[int, ...]
    policy_version: str = MATCHING_POLICY_VERSION

    @property
    def region_indices(self) -> tuple[int, ...]:
        """Matched proposal indices in stable proposal order."""

        return tuple(sorted(item.proposal_index for item in self.matches))

    def require_transportable(self) -> None:
        """Reject incomplete positive supervision before answer construction."""

        if self.status in (
            GroundingMatchStatus.PARTIAL_MATCH,
            GroundingMatchStatus.UNMATCHED_POSITIVE,
        ):
            raise UnmatchedGroundingError(
                f"cannot encode {self.status.value}: "
                f"{len(self.unmatched_gt_indices)} of {self.ground_truth_count} targets unmatched"
            )


@dataclass(frozen=True)
class SegmentationTransport:
    """Parsed SEG transport.  An empty tuple is an explicit valid negative."""

    region_indices: tuple[int, ...]


@dataclass(frozen=True)
class PanopticTransportInstance:
    region_index: int
    category_id: int


@dataclass(frozen=True)
class PanopticTransport:
    instances: tuple[PanopticTransportInstance, ...]


@dataclass(frozen=True)
class SegmentationPrediction:
    """Resolved detector masks in generated region-reference order."""

    region_indices: tuple[int, ...]
    masks: np.ndarray


@dataclass(frozen=True)
class PanopticInstance:
    """One generated class assignment resolved to a live detector proposal."""

    region_index: int
    category_id: int
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray


@dataclass(frozen=True)
class PanopticPrediction:
    """Instance predictions and their deterministic non-overlapping raster."""

    instances: tuple[PanopticInstance, ...]
    id_map: np.ndarray
    segments_info: tuple[dict[str, Any], ...]


def _validate_max_regions(max_regions: int) -> None:
    if isinstance(max_regions, bool) or not isinstance(max_regions, int) or max_regions < 1:
        raise ValueError("max_regions must be a positive integer")
    if max_regions > 1000:
        raise ValueError("v1 region transport supports at most 1000 regions")


def _validate_region_index(index: Any, max_regions: int) -> int:
    _validate_max_regions(max_regions)
    if isinstance(index, bool) or not isinstance(index, int):
        raise GroundingTransportError("region index must be an integer")
    if not 0 <= index < max_regions:
        raise GroundingTransportError(
            f"region index {index} lies outside [0, {max_regions})"
        )
    return index


def _region_name(index: Any, max_regions: int) -> str:
    return f"region_{_validate_region_index(index, max_regions):03d}"


def _parse_region_name(value: Any, max_regions: int) -> int:
    if not isinstance(value, str):
        raise GroundingTransportError("region reference must be a string")
    match = _REGION.fullmatch(value)
    if match is None:
        raise GroundingTransportError(f"invalid region reference {value!r}")
    return _validate_region_index(int(match.group(1)), max_regions)


def _payload(text: str, marker: str) -> Mapping[str, Any]:
    if not isinstance(text, str):
        raise GroundingTransportError("grounding transport must be text")
    stripped = text.strip()
    if not stripped.startswith(marker):
        raise GroundingTransportError(f"prediction must begin with {marker}")
    encoded = stripped[len(marker) :]
    if not encoded:
        raise GroundingTransportError(f"bare {marker} is not valid transport")
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise GroundingTransportError(f"invalid JSON after {marker}") from exc
    if not isinstance(value, Mapping):
        raise GroundingTransportError(f"{marker} payload must be an object")
    return value


def encode_seg_transport(
    region_indices: Iterable[int],
    *,
    max_regions: int = 100,
) -> str:
    """Encode strict SEG transport; an empty iterable is a valid negative."""

    indices = tuple(_validate_region_index(index, max_regions) for index in region_indices)
    if len(indices) != len(set(indices)):
        raise GroundingTransportError("SEG transport contains duplicate regions")
    payload = {"regions": [_region_name(index, max_regions) for index in indices]}
    return SEG_MARKER + json.dumps(payload, separators=(",", ":"))


def parse_seg_transport(text: str, *, max_regions: int = 100) -> SegmentationTransport:
    """Parse exact SEG transport, rejecting a bare marker and extra fields."""

    value = _payload(text, SEG_MARKER)
    if set(value) != {"regions"}:
        raise GroundingTransportError("SEG payload must contain only 'regions'")
    raw = value["regions"]
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise GroundingTransportError("SEG regions must be an array")
    indices = tuple(_parse_region_name(item, max_regions) for item in raw)
    if len(indices) != len(set(indices)):
        raise GroundingTransportError("SEG transport contains duplicate regions")
    return SegmentationTransport(indices)


def _category_set(category_ids: Collection[int]) -> frozenset[int]:
    categories = frozenset(category_ids)
    if not categories:
        raise ValueError("category_ids must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in categories):
        raise ValueError("category_ids must contain positive integers")
    return categories


def encode_panoptic_transport(
    instances: Iterable[PanopticTransportInstance],
    *,
    category_ids: Collection[int],
    max_regions: int = 100,
) -> str:
    """Encode region/category pairs without introducing detector class logits."""

    allowed = _category_set(category_ids)
    values = tuple(instances)
    regions: set[int] = set()
    encoded = []
    for item in values:
        if not isinstance(item, PanopticTransportInstance):
            raise TypeError("instances must contain PanopticTransportInstance values")
        region_index = _validate_region_index(item.region_index, max_regions)
        if region_index in regions:
            raise GroundingTransportError("PANOPTIC transport contains duplicate regions")
        regions.add(region_index)
        if item.category_id not in allowed:
            raise GroundingTransportError(f"unknown category_id {item.category_id}")
        encoded.append(
            {"region": _region_name(region_index, max_regions), "category_id": item.category_id}
        )
    payload = {"instances": encoded}
    return PANOPTIC_MARKER + json.dumps(payload, separators=(",", ":"))


def parse_panoptic_transport(
    text: str,
    *,
    category_ids: Collection[int],
    max_regions: int = 100,
) -> PanopticTransport:
    """Parse exact PANOPTIC transport using manifest-provided categories."""

    allowed = _category_set(category_ids)
    value = _payload(text, PANOPTIC_MARKER)
    if set(value) != {"instances"}:
        raise GroundingTransportError("PANOPTIC payload must contain only 'instances'")
    raw = value["instances"]
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise GroundingTransportError("PANOPTIC instances must be an array")

    instances = []
    regions: set[int] = set()
    for position, entry in enumerate(raw):
        if not isinstance(entry, Mapping) or set(entry) != {"region", "category_id"}:
            raise GroundingTransportError(
                f"PANOPTIC instances[{position}] must contain region and category_id"
            )
        region_index = _parse_region_name(entry["region"], max_regions)
        if region_index in regions:
            raise GroundingTransportError("PANOPTIC transport contains duplicate regions")
        regions.add(region_index)
        category_id = entry["category_id"]
        if isinstance(category_id, bool) or not isinstance(category_id, int):
            raise GroundingTransportError("PANOPTIC category_id must be an integer")
        if category_id not in allowed:
            raise GroundingTransportError(f"unknown category_id {category_id}")
        instances.append(PanopticTransportInstance(region_index, category_id))
    return PanopticTransport(tuple(instances))


def _binary_cpu_tensor(value: Any, name: str, *, dimensions: int) -> Any:
    import torch

    tensor = value.detach().to("cpu") if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if tensor.dtype != torch.bool:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
        if not torch.logical_or(tensor == 0, tensor == 1).all():
            raise ValueError(f"{name} must be binary")
        tensor = tensor.to(dtype=torch.bool)
    return tensor


def _qualified_hungarian_assignment(
    overlaps: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximize qualified match count, then total IoU among equal-cardinality plans."""

    from scipy.optimize import linear_sum_assignment

    if overlaps.ndim != 2:
        raise ValueError("overlaps must be a two-dimensional matrix")
    assignment_size = min(overlaps.shape)
    # A cardinality bonus larger than the greatest possible aggregate IoU
    # difference makes a threshold-qualified edge the primary objective.
    reward = (overlaps >= threshold) * (assignment_size + 1.0) + overlaps
    return linear_sum_assignment(-reward)


def match_proposals(
    ground_truth: Sequence[GroundTruthInstance],
    proposal_masks: Any,
    *,
    iou_threshold: float = 0.5,
) -> GroundingMatch:
    """Hungarian mask-IoU diagnostic over an unchanged proposal set.

    The returned status explicitly distinguishes an annotation-level negative
    from a detector miss.  No target box or mask is ever inserted into the
    proposal set.
    """

    import torch
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, int | float):
        raise TypeError("iou_threshold must be numeric")
    threshold = float(iou_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must lie in [0, 1]")
    if any(not isinstance(item, GroundTruthInstance) for item in ground_truth):
        raise TypeError("ground_truth must contain GroundTruthInstance values")

    proposals = _binary_cpu_tensor(proposal_masks, "proposal_masks", dimensions=3)
    proposal_count = int(proposals.shape[0])
    ground_truth_count = len(ground_truth)
    if ground_truth_count == 0:
        return GroundingMatch(
            GroundingMatchStatus.TRUE_NEGATIVE,
            0,
            proposal_count,
            (),
            (),
            tuple(range(proposal_count)),
        )

    targets = torch.stack(
        [_binary_cpu_tensor(item.mask, "ground-truth mask", dimensions=2) for item in ground_truth]
    )
    if tuple(targets.shape[-2:]) != tuple(proposals.shape[-2:]):
        raise ValueError("ground-truth and proposal masks must share spatial dimensions")
    if not targets.flatten(1).any(dim=1).all():
        raise ValueError("ground-truth instance masks must not be empty")
    if proposal_count == 0:
        return GroundingMatch(
            GroundingMatchStatus.UNMATCHED_POSITIVE,
            ground_truth_count,
            0,
            (),
            tuple(range(ground_truth_count)),
            (),
        )

    intersections = torch.logical_and(targets[:, None], proposals[None]).sum(dim=(-2, -1))
    unions = torch.logical_or(targets[:, None], proposals[None]).sum(dim=(-2, -1))
    overlaps = intersections.to(torch.float64) / unions.clamp_min(1).to(torch.float64)
    rows, columns = _qualified_hungarian_assignment(overlaps.numpy(), threshold)

    matches = []
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        iou = float(overlaps[row, column])
        if iou >= threshold:
            target = ground_truth[row]
            matches.append(
                ProposalMatch(row, column, iou, target.instance_id, target.category_id)
            )
    matches.sort(key=lambda item: item.gt_index)
    matched_gt = {item.gt_index for item in matches}
    matched_proposals = {item.proposal_index for item in matches}
    unmatched_gt = tuple(index for index in range(ground_truth_count) if index not in matched_gt)
    unmatched_proposals = tuple(
        index for index in range(proposal_count) if index not in matched_proposals
    )
    if not matches:
        status = GroundingMatchStatus.UNMATCHED_POSITIVE
    elif unmatched_gt:
        status = GroundingMatchStatus.PARTIAL_MATCH
    else:
        status = GroundingMatchStatus.FULL_MATCH
    return GroundingMatch(
        status,
        ground_truth_count,
        proposal_count,
        tuple(matches),
        unmatched_gt,
        unmatched_proposals,
    )


def encode_segmentation_match(match: GroundingMatch, *, max_regions: int = 100) -> str:
    """Encode only a true negative or fully matched positive."""

    match.require_transportable()
    return encode_seg_transport(match.region_indices, max_regions=max_regions)


def encode_panoptic_match(
    match: GroundingMatch,
    *,
    category_ids: Collection[int],
    max_regions: int = 100,
) -> str:
    """Encode category assignments only after a complete diagnostic match."""

    match.require_transportable()
    instances = tuple(
        PanopticTransportInstance(item.proposal_index, item.category_id)
        for item in sorted(match.matches, key=lambda item: item.proposal_index)
    )
    return encode_panoptic_transport(
        instances,
        category_ids=category_ids,
        max_regions=max_regions,
    )


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _detector_arrays(
    detector_output: Any,
    *,
    batch_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return one detector image without changing proposal indices.

    Both the public unbatched :class:`DetectorOutput` and the model's padded
    ``DetectorBatch`` are accepted.  A padded batch must use a contiguous valid
    prefix: otherwise compressing it here would silently renumber the region
    tokens generated by the language model.
    """

    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise GroundingResolutionError("batch_index must be a nonnegative integer")
    boxes_value = _field(detector_output, "boxes_xyxy")
    if boxes_value is None:
        boxes_value = _field(detector_output, "boxes")
    scores_value = _field(detector_output, "scores")
    masks_value = _field(detector_output, "masks")
    original_hw = _field(detector_output, "original_hw")
    if original_hw is None:
        original_hw = _field(detector_output, "image_sizes")
    mask_sizes = _field(detector_output, "mask_sizes")
    valid_value = _field(detector_output, "valid")
    if any(item is None for item in (boxes_value, scores_value, masks_value, original_hw)):
        raise GroundingResolutionError(
            "detector output must provide boxes, scores, masks, and original image sizes"
        )

    boxes = _to_numpy(boxes_value)
    scores = _to_numpy(scores_value)
    masks = _to_numpy(masks_value)
    original_sizes = _to_numpy(original_hw)
    if boxes.ndim == 3:
        batch_size, proposal_count = boxes.shape[:2]
        if batch_index >= boxes.shape[0]:
            raise GroundingResolutionError("batch_index is outside the detector batch")
        if scores.ndim != 2 or scores.shape != boxes.shape[:2]:
            raise GroundingResolutionError("batched detector scores must have shape [B, N]")
        if masks.ndim != 4 or masks.shape[:2] != boxes.shape[:2]:
            raise GroundingResolutionError("batched detector masks must have shape [B, N, H, W]")
        if original_sizes.shape != (boxes.shape[0], 2):
            raise GroundingResolutionError("image_sizes must have shape [B, 2]")
        boxes = boxes[batch_index]
        scores = scores[batch_index]
        masks = masks[batch_index]
        original_sizes = original_sizes[batch_index]
        if mask_sizes is not None:
            sizes = _to_numpy(mask_sizes)
            if sizes.shape != (batch_size, 2):
                raise GroundingResolutionError("mask_sizes must have shape [B, 2]")
            mask_height, mask_width = (int(value) for value in sizes[batch_index])
            if mask_height < 1 or mask_width < 1 or (
                mask_height > masks.shape[-2] or mask_width > masks.shape[-1]
            ):
                raise GroundingResolutionError("mask_sizes exceed the padded mask canvas")
            masks = masks[:, :mask_height, :mask_width]
        if valid_value is not None:
            valid = _to_numpy(valid_value)
            if valid.shape != (batch_size, proposal_count):
                raise GroundingResolutionError("detector valid mask must have shape [B, N]")
            valid = valid[batch_index].astype(bool)
            count = int(np.count_nonzero(valid))
            if not np.array_equal(valid, np.arange(len(valid)) < count):
                raise GroundingResolutionError(
                    "detector valid proposals must form a contiguous prefix"
                )
            boxes, scores, masks = boxes[:count], scores[:count], masks[:count]
    elif batch_index != 0:
        raise GroundingResolutionError("batch_index is outside the unbatched detector output")

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise GroundingResolutionError("detector boxes must have shape [proposals, 4]")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise GroundingResolutionError("detector scores must have shape [proposals]")
    if masks.ndim != 3 or masks.shape[0] != boxes.shape[0]:
        raise GroundingResolutionError("detector masks must have shape [proposals, H, W]")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise GroundingResolutionError("selected detector boxes/scores must be finite")
    if masks.dtype != np.bool_:
        if not np.isfinite(masks).all() or not np.logical_or(masks == 0, masks == 1).all():
            raise GroundingResolutionError("selected detector masks must be binary")
        masks = masks.astype(bool)
    try:
        height, width = (int(value) for value in original_sizes)
    except (TypeError, ValueError) as exc:
        raise GroundingResolutionError("original image size must contain height and width") from exc
    if height < 1 or width < 1 or masks.shape[-2:] != (height, width):
        raise GroundingResolutionError("detector masks do not match positive original_hw")
    return boxes.astype(np.float32), scores.astype(np.float32), masks, height, width


def resolve_segmentation(
    text: str,
    detector_output: Any,
    *,
    max_regions: int = 100,
    batch_index: int = 0,
) -> SegmentationPrediction:
    """Resolve generated SEG references without any ground-truth input."""

    transport = parse_seg_transport(text, max_regions=max_regions)
    _boxes, _scores, masks, height, width = _detector_arrays(
        detector_output,
        batch_index=batch_index,
    )
    for index in transport.region_indices:
        if index >= len(masks):
            raise GroundingResolutionError(f"generated region_{index:03d} was not proposed")
        if not masks[index].any():
            raise GroundingResolutionError(f"generated region_{index:03d} has an empty mask")
    selected = (
        np.stack([masks[index] for index in transport.region_indices]).astype(bool, copy=True)
        if transport.region_indices
        else np.empty((0, height, width), dtype=bool)
    )
    return SegmentationPrediction(transport.region_indices, selected)


def _mask_bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    x_min, x_max = int(columns.min()), int(columns.max())
    y_min, y_max = int(rows.min()), int(rows.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def resolve_panoptic(
    text: str,
    detector_output: Any,
    *,
    category_ids: Collection[int],
    max_regions: int = 100,
    batch_index: int = 0,
) -> PanopticPrediction:
    """Resolve generated classes and live proposal masks into panoptic output."""

    transport = parse_panoptic_transport(
        text,
        category_ids=category_ids,
        max_regions=max_regions,
    )
    boxes, scores, masks, height, width = _detector_arrays(
        detector_output,
        batch_index=batch_index,
    )
    instances = []
    for item in transport.instances:
        index = item.region_index
        if index >= len(masks):
            raise GroundingResolutionError(f"generated region_{index:03d} was not proposed")
        if not masks[index].any():
            raise GroundingResolutionError(f"generated region_{index:03d} has an empty mask")
        instances.append(
            PanopticInstance(
                region_index=index,
                category_id=item.category_id,
                score=float(scores[index]),
                box_xyxy=tuple(float(value) for value in boxes[index]),
                mask=masks[index].astype(bool, copy=True),
            )
        )

    priority = sorted(instances, key=lambda item: (-item.score, item.region_index))
    id_map = np.zeros((height, width), dtype=np.int32)
    segments = []
    for item in priority:
        visible = np.logical_and(item.mask, id_map == 0)
        if not visible.any():
            continue
        segment_id = item.region_index + 1
        id_map[visible] = segment_id
        segments.append(
            {
                "id": segment_id,
                "category_id": item.category_id,
                "iscrowd": 0,
                "area": int(visible.sum()),
                "bbox": _mask_bbox(visible),
            }
        )
    segments.sort(key=lambda item: item["id"])
    return PanopticPrediction(tuple(instances), id_map, tuple(segments))


def select_panoptic_category(
    prediction: PanopticPrediction, category_id: int
) -> PanopticPrediction:
    """Select a query category using predicted classes, retaining instance IDs."""

    if isinstance(category_id, bool) or not isinstance(category_id, int) or category_id < 1:
        raise ValueError("query category_id must be a positive integer")
    segments = tuple(
        segment for segment in prediction.segments_info if segment["category_id"] == category_id
    )
    identifiers = [segment["id"] for segment in segments]
    selected = np.isin(prediction.id_map, identifiers)
    return PanopticPrediction(
        tuple(item for item in prediction.instances if item.category_id == category_id),
        np.where(selected, prediction.id_map, 0),
        segments,
    )


def panoptic_id_map_to_rgb(id_map: Any) -> np.ndarray:
    """Encode a nonnegative integer segment-ID map as lossless COCO RGB pixels."""

    values = _to_numpy(id_map)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("id_map must be a two-dimensional integer array")
    if np.any(values < 0) or np.any(values > 0xFFFFFF):
        raise ValueError("panoptic IDs must lie in [0, 16777215]")
    unsigned = values.astype(np.uint32, copy=False)
    return np.stack(
        (
            unsigned % 256,
            (unsigned // 256) % 256,
            (unsigned // 65536) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
