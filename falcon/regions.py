"""Patch aggregation and mask-aware region encoding for Falcon.

Boxes use original-image ``xyxy`` coordinates. Masks are thresholded in their
declared representation (logits or probabilities), without a sigmoid conversion.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import roi_align


class PatchTokenAggregator(nn.Module):
    """Concatenate each non-overlapping 2x2 patch block, then project it.

    DINOv2 emits patch tokens in row-major order.  For a block, concatenation
    order is top-left, top-right, bottom-left, bottom-right.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(4 * input_dim, output_dim)

    def forward(self, patch_tokens: Tensor, grid_size: tuple[int, int]) -> Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [batch, patches, channels]")
        batch, count, channels = patch_tokens.shape
        height, width = grid_size
        if channels != self.input_dim:
            raise ValueError(f"expected {self.input_dim} channels, received {channels}")
        if count != height * width:
            raise ValueError(f"{count} tokens cannot form the declared {height}x{width} grid")
        if height % 2 or width % 2:
            raise ValueError("the DINOv2 patch grid must be even for 2x2 aggregation")

        grid = patch_tokens.reshape(batch, height // 2, 2, width // 2, 2, channels)
        blocks = grid.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, height // 2, width // 2, 4 * channels
        )
        return self.projection(blocks).permute(0, 3, 1, 2).contiguous()


class MaskAwareRegionEncoder(nn.Module):
    """Fuse ROI-aligned and binary-mask-pooled features by concatenation.

    Invalid/padded proposals are returned as exact zero vectors.  ``image_sizes``
    contains ``(height, width)`` for each original detector image.
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        *,
        roi_size: int = 4,
        sampling_ratio: int = 2,
        probability_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= probability_threshold <= 1.0:
            raise ValueError("probability_threshold must lie in [0, 1]")
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.roi_size = roi_size
        self.sampling_ratio = sampling_ratio
        self.probability_threshold = probability_threshold
        self.projection = nn.Linear(2 * feature_dim, output_dim)

    @staticmethod
    def _validate_inputs(
        feature_map: Tensor,
        boxes: Tensor,
        masks: Tensor,
        image_sizes: Tensor,
        valid: Tensor,
    ) -> None:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [batch, channels, height, width]")
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [batch, proposals, 4]")
        if masks.ndim != 4 or masks.shape[:2] != boxes.shape[:2]:
            raise ValueError("masks must have shape [batch, proposals, height, width]")
        if image_sizes.shape != (feature_map.shape[0], 2):
            raise ValueError("image_sizes must have shape [batch, 2] in (height, width) order")
        if valid.shape != boxes.shape[:2]:
            raise ValueError("valid must have shape [batch, proposals]")
        if boxes.shape[0] != feature_map.shape[0]:
            raise ValueError("feature_map and detector outputs must have the same batch size")

    def pool_components(
        self,
        feature_map: Tensor,
        boxes: Tensor,
        masks: Tensor,
        image_sizes: Tensor,
        valid: Tensor | None = None,
        *,
        masks_are_logits: bool = False,
        mask_sizes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return valid proposal indices, ROI vectors, and mask vectors."""

        if valid is None:
            valid = torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
        else:
            valid = valid.to(dtype=torch.bool)
        self._validate_inputs(feature_map, boxes, masks, image_sizes, valid)

        indices = valid.nonzero(as_tuple=False)
        if indices.numel() == 0:
            empty = feature_map.new_empty((0, self.feature_dim))
            return indices, empty, empty

        device = feature_map.device
        dtype = feature_map.dtype
        indices = indices.to(device=device)
        boxes = boxes.to(device=device, dtype=dtype)
        masks = masks.to(device=device)
        image_sizes = image_sizes.to(device=device, dtype=dtype)

        batch_ids, proposal_ids = indices.unbind(dim=1)
        selected_boxes = boxes[batch_ids, proposal_ids].clone()
        source_h = image_sizes[batch_ids, 0].clamp_min(1)
        source_w = image_sizes[batch_ids, 1].clamp_min(1)
        feature_h, feature_w = feature_map.shape[-2:]
        selected_boxes[:, 0::2] *= feature_w / source_w[:, None]
        selected_boxes[:, 1::2] *= feature_h / source_h[:, None]
        rois = torch.cat((batch_ids.to(dtype=dtype).unsqueeze(1), selected_boxes), dim=1)

        roi_features = roi_align(
            feature_map,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=self.sampling_ratio,
            aligned=True,
        )
        roi_vectors = roi_features.mean(dim=(-2, -1))

        threshold = 0.0 if masks_are_logits else self.probability_threshold
        if mask_sizes is None:
            mask_sizes = torch.tensor([list(masks.shape[-2:])] * feature_map.shape[0])
        if mask_sizes.shape != (feature_map.shape[0], 2):
            raise ValueError("mask_sizes must have shape [batch, 2]")
        binary_masks = feature_map.new_zeros((len(indices), 1, feature_h, feature_w))
        for batch_index in range(feature_map.shape[0]):
            selected = batch_ids == batch_index
            if not bool(selected.any()):
                continue
            height, width = (int(value) for value in mask_sizes[batch_index])
            if not (0 < height <= masks.shape[-2] and 0 < width <= masks.shape[-1]):
                raise ValueError("mask_sizes exceed the padded canvas or are nonpositive")
            selected_masks = masks[batch_index, proposal_ids[selected], :height, :width].unsqueeze(
                1
            )
            binary = (
                selected_masks
                if selected_masks.dtype == torch.bool
                else selected_masks >= threshold
            )
            binary_masks[selected] = F.interpolate(
                binary.to(dtype=dtype), size=(feature_h, feature_w), mode="nearest"
            ).to(dtype=dtype)
        selected_features = feature_map.index_select(0, batch_ids)
        denominator = binary_masks.sum(dim=(-2, -1)).clamp_min(1.0)
        mask_vectors = (selected_features * binary_masks).sum(dim=(-2, -1)) / denominator
        return indices, roi_vectors, mask_vectors

    def forward(
        self,
        feature_map: Tensor,
        boxes: Tensor,
        masks: Tensor,
        image_sizes: Tensor | Sequence[tuple[int, int]],
        valid: Tensor | None = None,
        *,
        masks_are_logits: bool = False,
        mask_sizes: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(image_sizes, Tensor):
            image_sizes = torch.as_tensor(image_sizes, device=feature_map.device)
        if valid is None:
            valid = torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
        indices, roi_vectors, mask_vectors = self.pool_components(
            feature_map,
            boxes,
            masks,
            image_sizes,
            valid,
            masks_are_logits=masks_are_logits,
            mask_sizes=mask_sizes,
        )

        batch, proposals = boxes.shape[:2]
        flat_output = feature_map.new_zeros((batch * proposals, self.output_dim))
        if indices.numel() == 0:
            return flat_output.reshape(batch, proposals, self.output_dim)

        fused = self.projection(torch.cat((roi_vectors, mask_vectors), dim=-1))
        flat_indices = indices[:, 0] * proposals + indices[:, 1]
        flat_output = flat_output.index_copy(0, flat_indices, fused)
        return flat_output.reshape(batch, proposals, self.output_dim)
