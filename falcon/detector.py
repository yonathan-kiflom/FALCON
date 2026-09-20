from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .artifacts import atomic_json, external_output

VARIANTS = ("seg-nano", "seg-small", "seg-medium", "seg-large", "seg-xlarge", "seg-2xlarge")
PROPOSAL_POLICY_VERSION = "falcon-proposals/v1"
# Pinned RF-DETR applies a strict ``scores > threshold`` filter.  Passing the
# user cutoff there would drop equality-boundary proposals before Falcon's
# documented inclusive policy can inspect them.
BACKEND_SCORE_THRESHOLD = -float("inf")
STAGE1_LOCK_NAME = ".stage1.lock"


@dataclass(frozen=True)
class RFDETRVariantSpec:
    """Architecture defaults from the pinned ``rfdetr==1.5.2`` package."""

    resolution: int
    patch_size: int
    num_windows: int
    decoder_layers: int
    num_queries: int
    num_select: int

    @property
    def input_divisor(self) -> int:
        return self.patch_size * self.num_windows


VARIANT_SPECS: dict[str, RFDETRVariantSpec] = {
    "seg-nano": RFDETRVariantSpec(312, 12, 1, 4, 100, 100),
    "seg-small": RFDETRVariantSpec(384, 12, 2, 4, 100, 100),
    "seg-medium": RFDETRVariantSpec(432, 12, 2, 5, 200, 200),
    "seg-large": RFDETRVariantSpec(504, 12, 2, 5, 200, 200),
    "seg-xlarge": RFDETRVariantSpec(624, 12, 2, 6, 300, 300),
    "seg-2xlarge": RFDETRVariantSpec(768, 12, 2, 6, 300, 300),
}


@dataclass(frozen=True)
class RFDETRModelConfig:
    """Falcon-owned RF-DETR model options validated before model construction."""

    resolution: int | None = None
    amp: bool | None = None

    def __post_init__(self) -> None:
        if self.amp is not None and not isinstance(self.amp, bool):
            raise TypeError("RF-DETR amp must be boolean when specified")

    def resolved_resolution(self, variant: str) -> int:
        if variant not in VARIANT_SPECS:
            raise ValueError(f"Unknown RF-DETR variant: {variant}")
        resolution = (
            VARIANT_SPECS[variant].resolution
            if self.resolution is None
            else self.resolution
        )
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1:
            raise ValueError("RF-DETR resolution must be a positive integer")
        divisor = VARIANT_SPECS[variant].input_divisor
        if resolution % divisor:
            raise ValueError(
                f"RF-DETR {variant} resolution must be divisible by {divisor}; "
                f"received {resolution}"
            )
        return resolution

    def constructor_kwargs(self, variant: str) -> dict[str, Any]:
        self.resolved_resolution(variant)
        values: dict[str, Any] = {}
        if self.resolution is not None:
            values["resolution"] = self.resolution
        if self.amp is not None:
            values["amp"] = self.amp
        return values


@dataclass(frozen=True)
class RFDETRBackendConfig:
    """Effective architecture values read from a constructed pinned backend."""

    variant: str
    resolution: int
    patch_size: int
    num_windows: int
    decoder_layers: int
    num_queries: int
    num_select: int
    amp: bool


@dataclass(frozen=True)
class RFDETRTrainConfig:
    """Typed Stage-1 settings passed explicitly through RF-DETR's public API."""

    epochs: int = 12
    batch_size: int = 1
    gradient_accumulation: int = 16
    workers: int = 4
    seed: int = 42
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 1.5e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "step"
    warmup_epochs: float = 0.0
    multi_scale: bool = True
    expanded_scales: bool = True

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "gradient_accumulation", "workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an unsigned 32-bit integer")
        for name in ("learning_rate", "encoder_learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 < float(value) < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "warmup_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= float(value) < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lr_scheduler not in ("cosine", "step"):
            raise ValueError("lr_scheduler must be 'cosine' or 'step'")
        if not isinstance(self.multi_scale, bool) or not isinstance(self.expanded_scales, bool):
            raise TypeError("multi_scale and expanded_scales must be boolean")

    def backend_kwargs(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.gradient_accumulation,
            "num_workers": self.workers,
            "seed": self.seed,
            "lr": self.learning_rate,
            "lr_encoder": self.encoder_learning_rate,
            "weight_decay": self.weight_decay,
            "lr_scheduler": self.lr_scheduler,
            "warmup_epochs": self.warmup_epochs,
            "multi_scale": self.multi_scale,
            "expanded_scales": self.expanded_scales,
        }


@dataclass(frozen=True)
class ProposalPolicy:
    """One deterministic proposal-selection contract for train and inference."""

    score_threshold: float = 0.15
    nms_threshold: float = 0.6
    max_regions: int = 100
    mask_threshold: float = 0.5
    reject_empty_masks: bool = True
    version: str = PROPOSAL_POLICY_VERSION

    def __post_init__(self) -> None:
        for name in ("score_threshold", "nms_threshold", "mask_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if isinstance(self.max_regions, bool) or not isinstance(self.max_regions, int):
            raise TypeError("max_regions must be an integer")
        if self.max_regions < 1:
            raise ValueError("max_regions must be positive")
        if not isinstance(self.reject_empty_masks, bool):
            raise TypeError("reject_empty_masks must be boolean")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("proposal policy version must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return serializable proposal settings."""

        return {
            "version": self.version,
            "backend_score_threshold": "-inf",
            "score_threshold": float(self.score_threshold),
            "nms_threshold": float(self.nms_threshold),
            "max_regions": self.max_regions,
            "mask_threshold": float(self.mask_threshold),
            "mask_threshold_operator": ">=",
            "reject_empty_masks": self.reject_empty_masks,
            "ordering": "score_descending_then_backend_index",
            "nms_tie_break": "lower_backend_index",
        }

    @staticmethod
    def _stable_nms(boxes: torch.Tensor, threshold: float) -> torch.Tensor:
        """NMS for pre-sorted boxes with an explicit lower-index tie break."""

        remaining = torch.arange(len(boxes), dtype=torch.long)
        kept = []
        while remaining.numel():
            current = remaining[0]
            kept.append(current)
            if remaining.numel() == 1:
                break
            rest = remaining[1:]
            left_top = torch.maximum(boxes[current, :2], boxes[rest, :2])
            right_bottom = torch.minimum(boxes[current, 2:], boxes[rest, 2:])
            intersection = (right_bottom - left_top).clamp_min(0).prod(dim=1)
            current_area = (boxes[current, 2:] - boxes[current, :2]).prod()
            rest_area = (boxes[rest, 2:] - boxes[rest, :2]).prod(dim=1)
            overlap = intersection / (current_area + rest_area - intersection).clamp_min(1e-12)
            remaining = rest[overlap <= threshold]
        return torch.stack(kept) if kept else torch.empty(0, dtype=torch.long)

    def select(
        self,
        boxes: Any,
        scores: Any,
        masks: Any,
        original_hw: tuple[int, int],
        *,
        mask_finite: Any | None = None,
    ) -> DetectorOutput:
        """Validate and select proposals without consulting ground truth."""

        if (
            not isinstance(original_hw, tuple)
            or len(original_hw) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in original_hw
            )
        ):
            raise ValueError("original_hw must contain positive integer height and width")
        box_tensor = torch.as_tensor(boxes).detach().to("cpu", torch.float32)
        score_tensor = torch.as_tensor(scores).detach().to("cpu", torch.float32)
        mask_tensor = torch.as_tensor(masks).detach().to("cpu")
        if box_tensor.ndim != 2 or box_tensor.shape[-1] != 4:
            raise ValueError("boxes must have shape [proposals, 4]")
        if score_tensor.ndim != 1 or score_tensor.shape[0] != box_tensor.shape[0]:
            raise ValueError("scores must have shape [proposals]")
        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != box_tensor.shape[0]:
            raise ValueError("masks must have shape [proposals, height, width]")
        if tuple(mask_tensor.shape[-2:]) != original_hw:
            raise ValueError("proposal masks must match original_hw")
        if mask_tensor.dtype != torch.bool:
            raise TypeError("ProposalPolicy.select requires already-binarized masks")
        if mask_finite is None:
            finite_masks = torch.ones(len(box_tensor), dtype=torch.bool)
        else:
            finite_masks = torch.as_tensor(mask_finite).detach().to("cpu", torch.bool)
            if finite_masks.shape != (len(box_tensor),):
                raise ValueError("mask_finite must have shape [proposals]")

        height, width = original_hw
        box_tensor = box_tensor.clone()
        box_tensor[:, 0::2].clamp_(0, width)
        box_tensor[:, 1::2].clamp_(0, height)
        valid = torch.isfinite(box_tensor).all(dim=1) & torch.isfinite(score_tensor)
        valid &= finite_masks & (score_tensor >= self.score_threshold)
        valid &= (box_tensor[:, 2] > box_tensor[:, 0]) & (box_tensor[:, 3] > box_tensor[:, 1])
        if self.reject_empty_masks:
            valid &= mask_tensor.flatten(1).any(dim=1)

        raw_indices = torch.arange(len(box_tensor), dtype=torch.long)
        box_tensor = box_tensor[valid]
        score_tensor = score_tensor[valid]
        mask_tensor = mask_tensor[valid]
        raw_indices = raw_indices[valid]
        if not len(box_tensor):
            return DetectorOutput(
                box_tensor,
                score_tensor,
                mask_tensor,
                original_hw,
                raw_indices,
            )

        order = torch.argsort(score_tensor, descending=True, stable=True)
        box_tensor = box_tensor[order]
        score_tensor = score_tensor[order]
        mask_tensor = mask_tensor[order]
        raw_indices = raw_indices[order]
        keep = self._stable_nms(box_tensor, float(self.nms_threshold))[: self.max_regions]
        return DetectorOutput(
            box_tensor[keep],
            score_tensor[keep],
            mask_tensor[keep],
            original_hw,
            raw_indices[keep],
        )


@dataclass
class DetectorOutput:
    """Post-threshold, post-NMS RF-DETR output in original-image coordinates."""

    boxes_xyxy: torch.Tensor
    scores: torch.Tensor
    masks: torch.Tensor
    original_hw: tuple[int, int]
    proposal_ids: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> DetectorOutput:
        return DetectorOutput(
            self.boxes_xyxy.to(device),
            self.scores.to(device),
            self.masks.to(device),
            self.original_hw,
            None if self.proposal_ids is None else self.proposal_ids.to(device),
        )


def _models() -> dict[str, type]:
    # Albumentations otherwise performs a PyPI version check during RF-DETR import.
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    try:
        from rfdetr import (
            RFDETRSeg2XLarge,
            RFDETRSegLarge,
            RFDETRSegMedium,
            RFDETRSegNano,
            RFDETRSegSmall,
            RFDETRSegXLarge,
        )
    except ImportError as error:
        raise ImportError(
            "Install the pinned environment with `conda env create -f environment.yml`"
        ) from error
    return {
        "seg-nano": RFDETRSegNano,
        "seg-small": RFDETRSegSmall,
        "seg-medium": RFDETRSegMedium,
        "seg-large": RFDETRSegLarge,
        "seg-xlarge": RFDETRSegXLarge,
        "seg-2xlarge": RFDETRSeg2XLarge,
    }


class RFDETRDetector:
    """Frozen RF-DETR with a portable, weights-free construction path."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        variant: str = "seg-2xlarge",
        device: str = "cuda",
        score_threshold: float | None = None,
        nms_threshold: float | None = None,
        max_regions: int | None = None,
        mask_threshold: float | None = None,
        resolution: int | None = None,
        proposal_policy: ProposalPolicy | None = None,
        model_config: RFDETRModelConfig | None = None,
        model: Any | None = None,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown RF-DETR variant: {variant}")
        if model_config is not None and resolution is not None:
            if model_config.resolution != resolution:
                raise ValueError("resolution conflicts with model_config.resolution")
        resolved_model_config = model_config or RFDETRModelConfig(resolution=resolution)
        # This validation is intentionally before _models() so an incompatible
        # request cannot trigger RF-DETR construction or weight download.
        resolved_model_config.resolved_resolution(variant)

        legacy_values = {
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
            "max_regions": max_regions,
            "mask_threshold": mask_threshold,
        }
        if proposal_policy is None:
            proposal_policy = ProposalPolicy(
                score_threshold=0.15 if score_threshold is None else score_threshold,
                nms_threshold=0.6 if nms_threshold is None else nms_threshold,
                max_regions=100 if max_regions is None else max_regions,
                mask_threshold=0.5 if mask_threshold is None else mask_threshold,
            )
        else:
            if not isinstance(proposal_policy, ProposalPolicy):
                raise TypeError("proposal_policy must be a ProposalPolicy")
            for name, value in legacy_values.items():
                if value is not None and value != getattr(proposal_policy, name):
                    raise ValueError(f"{name} conflicts with proposal_policy.{name}")
        if model is None:
            kwargs: dict[str, Any] = {"device": device}
            if checkpoint is not None:
                kwargs["pretrain_weights"] = str(Path(checkpoint).expanduser().resolve())
            kwargs.update(resolved_model_config.constructor_kwargs(variant))
            model = _models()[variant](**kwargs)
        self.model = model
        self.variant = variant
        self.model_config = resolved_model_config
        self.proposal_policy = proposal_policy
        # Keep the original scalar attributes for downstream compatibility.
        self.score_threshold = proposal_policy.score_threshold
        self.nms_threshold = proposal_policy.nms_threshold
        self.max_regions = proposal_policy.max_regions
        self.mask_threshold = proposal_policy.mask_threshold
        self.backend_config = self._backend_config(model, variant, resolved_model_config)

    @property
    def torch_model(self) -> torch.nn.Module:
        """The module to register in a complete FALCON state dictionary."""

        return self.model.model.model

    def export_config(self) -> dict[str, Any]:
        """Export architecture and proposal settings without machine-local paths."""

        architecture = self.model.model_config.model_dump()
        architecture.pop("pretrain_weights", None)
        architecture.pop("device", None)
        # RF-DETR may replace its classification head while loading a checkpoint
        # without updating model_config.num_classes.
        architecture["num_classes"] = int(self.torch_model.class_embed.bias.shape[0]) - 1
        return {
            "variant": self.variant,
            "architecture": architecture,
            "proposal_policy": asdict(self.proposal_policy),
        }

    @classmethod
    def from_config(
        cls,
        detector_config: Mapping[str, Any],
        device: str | torch.device = "cpu",
    ) -> RFDETRDetector:
        """Construct the pinned segmentation architecture without loading weights.

        Unlike RF-DETR's training constructor, this also supports Transformers'
        meta-device initialization for low-memory checkpoint loading.
        """

        if not isinstance(detector_config, Mapping) or set(detector_config) != {
            "variant", "architecture", "proposal_policy"
        }:
            raise ValueError("Invalid portable RF-DETR configuration")
        variant = detector_config["variant"]
        if variant not in VARIANTS:
            raise ValueError(f"Unknown RF-DETR variant: {variant}")
        architecture = dict(detector_config["architecture"])
        if "pretrain_weights" in architecture or "device" in architecture:
            raise ValueError("Portable detector architecture cannot name weights or devices")
        # In pinned RF-DETR, patch_size != 14 disables the otherwise implicit
        # DINOv2 Hub initialization. All supported segmentation variants use 12.
        if architecture.get("patch_size") != 12 or architecture.get("encoder") not in {
            "dinov2_windowed_small", "dinov2_windowed_base"
        } or architecture.get("segmentation_head") is not True:
            raise ValueError("Unsupported portable RF-DETR segmentation architecture")
        policy = ProposalPolicy(**detector_config["proposal_policy"])
        model_options = RFDETRModelConfig(
            resolution=architecture.get("resolution"), amp=architecture.get("amp")
        )
        model_options.resolved_resolution(variant)
        model_class = _models()[variant]
        from rfdetr.main import populate_args
        from rfdetr.models.lwdetr import PostProcess, build_model

        class ConfigOnlyModel(model_class):
            def maybe_download_pretrain_weights(self):
                # Both the detector and its DINO encoder come from FALCON's
                # state dictionary; no external initialization is needed.
                pass

            def get_model(self, config):
                args = populate_args(**config.model_dump())
                # Accelerate intercepts parameter registration but RF-DETR also
                # creates plain tensors for its initial classification bias.
                # Keep those on the same initialization device as parameters.
                initialization_device = torch.nn.Embedding(1, 1).weight.device
                with torch.device(initialization_device):
                    module = build_model(args)
                return SimpleNamespace(
                    args=args,
                    resolution=args.resolution,
                    model=module,
                    device=torch.device(device),
                    postprocess=PostProcess(num_select=args.num_select),
                    class_names=[],
                )

        backend = ConfigOnlyModel(**architecture, pretrain_weights=None, device="cpu")
        detector = cls(
            variant=variant,
            model=backend,
            model_config=model_options,
            proposal_policy=policy,
        )
        detector.torch_model.requires_grad_(False)
        if not next(detector.torch_model.parameters()).is_meta:
            detector.to(device)
        return detector.eval()

    def to(self, device: str | torch.device) -> RFDETRDetector:
        self.torch_model.to(device)
        self.model.model.device = torch.device(device)
        return self

    def eval(self) -> RFDETRDetector:
        self.torch_model.eval()
        return self

    @staticmethod
    def _backend_config(
        model: Any,
        variant: str,
        requested: RFDETRModelConfig,
    ) -> RFDETRBackendConfig:
        fallback = VARIANT_SPECS[variant]
        value = getattr(model, "model_config", None)

        def read(name: str, default: int) -> int:
            item = getattr(value, name, default)
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise RuntimeError(f"RF-DETR backend reported invalid {name}: {item!r}")
            return item

        amp = getattr(value, "amp", True if requested.amp is None else requested.amp)
        if not isinstance(amp, bool):
            raise RuntimeError(f"RF-DETR backend reported invalid amp: {amp!r}")

        effective = RFDETRBackendConfig(
            variant=variant,
            resolution=read("resolution", requested.resolved_resolution(variant)),
            patch_size=read("patch_size", fallback.patch_size),
            num_windows=read("num_windows", fallback.num_windows),
            decoder_layers=read("dec_layers", fallback.decoder_layers),
            num_queries=read("num_queries", fallback.num_queries),
            num_select=read("num_select", fallback.num_select),
            amp=amp,
        )
        if effective.resolution % (effective.patch_size * effective.num_windows):
            raise RuntimeError("constructed RF-DETR backend has an incompatible input resolution")
        return effective

    @staticmethod
    def _image_hw(image: Any) -> tuple[int, int]:
        if isinstance(image, str | Path):
            with Image.open(image) as opened:
                return int(opened.height), int(opened.width)
        if isinstance(image, Image.Image):
            return int(image.height), int(image.width)
        shape = tuple(image.shape) if hasattr(image, "shape") else ()
        if len(shape) == 2:
            return int(shape[0]), int(shape[1])
        if len(shape) != 3:
            raise ValueError("RF-DETR input must be one PIL, NumPy, or tensor image")
        if shape[0] in (1, 3, 4):
            return int(shape[-2]), int(shape[-1])
        return int(shape[0]), int(shape[1])

    def _masks(
        self,
        value: Any,
        *,
        count: int,
        original_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value is None:
            if count:
                raise RuntimeError("RF-DETR segmentation output did not contain masks")
            return (
                torch.empty((0, *original_hw), dtype=torch.bool),
                torch.empty(0, dtype=torch.bool),
            )
        array = (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        if count == 0 and array.size == 0:
            return (
                torch.empty((0, *original_hw), dtype=torch.bool),
                torch.empty(0, dtype=torch.bool),
            )
        if count == 1 and array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[0] != count:
            raise RuntimeError("Invalid RF-DETR mask shape")
        masks = torch.as_tensor(array)
        if masks.dtype == torch.bool:
            finite = torch.ones(count, dtype=torch.bool)
        else:
            masks = masks.to(dtype=torch.float32)
            finite = torch.isfinite(masks).flatten(1).all(dim=1)
            # ProposalPolicy v1 deliberately preserves the existing >= mask
            # threshold semantics.  Non-finite proposals are rejected later.
            masks = masks >= self.mask_threshold
        if tuple(masks.shape[-2:]) != original_hw:
            masks = F.interpolate(
                masks.to(dtype=torch.float32).unsqueeze(1),
                size=original_hw,
                mode="nearest",
            ).squeeze(1) > 0.5
        return masks, finite

    @torch.inference_mode()
    def __call__(self, image: Any) -> DetectorOutput:
        # An enclosing PreTrainedModel may have moved the registered module.
        # RF-DETR also caches the input device outside that module.
        backend = getattr(self.model, "model", None)
        module = getattr(backend, "model", None)
        if isinstance(module, torch.nn.Module):
            backend.device = next(module.parameters()).device
            module.eval()
        detection = self.model.predict(image, threshold=BACKEND_SCORE_THRESHOLD)
        if isinstance(detection, list | tuple):
            if len(detection) != 1:
                raise RuntimeError("RFDETRDetector accepts exactly one image per call")
            detection = detection[0]
        original_hw = self._image_hw(image)
        boxes = torch.as_tensor(detection.xyxy).detach().to("cpu", torch.float32).reshape(-1, 4)
        confidence = detection.confidence
        if confidence is None:
            if len(boxes):
                raise RuntimeError("RF-DETR output did not contain confidence scores")
            scores = torch.empty(0, dtype=torch.float32)
        else:
            scores = torch.as_tensor(confidence).detach().to("cpu", torch.float32).reshape(-1)
        masks, mask_finite = self._masks(
            detection.mask,
            count=len(boxes),
            original_hw=original_hw,
        )
        if len(boxes) != len(scores):
            raise RuntimeError("RF-DETR box and score counts differ")
        return self.proposal_policy.select(
            boxes,
            scores,
            masks,
            original_hw,
            mask_finite=mask_finite,
        )


def _protect_stage1_output(output_root: Path, *, resume: bool) -> None:
    if resume:
        if not output_root.is_dir():
            raise ValueError("Stage-1 resume requires its existing output directory")
        extra_weights = [
            path.name
            for path in output_root.iterdir()
            if path.suffix in {".pt", ".pth", ".safetensors"} and path.name != "last.pth"
        ]
        if extra_weights:
            raise ValueError(
                "Stage-1 resume expects only last.pth; move other weights out of the "
                f"output directory first: {', '.join(sorted(extra_weights))}"
            )
        return
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError("Stage-1 output path exists and is not a directory")
        # Keep the lock inode stable across processes.
        unexpected = next(
            (entry for entry in output_root.iterdir() if entry.name != STAGE1_LOCK_NAME),
            None,
        )
        if unexpected is not None:
            raise ValueError(
                "Stage-1 output directory is not empty; choose a new directory to avoid "
                "overwriting prior run artifacts"
            )


@contextmanager
def _stage1_output_lock(output_root: Path) -> Iterator[None]:
    """Hold the cooperative Stage-1 output lock for the complete run lifetime."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / STAGE1_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ValueError(f"cannot safely open Stage-1 output lock: {exc}") from exc
    locked = False
    try:
        opened = os.fstat(descriptor)
        linked = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != linked.st_dev
            or opened.st_ino != linked.st_ino
            or opened.st_nlink != 1
        ):
            raise ValueError("Stage-1 output lock is not a private regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Stage-1 output is locked by another training process"
            ) from exc
        locked = True
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _expected_backend_config(
    variant: str,
    requested: RFDETRModelConfig,
) -> RFDETRBackendConfig:
    spec = VARIANT_SPECS[variant]
    return RFDETRBackendConfig(
        variant=variant,
        resolution=requested.resolved_resolution(variant),
        patch_size=spec.patch_size,
        num_windows=spec.num_windows,
        decoder_layers=spec.decoder_layers,
        num_queries=spec.num_queries,
        num_select=spec.num_select,
        amp=True if requested.amp is None else requested.amp,
    )


def _read_stage1_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("Stage-1 resume requires stage1_run.json in its output directory") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Stage-1 run manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("stage") != 1:
        raise ValueError("Stage-1 resume requires Stage-1 run metadata")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("Stage-1 run metadata is missing its configuration")
    return payload


def _validate_resume_checkpoint(
    output_root: Path,
    checkpoint: str | Path,
) -> Path:
    path = Path(checkpoint).expanduser().resolve(strict=True)
    if not path.is_file() or path != output_root / "last.pth":
        raise ValueError("Stage-1 --resume must point to last.pth inside its output directory")
    return path


@contextmanager
def _rolling_checkpoint(
    output_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> Iterator[None]:
    """Redirect RF-DETR 1.5.2 epoch saves to one atomic, resumable checkpoint."""

    import rfdetr.main as backend

    original_save = backend.save_on_master

    def save_checkpoint(weights: Any, filename: Any, *args: Any, **kwargs: Any) -> None:
        path = Path(filename)
        if path.parent != output_root or not path.name.startswith("checkpoint"):
            original_save(weights, filename, *args, **kwargs)
            return
        if path.name != "checkpoint.pth" or not backend.is_main_process():
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output_root, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(weights, stream, *args, **kwargs)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output_root / "last.pth")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        manifest["last_completed_epoch"] = weights["epoch"]
        atomic_json(manifest_path, manifest)

    # Epoch callbacks run after duplicate saves, so intercept the backend writer.
    backend.save_on_master = save_checkpoint
    try:
        yield
    finally:
        backend.save_on_master = original_save


def _train_detector(
    args: argparse.Namespace,
    *,
    data_root: Path,
    output_root: Path,
) -> None:
    config_path = getattr(args, "config", None)
    loaded_config = None
    if config_path is None:
        detector_values: dict[str, Any] = {}
        stage_values: dict[str, Any] = {}
    else:
        from .config import (
            apply_stage_overrides,
            load_config,
            resolve_stage_config,
            validate_config,
        )

        loaded_config = load_config(config_path)
        override_names = (
            "epochs",
            "batch_size",
            "gradient_accumulation",
            "learning_rate",
            "encoder_learning_rate",
            "weight_decay",
            "lr_scheduler",
            "warmup_epochs",
            "multi_scale",
            "expanded_scales",
            "precision",
            "tf32",
        )
        stage_overrides = {
            name: getattr(args, name)
            for name in override_names
            if getattr(args, name, None) is not None
        }
        loaded_config = apply_stage_overrides(loaded_config, 1, stage_overrides)
        if getattr(args, "variant", None) is not None:
            loaded_config["detector"]["variant"] = args.variant
        if getattr(args, "resolution", None) is not None:
            loaded_config["detector"]["resolution"] = args.resolution
        validate_config(loaded_config)
        detector_values = loaded_config["detector"]
        stage_values = resolve_stage_config(loaded_config, 1)

    def option(name: str, fallback: Any, *, config_name: str | None = None) -> Any:
        value = getattr(args, name, None)
        if value is not None:
            return value
        return stage_values.get(config_name or name, fallback)

    variant = getattr(args, "variant", None) or detector_values.get("variant", "seg-2xlarge")
    resolution = getattr(args, "resolution", None)
    if resolution is None:
        resolution = detector_values.get("resolution")
    precision = option("precision", "backend_mixed")
    if precision not in ("backend_mixed", "fp32"):
        raise ValueError("Stage-1 precision must be 'backend_mixed' or 'fp32'")
    tf32 = option("tf32", True)
    if not isinstance(tf32, bool):
        raise TypeError("Stage-1 tf32 must be boolean")
    model_options = RFDETRModelConfig(
        resolution=resolution,
        amp=precision != "fp32",
    )
    # Fail before importing/constructing RF-DETR, which can otherwise download
    # weights as part of its constructor.
    model_options.resolved_resolution(variant)
    train_options = RFDETRTrainConfig(
        epochs=option("epochs", 12),
        batch_size=option("batch_size", 1),
        gradient_accumulation=option("gradient_accumulation", 16),
        workers=option("workers", 4, config_name="workers"),
        seed=42 if loaded_config is None else loaded_config["training"]["seed"],
        learning_rate=option("learning_rate", 1e-4),
        encoder_learning_rate=option("encoder_learning_rate", 1.5e-4),
        weight_decay=option("weight_decay", 1e-4),
        lr_scheduler=option("lr_scheduler", "step"),
        warmup_epochs=option("warmup_epochs", 0.0),
        multi_scale=option("multi_scale", True),
        expanded_scales=option("expanded_scales", True),
    )

    resume_argument = getattr(args, "resume", None)
    weights_path = getattr(args, "weights", None)
    resolved_weights = None
    if weights_path and resume_argument is None:
        resolved_weights = Path(weights_path).expanduser().resolve(strict=True)
        if not resolved_weights.is_file():
            raise ValueError("Stage-1 --weights must point to a checkpoint file")
    expected_backend = _expected_backend_config(variant, model_options)
    run_config = {
        "data_dir": str(data_root),
        "backend": asdict(expected_backend),
        "training": asdict(train_options),
        "precision": precision,
        "tf32": tf32,
    }
    manifest_path = output_root / "stage1_run.json"
    resume_checkpoint = None
    if resume_argument is None:
        manifest = {
            "stage": 1,
            "status": "initialized",
            "config": run_config,
            "initial_weights": None if resolved_weights is None else str(resolved_weights),
            "checkpoint": "last.pth",
            "last_completed_epoch": None,
        }
        atomic_json(manifest_path, manifest)
    else:
        manifest = _read_stage1_manifest(manifest_path)
        if manifest["config"] != run_config:
            raise ValueError(
                "Stage-1 resume settings differ from stage1_run.json; use the original "
                "dataset, model configuration, and training options"
            )
        resume_checkpoint = _validate_resume_checkpoint(output_root, resume_argument)
        manifest.pop("error", None)

    try:
        model_cls = _models()[variant]
        # RF-DETR imports torch before configuring its engine; set the requested
        # backend policy after import and before constructing/training the model.
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        kwargs: dict[str, Any] = {"device": args.device}
        kwargs.update(model_options.constructor_kwargs(variant))
        if resume_checkpoint is not None:
            kwargs["pretrain_weights"] = None
        elif resolved_weights is not None:
            kwargs["pretrain_weights"] = str(resolved_weights)
        random.seed(train_options.seed)
        np.random.seed(train_options.seed)
        torch.manual_seed(train_options.seed)
        model = model_cls(**kwargs)
        backend_config = RFDETRDetector._backend_config(model, variant, model_options)
        if backend_config != expected_backend:
            raise RuntimeError("RF-DETR backend disagrees with the requested configuration")
        manifest["status"] = "running"
        atomic_json(manifest_path, manifest)
        backend_kwargs = {
            "dataset_dir": str(data_root),
            "output_dir": str(output_root),
            **train_options.backend_kwargs(),
            "device": args.device,
            "tensorboard": False,
            "wandb": False,
            "run_test": False,
        }
        if resume_checkpoint is not None:
            backend_kwargs["resume"] = str(resume_checkpoint)
        # RF-DETR reinitializes the class head before its own training seed call.
        random.seed(train_options.seed)
        np.random.seed(train_options.seed)
        torch.manual_seed(train_options.seed)
        with _rolling_checkpoint(output_root, manifest_path, manifest):
            model.train(**backend_kwargs)
        if not (output_root / "last.pth").is_file():
            raise RuntimeError("RF-DETR training finished without saving last.pth")
        manifest["status"] = "completed"
        atomic_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        try:
            atomic_json(manifest_path, manifest)
        except OSError:
            pass
        raise


def train_detector(args: argparse.Namespace) -> None:
    """Train Stage 1 and retain one rolling last.pth checkpoint."""

    data_root = Path(args.data_dir).expanduser().resolve(strict=True)
    output_root = external_output(args.output_dir, data_root)
    resume = getattr(args, "resume", None) is not None
    _protect_stage1_output(output_root, resume=resume)
    with _stage1_output_lock(output_root):
        _protect_stage1_output(output_root, resume=resume)
        _train_detector(args, data_root=data_root, output_root=output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the frozen Falcon RF-DETR stage")
    parser.add_argument("--data-dir", required=True, help="RF-DETR COCO dataset root")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New/empty output, or the existing Stage-1 directory for resume",
    )
    parser.add_argument("--weights", help="RF-DETR segmentation weights")
    parser.add_argument(
        "--resume",
        help="Resume from OUTPUT_DIR/last.pth at the next epoch",
    )
    parser.add_argument("--config", help="Optional Falcon YAML; explicit CLI options win")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--encoder-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr-scheduler", choices=("cosine", "step"))
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--precision", choices=("backend_mixed", "fp32"))
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--multi-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--expanded-scales",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main() -> None:
    train_detector(build_parser().parse_args())


if __name__ == "__main__":
    main()
