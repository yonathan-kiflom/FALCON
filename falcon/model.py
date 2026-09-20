"""Falcon model core with injectable perception backends.

Attach a live detector or pass an in-memory :class:`DetectorBatch` to reuse
frozen detector outputs across tasks on the same image.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .capabilities import ALL_SAFETY_HEADS, SafetyCapabilities
from .regions import MaskAwareRegionEncoder, PatchTokenAggregator
from .safety import SafetyOutput, StructuredSafetyAdapter


@dataclass(frozen=True)
class FalconConfig:
    vision_dim: int
    language_dim: int
    region_dim: int = 1024
    image_size: int = 448
    patch_size: int = 14
    vision_prefix_tokens: int = 1
    roi_size: int = 4
    max_regions: int = 100
    max_text_tokens: int = 256
    text_overflow_policy: str = "error"
    mask_probability_threshold: float = 0.5
    risk_loss_weight: float = 1.0
    presence_loss_weight: float = 1.0
    link_loss_weight: float = 1.0
    safety_capabilities: SafetyCapabilities = SafetyCapabilities()

    def __post_init__(self) -> None:
        if self.vision_dim < 1 or self.language_dim < 1 or self.region_dim < 1:
            raise ValueError("model dimensions must be positive")
        if self.image_size < 1 or self.patch_size < 1 or self.roi_size < 1:
            raise ValueError("image_size, patch_size, and roi_size must be positive")
        if self.image_size % (2 * self.patch_size):
            raise ValueError("image_size must be divisible by twice patch_size")
        if self.vision_prefix_tokens < 0:
            raise ValueError("vision_prefix_tokens cannot be negative")
        if self.max_regions < 1:
            raise ValueError("max_regions must be positive")
        if self.max_text_tokens < 3:
            raise ValueError("max_text_tokens must allow a prompt token, answer token, and EOS")
        if self.text_overflow_policy not in ("error", "truncate"):
            raise ValueError("text_overflow_policy must be error or truncate")
        if not 0.0 <= self.mask_probability_threshold <= 1.0:
            raise ValueError("mask_probability_threshold must lie in [0, 1]")
        for value in (
            self.risk_loss_weight,
            self.presence_loss_weight,
            self.link_loss_weight,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("structured loss weights must be finite and nonnegative")


@dataclass
class DetectorBatch:
    """Batched live-detector output.

    Boxes are absolute ``xyxy`` coordinates in the original images.  Masks have
    shape ``[batch, proposals, height, width]`` and are explicitly declared as
    logits or probabilities.  ``valid`` marks padding; proposal order is owned by
    the detector and must already be deterministic (normally score descending).
    """

    boxes: Tensor
    masks: Tensor
    valid: Tensor | None = None
    scores: Tensor | None = None
    image_sizes: Tensor | None = None
    masks_are_logits: bool = False
    mask_sizes: Tensor | None = None

    def batched(self) -> DetectorBatch:
        boxes = self.boxes.unsqueeze(0) if self.boxes.ndim == 2 else self.boxes
        masks = self.masks.unsqueeze(0) if self.masks.ndim == 3 else self.masks
        valid = self.valid
        scores = self.scores
        image_sizes = self.image_sizes
        mask_sizes = self.mask_sizes
        if valid is not None and valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if scores is not None and scores.ndim == 1:
            scores = scores.unsqueeze(0)
        if image_sizes is not None and image_sizes.ndim == 1:
            image_sizes = image_sizes.unsqueeze(0)
        if mask_sizes is not None and mask_sizes.ndim == 1:
            mask_sizes = mask_sizes.unsqueeze(0)
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("detector boxes must have shape [batch, proposals, 4]")
        if masks.ndim != 4 or masks.shape[:2] != boxes.shape[:2]:
            raise ValueError("detector masks must have shape [batch, proposals, height, width]")
        if valid is None:
            valid = torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
        elif valid.shape != boxes.shape[:2]:
            raise ValueError("detector valid mask must have shape [batch, proposals]")
        if scores is not None and scores.shape != boxes.shape[:2]:
            raise ValueError("detector scores must have shape [batch, proposals]")
        if image_sizes is not None and image_sizes.shape != (boxes.shape[0], 2):
            raise ValueError("detector image_sizes must have shape [batch, 2]")
        if mask_sizes is None:
            mask_sizes = torch.tensor(
                [list(masks.shape[-2:])] * boxes.shape[0], device=masks.device, dtype=torch.long
            )
        if mask_sizes.shape != (boxes.shape[0], 2) or bool((mask_sizes <= 0).any()):
            raise ValueError("detector mask_sizes must have shape [batch, 2] and be positive")
        if bool((mask_sizes > mask_sizes.new_tensor(masks.shape[-2:])).any()):
            raise ValueError("mask_sizes cannot exceed the padded mask canvas")
        return DetectorBatch(
            boxes=boxes,
            masks=masks,
            valid=valid.to(dtype=torch.bool),
            scores=scores,
            image_sizes=image_sizes,
            masks_are_logits=self.masks_are_logits,
            mask_sizes=mask_sizes,
        )

    def to(self, device: torch.device) -> DetectorBatch:
        return DetectorBatch(
            boxes=self.boxes.to(device),
            masks=self.masks.to(device),
            valid=None if self.valid is None else self.valid.to(device),
            scores=None if self.scores is None else self.scores.to(device),
            image_sizes=None if self.image_sizes is None else self.image_sizes.to(device),
            masks_are_logits=self.masks_are_logits,
            mask_sizes=None if self.mask_sizes is None else self.mask_sizes.to(device),
        )

    @classmethod
    def pack(cls, outputs: Sequence[DetectorBatch]) -> DetectorBatch:
        """Pad proposals and canvases without resampling original instance masks."""
        if not outputs:
            raise ValueError("cannot pack an empty detector batch")
        outputs = [output.batched() for output in outputs]
        if any(output.boxes.shape[0] != 1 for output in outputs):
            raise ValueError("pack expects single-image detector outputs")
        if len(outputs) == 1:
            return outputs[0]
        first = outputs[0]
        if any(output.masks_are_logits != first.masks_are_logits for output in outputs):
            raise ValueError("all masks in a batch must use the same representation")
        if any((output.scores is None) != (first.scores is None) for output in outputs):
            raise ValueError("all outputs must consistently include proposal scores")
        if any(output.image_sizes is None for output in outputs):
            raise ValueError("original image sizes are required for a multi-image batch")
        count = max(output.boxes.shape[1] for output in outputs)
        height = max(output.masks.shape[-2] for output in outputs)
        width = max(output.masks.shape[-1] for output in outputs)
        boxes = first.boxes.new_zeros((len(outputs), count, 4))
        masks = first.masks.new_zeros((len(outputs), count, height, width))
        valid = torch.zeros((len(outputs), count), device=first.boxes.device, dtype=torch.bool)
        scores = None if first.scores is None else first.scores.new_zeros((len(outputs), count))
        for index, output in enumerate(outputs):
            n, h, w = output.masks.shape[1:]
            boxes[index, :n] = output.boxes[0].to(boxes)
            masks[index, :n, :h, :w] = output.masks[0].to(masks)
            valid[index, :n] = output.valid[0].to(valid)
            if scores is not None:
                scores[index, :n] = output.scores[0].to(scores)
        return cls(
            boxes,
            masks,
            valid,
            scores,
            torch.cat([output.image_sizes.to(boxes.device) for output in outputs]),
            first.masks_are_logits,
            torch.cat([output.mask_sizes.to(boxes.device) for output in outputs]),
        )


@dataclass
class VisionFeatures:
    """Frozen DINO patch tokens that may be reused for tasks on one image."""

    patch_tokens: Tensor
    grid_size: tuple[int, int]

    def to(self, device: torch.device) -> VisionFeatures:
        return VisionFeatures(self.patch_tokens.to(device), self.grid_size)


@dataclass
class PrefixBatch:
    inputs_embeds: Tensor
    attention_mask: Tensor
    labels: Tensor | None
    image_token_count: int
    region_token_count: int
    safety_token_count: int


@dataclass
class FalconEncoding:
    prefix: PrefixBatch
    safety: SafetyOutput
    region_embeddings: Tensor
    detector_output: DetectorBatch


@dataclass
class FalconOutput:
    loss: Tensor | None
    language_loss: Tensor | None
    structured_losses: dict[str, Tensor]
    language_output: Any
    safety: SafetyOutput
    region_embeddings: Tensor
    detector_output: DetectorBatch
    prefix: PrefixBatch


@dataclass
class ResolvedRegion:
    reference: str
    proposal_index: int
    box_xyxy: Tensor
    mask: Tensor
    score: Tensor | None


_REGION_REFERENCE = re.compile(r"(?<![A-Za-z0-9_])region_(\d{3})(?![A-Za-z0-9_])")


def parse_region_references(text: str, *, max_regions: int = 100) -> tuple[int, ...]:
    """Parse unique ``region_NNN`` references in first-occurrence order."""

    found: list[int] = []
    seen: set[int] = set()
    for match in _REGION_REFERENCE.finditer(text):
        index = int(match.group(1))
        if index < max_regions and index not in seen:
            found.append(index)
            seen.add(index)
    return tuple(found)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


class FalconModel(nn.Module):
    """DINOv2 + live RF-DETR proposals + SSA + a Vicuna/Llama decoder."""

    def __init__(
        self,
        config: FalconConfig,
        *,
        vision_encoder: nn.Module,
        language_model: nn.Module,
        detector: Any | None = None,
        image_processor: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.detector = detector
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.patch_aggregator = PatchTokenAggregator(config.vision_dim, config.region_dim)
        self.region_encoder = MaskAwareRegionEncoder(
            config.region_dim,
            config.region_dim,
            roi_size=config.roi_size,
            probability_threshold=config.mask_probability_threshold,
        )
        self.safety_adapter = StructuredSafetyAdapter(
            config.region_dim, config.language_dim, capabilities=config.safety_capabilities
        )
        self.image_projection = nn.Linear(config.region_dim, config.language_dim)
        self.region_projection = nn.Linear(config.region_dim, config.language_dim)
        self.region_index_embedding = nn.Embedding(config.max_regions, config.language_dim)
        self.training_stage = 2
        self.set_training_stage(2)

    @classmethod
    def from_pretrained(
        cls,
        *,
        vision_model_name_or_path: str,
        language_model_name_or_path: str,
        detector: Any | None = None,
        region_dim: int = 1024,
        image_size: int = 448,
        patch_size: int = 14,
        roi_size: int = 4,
        max_regions: int = 100,
        max_text_tokens: int = 256,
        text_overflow_policy: str = "error",
        risk_loss_weight: float = 1.0,
        presence_loss_weight: float = 1.0,
        link_loss_weight: float = 1.0,
        safety_capabilities: SafetyCapabilities = ALL_SAFETY_HEADS,
        vision_kwargs: Mapping[str, Any] | None = None,
        language_kwargs: Mapping[str, Any] | None = None,
    ) -> FalconModel:
        """Construct the two pretrained encoders without hard-coded model paths."""

        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        vision_encoder = AutoModel.from_pretrained(
            vision_model_name_or_path,
            **dict(vision_kwargs or {}),
        )
        language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name_or_path,
            **dict(language_kwargs or {}),
        )
        image_processor = AutoImageProcessor.from_pretrained(
            vision_model_name_or_path,
            size={"height": image_size, "width": image_size},
            crop_size={"height": image_size, "width": image_size},
            use_fast=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(language_model_name_or_path, use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        vision_dim = int(vision_encoder.config.hidden_size)
        language_dim = int(language_model.config.hidden_size)
        return cls(
            FalconConfig(
                vision_dim=vision_dim,
                language_dim=language_dim,
                region_dim=region_dim,
                image_size=image_size,
                patch_size=patch_size,
                roi_size=roi_size,
                max_regions=max_regions,
                max_text_tokens=max_text_tokens,
                text_overflow_policy=text_overflow_policy,
                risk_loss_weight=risk_loss_weight,
                presence_loss_weight=presence_loss_weight,
                link_loss_weight=link_loss_weight,
                safety_capabilities=safety_capabilities,
            ),
            vision_encoder=vision_encoder,
            language_model=language_model,
            detector=detector,
            image_processor=image_processor,
            tokenizer=tokenizer,
        )

    @property
    def device(self) -> torch.device:
        return next(self.patch_aggregator.parameters()).device

    @staticmethod
    def _as_image_batch(images: Any) -> list[Any]:
        if isinstance(images, list | tuple):
            return list(images)
        if isinstance(images, Tensor) and images.ndim == 4:
            return list(images.unbind(0))
        if hasattr(images, "ndim") and images.ndim == 4:
            return [images[index] for index in range(len(images))]
        return [images]

    @staticmethod
    def _image_size(image: Any) -> tuple[int, int]:
        # PIL exposes (width, height); NumPy and torch expose a shape.
        if hasattr(image, "size") and isinstance(image.size, tuple):
            width, height = image.size
            return int(height), int(width)
        shape = tuple(image.shape) if hasattr(image, "shape") else ()
        if len(shape) < 2:
            raise ValueError("cannot infer original image size")
        if len(shape) == 2:
            return int(shape[0]), int(shape[1])
        if shape[0] in (1, 3, 4):
            return int(shape[-2]), int(shape[-1])
        return int(shape[0]), int(shape[1])

    @staticmethod
    def _normalise_text_batch(
        texts: str | Sequence[str] | None,
        batch_size: int,
        name: str,
    ) -> list[str] | None:
        if texts is None:
            return None
        values = [texts] * batch_size if isinstance(texts, str) else list(texts)
        if len(values) != batch_size:
            raise ValueError(f"{name} must contain one string per image")
        return values

    def _process_images(self, images: list[Any]) -> Tensor:
        if self.image_processor is None:
            raise RuntimeError("an AutoImageProcessor-compatible object is required for raw images")
        processed = self.image_processor(images=images, return_tensors="pt")
        pixel_values = _field(processed, "pixel_values")
        if pixel_values is None:
            raise ValueError("image processor did not return pixel_values")
        if tuple(pixel_values.shape[-2:]) != (self.config.image_size, self.config.image_size):
            raise ValueError(
                "image processor returned "
                f"{tuple(pixel_values.shape[-2:])}; expected "
                f"{self.config.image_size}x{self.config.image_size}"
            )
        return pixel_values.to(self.device)

    def _encode_one_text(self, text: str, *, special_tokens: bool) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("a Vicuna/Llama tokenizer is required for string prompts")
        encoded = self.tokenizer(
            text,
            add_special_tokens=special_tokens,
            return_attention_mask=False,
        )
        token_ids = _field(encoded, "input_ids")
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.reshape(-1).tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return list(token_ids)

    def _answer_token_ids(self, answer: str) -> list[int]:
        """Use the same nonempty-answer/EOS contract for preflight and training."""
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("the tokenizer must define eos_token_id for supervised answers")
        answer_ids = self._encode_one_text(answer, special_tokens=False)
        content = answer_ids[:-1] if answer_ids and answer_ids[-1] == eos_id else answer_ids
        if not content:
            raise ValueError("a supervised answer must contain at least one token")
        return content + [eos_id]

    def text_token_counts(self, prompt: str, answer: str | None = None) -> dict[str, int]:
        """Count untruncated text on the CPU without running either backbone.

        ``answer=None`` measures a query whose eventual proposal-dependent target
        is unknown. It must not be interpreted as a full training-length bound.
        """
        prompt_count = len(self._encode_one_text(prompt, special_tokens=True))
        answer_count = 0 if answer is None else len(self._answer_token_ids(answer))
        return {
            "prompt_tokens": prompt_count,
            "answer_tokens": answer_count,
            "total_tokens": prompt_count + answer_count,
        }

    def _tokenize(
        self,
        prompts: list[str],
        answers: list[str] | None,
        supervision: Sequence[bool] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        sequences: list[list[int]] = []
        label_sequences: list[list[int]] | None = [] if answers is not None else None
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        def truncate_prompt(token_ids: list[int], budget: int) -> list[int]:
            if len(token_ids) <= budget:
                return token_ids
            if self.config.text_overflow_policy == "error":
                raise ValueError(
                    f"Prompt needs {len(token_ids)} tokens but only {budget} fit; "
                    "increase max_text_tokens instead of silently removing query context"
                )
            if budget <= 0:
                return []
            if budget == 1:
                return token_ids[:1]
            # Retain BOS and the recent USER/ASSISTANT suffix.
            return token_ids[:1] + token_ids[-(budget - 1) :]

        for index, prompt in enumerate(prompts):
            prompt_ids = self._encode_one_text(prompt, special_tokens=True)
            if answers is None or (supervision is not None and not supervision[index]):
                prompt_sequence = truncate_prompt(prompt_ids, self.config.max_text_tokens)
                sequences.append(prompt_sequence)
                if label_sequences is not None:
                    label_sequences.append([-100] * len(prompt_sequence))
                continue
            answer_ids = self._answer_token_ids(answers[index])
            if (
                self.config.text_overflow_policy == "error"
                and len(prompt_ids) + len(answer_ids) > self.config.max_text_tokens
            ):
                raise ValueError(
                    f"Prompt+answer needs {len(prompt_ids) + len(answer_ids)} tokens "
                    f"({len(prompt_ids)} prompt, {len(answer_ids)} answer including EOS), "
                    f"exceeding max_text_tokens={self.config.max_text_tokens}; increase "
                    "the budget to preserve the full query and target"
                )
            answer_budget = self.config.max_text_tokens - 1
            if len(answer_ids) > answer_budget:
                answer_ids = answer_ids[: answer_budget - 1] + [eos_id]
            prompt_budget = self.config.max_text_tokens - len(answer_ids)
            prompt_ids = truncate_prompt(prompt_ids, prompt_budget)
            sequences.append(prompt_ids + answer_ids)
            assert label_sequences is not None
            label_sequences.append([-100] * len(prompt_ids) + answer_ids)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100) if label_sequences is not None else None
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, :length] = torch.as_tensor(sequence)
            attention_mask[row, :length] = 1
            if labels is not None and label_sequences is not None:
                labels[row, :length] = torch.as_tensor(label_sequences[row])
        return (
            input_ids.to(self.device),
            attention_mask.to(self.device),
            (None if labels is None else labels.to(self.device)),
        )

    @staticmethod
    def _coerce_detector_output(value: Any) -> DetectorBatch:
        if isinstance(value, list | tuple):
            return DetectorBatch.pack([FalconModel._coerce_detector_output(item) for item in value])
        if isinstance(value, DetectorBatch):
            return value.batched()
        boxes = _field(value, "boxes")
        if boxes is None:
            boxes = _field(value, "boxes_xyxy")
        masks = _field(value, "masks")
        if boxes is None or masks is None:
            raise TypeError("detector output must expose boxes/boxes_xyxy and masks")
        image_sizes = _field(value, "image_sizes")
        if image_sizes is None:
            image_sizes = _field(value, "original_hw")
        valid = _field(value, "valid")
        scores = _field(value, "scores")
        return DetectorBatch(
            boxes=torch.as_tensor(boxes),
            masks=torch.as_tensor(masks),
            valid=None if valid is None else torch.as_tensor(valid),
            scores=None if scores is None else torch.as_tensor(scores),
            image_sizes=None if image_sizes is None else torch.as_tensor(image_sizes),
            masks_are_logits=bool(_field(value, "masks_are_logits") or False),
            mask_sizes=_field(value, "mask_sizes"),
        ).batched()

    def _detect(self, images: list[Any]) -> DetectorBatch:
        if self.detector is None:
            raise RuntimeError("attach a live detector or provide detector_outputs")
        if isinstance(self.detector, nn.Module):
            self.detector.eval()
        with torch.no_grad():
            outputs = [self._coerce_detector_output(self.detector(image)) for image in images]
        for image, output in zip(images, outputs, strict=False):
            if output.image_sizes is None:
                output.image_sizes = torch.as_tensor([self._image_size(image)])
        return DetectorBatch.pack(outputs)

    def _extract_vision_features(self, pixel_values: Tensor) -> VisionFeatures:
        self.vision_encoder.eval()
        with torch.no_grad():
            encoded = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
            hidden_state = _field(encoded, "last_hidden_state")
            if hidden_state is None:
                raise ValueError("vision encoder did not return last_hidden_state")
            patch_tokens = hidden_state[:, self.config.vision_prefix_tokens :].detach()
        pixel_height, pixel_width = pixel_values.shape[-2:]
        grid_size = (
            pixel_height // self.config.patch_size,
            pixel_width // self.config.patch_size,
        )
        return VisionFeatures(patch_tokens, grid_size)

    def extract_vision_features(self, images: Any) -> VisionFeatures:
        """Run frozen DINO once for one or more raw images."""

        image_batch = self._as_image_batch(images)
        return self._extract_vision_features(self._process_images(image_batch))

    def encode_vision_features(self, features: VisionFeatures) -> tuple[Tensor, Tensor]:
        """Apply the trainable aggregation and image projection layers."""

        features = features.to(self.device)
        feature_map = self.patch_aggregator(features.patch_tokens, features.grid_size)
        image_tokens = feature_map.flatten(start_dim=2).transpose(1, 2)
        return feature_map, self.image_projection(image_tokens)

    def encode_images(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        """Return the aggregated feature map and projected DINO image tokens."""

        return self.encode_vision_features(self._extract_vision_features(pixel_values))

    def build_prefix(
        self,
        *,
        image_tokens: Tensor,
        region_tokens: Tensor,
        region_valid: Tensor,
        safety_tokens: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None,
    ) -> PrefixBatch:
        """Build ``[image | region | SSA | text]`` inputs for Vicuna/Llama."""

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        dtype = text_embeddings.dtype
        image_tokens = image_tokens.to(dtype=dtype)
        region_tokens = region_tokens.to(dtype=dtype)
        safety_tokens = safety_tokens.to(dtype=dtype)
        inputs_embeds = torch.cat(
            (image_tokens, region_tokens, safety_tokens, text_embeddings),
            dim=1,
        )

        batch = input_ids.shape[0]
        image_mask = torch.ones(
            (batch, image_tokens.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        safety_mask = torch.ones(
            (batch, safety_tokens.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        prefix_attention = torch.cat(
            (image_mask, region_valid.to(attention_mask.dtype), safety_mask, attention_mask),
            dim=1,
        )
        prefix_labels = None
        if labels is not None:
            ignored = torch.full(
                (batch, image_tokens.shape[1] + region_tokens.shape[1] + safety_tokens.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            prefix_labels = torch.cat((ignored, labels), dim=1)
        return PrefixBatch(
            inputs_embeds=inputs_embeds,
            attention_mask=prefix_attention,
            labels=prefix_labels,
            image_token_count=image_tokens.shape[1],
            region_token_count=region_tokens.shape[1],
            safety_token_count=safety_tokens.shape[1],
        )

    def prepare_multimodal_inputs(
        self,
        *,
        images: Any,
        prompts: str | Sequence[str],
        answers: str | Sequence[str] | None = None,
        detector_outputs: DetectorBatch | Mapping[str, Any] | Any | None = None,
        vision_features: VisionFeatures | None = None,
        language_supervision: Tensor | Sequence[bool] | None = None,
    ) -> FalconEncoding:
        image_batch = self._as_image_batch(images)
        prompts_batch = self._normalise_text_batch(prompts, len(image_batch), "prompts")
        answers_batch = self._normalise_text_batch(answers, len(image_batch), "answers")
        assert prompts_batch is not None

        supervision = None
        if language_supervision is not None:
            supplied = torch.as_tensor(language_supervision)
            if supplied.dtype != torch.bool or supplied.shape != (len(image_batch),):
                raise ValueError("language_supervision must contain one boolean per example")
            if answers_batch is None:
                raise ValueError("language_supervision requires answer labels")
            supervision = supplied.tolist()
        input_ids, attention_mask, labels = self._tokenize(
            prompts_batch, answers_batch, supervision
        )
        detected = (
            self._detect(image_batch)
            if detector_outputs is None
            else self._coerce_detector_output(detector_outputs)
        ).to(self.device)
        if detected.image_sizes is None:
            detected.image_sizes = torch.as_tensor(
                [self._image_size(image) for image in image_batch],
                device=self.device,
            )
        if detected.boxes.shape[0] != len(image_batch):
            raise ValueError("detector output batch size does not match the image batch")

        features = (
            self.extract_vision_features(image_batch)
            if vision_features is None
            else vision_features.to(self.device)
        )
        if features.patch_tokens.shape[0] != len(image_batch):
            raise ValueError("vision feature batch size does not match the image batch")
        feature_map, image_tokens = self.encode_vision_features(features)
        assert detected.valid is not None and detected.image_sizes is not None
        region_embeddings = self.region_encoder(
            feature_map,
            detected.boxes,
            detected.masks,
            detected.image_sizes,
            detected.valid,
            masks_are_logits=detected.masks_are_logits,
            mask_sizes=detected.mask_sizes,
        )
        safety = self.safety_adapter(region_embeddings, detected.valid)
        region_tokens = self.region_projection(region_embeddings)
        proposal_count = region_tokens.shape[1]
        if proposal_count > self.config.max_regions:
            raise ValueError(
                f"detector returned {proposal_count} proposals; "
                f"max_regions={self.config.max_regions}"
            )
        region_indices = torch.arange(proposal_count, device=self.device)
        region_tokens = region_tokens + self.region_index_embedding(region_indices).unsqueeze(0)
        prefix = self.build_prefix(
            image_tokens=image_tokens,
            region_tokens=region_tokens,
            region_valid=detected.valid,
            safety_tokens=safety.tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self._validate_context_budget(prefix)
        return FalconEncoding(
            prefix=prefix,
            safety=safety,
            region_embeddings=region_embeddings,
            detector_output=detected,
        )

    def _validate_context_budget(self, prefix: PrefixBatch, *, additional_tokens: int = 0) -> None:
        """Respect the loaded language model's declared positional context."""
        language_config = getattr(self.language_model, "config", None)
        limit = getattr(language_config, "max_position_embeddings", None)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return
        needed = prefix.inputs_embeds.shape[1] + additional_tokens
        if needed > limit:
            raise ValueError(
                f"Multimodal context needs {needed} positions including image/region/safety "
                f"tokens and {additional_tokens} requested new tokens, but the language model "
                f"declares max_position_embeddings={limit}; reduce the text/generation budget "
                "or use a verified longer-context backbone"
            )

    def forward(
        self,
        *,
        images: Any,
        prompts: str | Sequence[str],
        answers: str | Sequence[str] | None = None,
        targets: Mapping[str, Any] | None = None,
        detector_outputs: DetectorBatch | Mapping[str, Any] | Any | None = None,
        vision_features: VisionFeatures | None = None,
        language_supervision: Tensor | Sequence[bool] | None = None,
    ) -> FalconOutput:
        """Run raw images through live detection, DINOv2, SSA, and Vicuna/Llama."""

        encoded = self.prepare_multimodal_inputs(
            images=images,
            prompts=prompts,
            answers=answers,
            detector_outputs=detector_outputs,
            vision_features=vision_features,
            language_supervision=language_supervision,
        )
        prefix = encoded.prefix
        if language_supervision is not None:
            supervision = torch.as_tensor(language_supervision, device=self.device)
            if supervision.dtype != torch.bool or supervision.shape != (
                prefix.inputs_embeds.shape[0],
            ):
                raise ValueError("language_supervision must contain one boolean per example")
            if prefix.labels is None:
                raise ValueError("language_supervision requires answer labels")
            prefix.labels = prefix.labels.masked_fill(~supervision[:, None], -100)
            # Causal-LM cross entropy over zero labels is NaN in common backends.
            # A batch with no transportable targets still trains available SSA heads.
            if not bool(supervision.any()):
                prefix.labels = None
        language_output = self.language_model(
            inputs_embeds=prefix.inputs_embeds,
            attention_mask=prefix.attention_mask,
            labels=prefix.labels,
            return_dict=True,
        )
        language_loss = _field(language_output, "loss")
        structured_losses: dict[str, Tensor] = {}
        total_loss = language_loss
        if targets is not None:
            missing = {"risk", "presence", "links"}.difference(targets)
            if missing:
                raise ValueError(f"structured targets are missing: {sorted(missing)}")
            structured_losses = self.safety_adapter.loss(
                encoded.safety,
                risk=self._target_tensor(targets["risk"]),
                presence=self._target_tensor(targets["presence"]),
                links=self._target_tensor(targets["links"]),
            )
            structured_total = (
                self.config.risk_loss_weight * structured_losses["risk"]
                + self.config.presence_loss_weight * structured_losses["presence"]
                + self.config.link_loss_weight * structured_losses["links"]
            )
            total_loss = structured_total if total_loss is None else total_loss + structured_total
        if total_loss is not None and self.training and not total_loss.requires_grad:
            # An ablation can disable every structured head while a detector miss
            # masks all language targets. Keep a genuine differentiable zero,
            # not an empty-mask label or NaN, for the distributed training step.
            total_loss = total_loss + prefix.inputs_embeds.sum() * 0.0
        return FalconOutput(
            loss=total_loss,
            language_loss=language_loss,
            structured_losses=structured_losses,
            language_output=language_output,
            safety=encoded.safety,
            region_embeddings=encoded.region_embeddings,
            detector_output=encoded.detector_output,
            prefix=prefix,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        images: Any,
        prompts: str | Sequence[str],
        detector_outputs: DetectorBatch | Mapping[str, Any] | Any | None = None,
        vision_features: VisionFeatures | None = None,
        **generation_kwargs: Any,
    ) -> Tensor:
        encoded = self.prepare_multimodal_inputs(
            images=images,
            prompts=prompts,
            detector_outputs=detector_outputs,
            vision_features=vision_features,
        )
        new_tokens = generation_kwargs.get("max_new_tokens")
        if new_tokens is not None:
            if isinstance(new_tokens, bool) or not isinstance(new_tokens, int) or new_tokens < 1:
                raise ValueError("max_new_tokens must be a positive integer")
            self._validate_context_budget(encoded.prefix, additional_tokens=new_tokens)
        return self.language_model.generate(
            inputs_embeds=encoded.prefix.inputs_embeds,
            attention_mask=encoded.prefix.attention_mask,
            **generation_kwargs,
        )

    def enable_lora(
        self,
        *,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> None:
        """Attach PEFT LoRA adapters to Vicuna/Llama attention projections."""

        from peft import LoraConfig, TaskType, get_peft_model

        if any("lora_" in name for name, _ in self.language_model.named_parameters()):
            raise RuntimeError("LoRA adapters are already attached")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=list(target_modules),
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        # Attaching PEFT makes adapters trainable by default. Preserve the
        # current paper stage until the caller explicitly enters Stage 3.
        self.set_training_stage(self.training_stage)

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def set_training_stage(self, stage: int) -> None:
        """Apply paper freezing: Stage 2 projections+SSA; Stage 3 adds LoRA."""

        if stage not in (2, 3):
            raise ValueError("the multimodal model supports training stage 2 or 3")
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (
            self.patch_aggregator,
            self.region_encoder,
            self.safety_adapter,
            self.image_projection,
            self.region_projection,
            self.region_index_embedding,
        ):
            self._set_module_trainable(module, True)
        self.safety_adapter.freeze_unavailable_heads()

        if stage == 3:
            lora_parameters = [
                parameter
                for name, parameter in self.language_model.named_parameters()
                if "lora_" in name
            ]
            if not lora_parameters:
                raise RuntimeError("Stage 3 requires enable_lora() before set_training_stage(3)")
            for parameter in lora_parameters:
                parameter.requires_grad = True
        self.training_stage = stage
        self.vision_encoder.eval()
        if isinstance(self.detector, nn.Module):
            self.detector.eval()
        if stage == 2:
            self.language_model.eval()
        elif self.training:
            self.language_model.train()

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def train(self, mode: bool = True) -> FalconModel:
        super().train(mode)
        self.vision_encoder.eval()
        if isinstance(self.detector, nn.Module):
            self.detector.eval()
        if self.training_stage == 2:
            self.language_model.eval()
        return self

    def _target_tensor(self, value: Any) -> Tensor:
        """Convert nullable structured labels to tensors, representing null as NaN."""

        def replace_nulls(item: Any) -> Any:
            if item is None:
                return float("nan")
            if isinstance(item, list | tuple):
                return [replace_nulls(child) for child in item]
            return item

        if isinstance(value, Tensor):
            return value.to(self.device)
        return torch.as_tensor(replace_nulls(value), dtype=torch.float32, device=self.device)

    def resolve_region_references(
        self,
        text: str,
        detector_output: DetectorBatch | Any,
        *,
        batch_index: int = 0,
    ) -> tuple[ResolvedRegion, ...]:
        """Resolve generated region labels to the exact proposal box and mask."""

        detected = self._coerce_detector_output(detector_output)
        if not 0 <= batch_index < detected.boxes.shape[0]:
            raise IndexError("batch_index is outside the detector batch")
        assert detected.valid is not None
        resolved = []
        for index in parse_region_references(text, max_regions=self.config.max_regions):
            if index >= detected.boxes.shape[1] or not bool(detected.valid[batch_index, index]):
                continue
            score = None if detected.scores is None else detected.scores[batch_index, index]
            height, width = (int(value) for value in detected.mask_sizes[batch_index])
            resolved.append(
                ResolvedRegion(
                    reference=f"region_{index:03d}",
                    proposal_index=index,
                    box_xyxy=detected.boxes[batch_index, index],
                    mask=detected.masks[batch_index, index, :height, :width],
                    score=score,
                )
            )
        return tuple(resolved)
