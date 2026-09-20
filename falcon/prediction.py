"""Shared live prediction for local and Hugging Face FALCON models."""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .capabilities import SafetyCapabilities
from .grounding import (
    PanopticPrediction,
    SegmentationPrediction,
    resolve_panoptic,
    resolve_segmentation,
    select_panoptic_category,
)
from .model import FalconModel


def _candidate_relations_from_answer(text: str) -> list[dict[str, str]] | None:
    """Extract only explicit observable geometry from a structured model answer.

    Safety examples are trained to emit a compact JSON object.  Risk and
    physical links may be unavailable, so this parser never derives either
    from geometry or component presence.
    """

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or "candidate_relations" not in payload:
        return None
    raw = payload["candidate_relations"]
    if not isinstance(raw, list):
        raise ValueError("generated candidate_relations must be an array")
    horizontal_values = {
        "left_of",
        "right_of",
        "horizontally_aligned",
        "left of",
        "right of",
        "horizontally aligned",
    }
    vertical_values = {"above", "below", "vertically_aligned", "vertically aligned"}
    parsed: list[dict[str, str]] = []
    for index, relation in enumerate(raw):
        if not isinstance(relation, dict):
            raise ValueError(
                f"generated candidate_relations[{index}] must be an object"
            )
        if relation.get("connection_observed") not in (None, False):
            raise ValueError(
                f"generated candidate_relations[{index}] asserts an unavailable "
                "physical connection"
            )
        source = relation.get("src_component")
        target = relation.get("dst_component")
        horizontal = relation.get("horizontal_relation")
        vertical = relation.get("vertical_relation")
        if not all(
            isinstance(value, str) and value.strip() for value in (source, target)
        ):
            raise ValueError(
                f"generated candidate_relations[{index}] requires component names"
            )
        if horizontal is not None and horizontal not in horizontal_values:
            raise ValueError(
                f"generated candidate_relations[{index}] has unsupported "
                "horizontal geometry"
            )
        if vertical is not None and vertical not in vertical_values:
            raise ValueError(
                f"generated candidate_relations[{index}] has unsupported vertical geometry"
            )
        if horizontal is None and vertical is None:
            raise ValueError(
                f"generated candidate_relations[{index}] requires at least one "
                "supported geometry axis"
            )
        parsed_relation = {
            "src_component": source.strip().casefold(),
            "dst_component": target.strip().casefold(),
        }
        if horizontal is not None:
            parsed_relation["horizontal_relation"] = (
                str(horizontal).replace(" ", "_").casefold()
            )
        if vertical is not None:
            parsed_relation["vertical_relation"] = (
                str(vertical).replace(" ", "_").casefold()
            )
        parsed.append(parsed_relation)
    return parsed


def _safety_prediction(safety: Any) -> dict[str, Any]:
    prediction = getattr(safety, "prediction", None)
    if callable(prediction):
        value = prediction(0)
        if not isinstance(value, Mapping):
            raise TypeError("SafetyOutput.prediction() must return a mapping")
        return dict(value)

    # Honor capabilities for callers without a prediction serializer.
    capabilities = getattr(safety, "capabilities", SafetyCapabilities())
    if not isinstance(capabilities, SafetyCapabilities):
        raise TypeError("safety.capabilities must be SafetyCapabilities")
    risk = (
        float(safety.risk_probability[0].detach().float().cpu())
        if capabilities.risk
        else None
    )
    presence = [
        float(value.detach().float().cpu()) if enabled else None
        for value, enabled in zip(
            safety.presence_probabilities[0], capabilities.presence, strict=True
        )
    ]
    links = [
        float(value.detach().float().cpu()) if enabled else None
        for value, enabled in zip(
            safety.link_probabilities[0], capabilities.links, strict=True
        )
    ]
    return {
        "risk": risk,
        "presence": presence,
        "links": links,
        "capabilities": capabilities.as_dict(),
    }


def _generate(
    model: FalconModel,
    *,
    image: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], Any]:
    """Run generation once and retain the exact selected detector proposals."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if isinstance(image, str | os.PathLike):
        # Transformers image processors consume decoded pixels, not path
        # strings.  Detach from the file handle once here so RF-DETR, DINO, and
        # native-size bookkeeping all see the exact same RGB image object.
        with Image.open(Path(image).expanduser()) as opened:
            image = opened.convert("RGB").copy()
    formatted_prompt = f"USER: {prompt}\nASSISTANT:"
    encoded = model.prepare_multimodal_inputs(images=image, prompts=formatted_prompt)
    # We call the language model directly in order to preserve the exact
    # detector output for grounding resolution, so mirror FalconModel.generate's
    # context check before allocating the generation cache.
    model._validate_context_budget(
        encoded.prefix,
        additional_tokens=max_new_tokens,
    )
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "eos_token_id": model.tokenizer.eos_token_id,
        "pad_token_id": model.tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    generated = model.language_model.generate(
        inputs_embeds=encoded.prefix.inputs_embeds,
        attention_mask=encoded.prefix.attention_mask,
        **generation_kwargs,
    )
    text = model.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    candidate_relations = _candidate_relations_from_answer(text)
    structured_safety = _safety_prediction(encoded.safety)
    structured_safety["inference_ablation"] = {
        "name": getattr(model, "inference_ssa_ablation", "none"),
        "scope": "diagnostic",
        "paper_trained_ablation_claim": False,
    }
    if candidate_relations is not None:
        structured_safety["candidate_relations"] = candidate_relations
    return {
        "prompt": prompt,
        "answer": text,
        "safety": structured_safety,
    }, encoded


def _resolved_region_payload(
    model: FalconModel,
    text: str,
    detector_output: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    resolved = model.resolve_region_references(text, detector_output)
    region_ids = [region.proposal_index for region in resolved]
    boxes = (
        torch.stack([region.box_xyxy for region in resolved]).float().cpu()
        if resolved
        else torch.empty(0, 4)
    )
    scores = [
        None if region.score is None else float(region.score.float().cpu())
        for region in resolved
    ]
    if resolved:
        binary_masks = []
        for region in resolved:
            mask = region.mask.detach()
            if mask.dtype == torch.bool:
                binary = mask
            elif detector_output.masks_are_logits:
                binary = mask >= 0.0
            else:
                binary = mask >= model.config.mask_probability_threshold
            binary_masks.append(binary.to(dtype=torch.bool))
        masks = torch.stack(binary_masks).cpu().numpy()
    else:
        image_sizes = getattr(detector_output, "image_sizes", None)
        if image_sizes is None:
            raise ValueError(
                "detector output lacks native image size for empty prediction"
            )
        sizes = torch.as_tensor(image_sizes).detach().cpu()
        if sizes.shape != (1, 2):
            raise ValueError("single-image inference requires one detector image size")
        height, width = (int(value) for value in sizes[0].tolist())
        if height < 1 or width < 1:
            raise ValueError("detector output contains an invalid native image size")
        masks = np.empty((0, height, width), dtype=bool)
    return {
        "region_ids": region_ids,
        "boxes_xyxy": boxes.tolist(),
        "scores": scores,
    }, masks


@torch.inference_mode()
def predict(
    model: FalconModel,
    *,
    image: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one live detector→Falcon→LLM pass without any cache dependency."""

    result, encoded = _generate(
        model,
        image=image,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    regions, masks = _resolved_region_payload(
        model,
        result["answer"],
        encoded.detector_output,
    )
    result.update(regions)
    return result, masks


@torch.inference_mode()
def predict_segmentation(
    model: FalconModel,
    *,
    image: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], SegmentationPrediction]:
    """Generate and strictly resolve ``<SEG>`` transport against live proposals."""

    result, encoded = _generate(
        model,
        image=image,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    prediction = resolve_segmentation(
        result["answer"],
        encoded.detector_output,
        max_regions=model.config.max_regions,
    )
    result["region_ids"] = list(prediction.region_indices)
    return result, prediction


@torch.inference_mode()
def predict_panoptic(
    model: FalconModel,
    *,
    image: Any,
    prompt: str,
    category_ids: Collection[int],
    query_category_id: int | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], PanopticPrediction]:
    """Generate panoptic instances, optionally selecting a queried predicted class."""

    if query_category_id is not None and (
        isinstance(query_category_id, bool)
        or not isinstance(query_category_id, int)
        or query_category_id not in category_ids
    ):
        raise ValueError("query_category_id must belong to the manifest category IDs")

    result, encoded = _generate(
        model,
        image=image,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    prediction = resolve_panoptic(
        result["answer"],
        encoded.detector_output,
        category_ids=category_ids,
        max_regions=model.config.max_regions,
    )
    if query_category_id is not None:
        prediction = select_panoptic_category(prediction, query_category_id)
    result["region_ids"] = [instance.region_index for instance in prediction.instances]
    result["segments_info"] = [dict(segment) for segment in prediction.segments_info]
    return result, prediction
