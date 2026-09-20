"""Configuration for the complete Hugging Face FALCON model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from transformers import PretrainedConfig

from .capabilities import (
    SafetyCapabilities,
    apply_ablation,
    capabilities_from_coverage,
)
from .config import resolve_stage_config, validate_config


class FalconConfig(PretrainedConfig):
    model_type = "falcon_x"
    is_composition = True

    def __init__(
        self,
        core_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        language_config: dict[str, Any] | None = None,
        detector_config: dict[str, Any] | None = None,
        image_processor_config: dict[str, Any] | None = None,
        lora_config: dict[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        language_generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        for name, value in (
            ("core_config", core_config),
            ("vision_config", vision_config),
            ("language_config", language_config),
            ("detector_config", detector_config),
            ("image_processor_config", image_processor_config),
            ("lora_config", lora_config),
            ("runtime_config", runtime_config),
            ("checkpoint_metadata", checkpoint_metadata),
            ("language_generation_config", language_generation_config),
        ):
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be a mapping")
            setattr(self, name, deepcopy(value or {}))

    def __getattr__(self, name: str) -> Any:
        # Inference consumes these scalar fields on both native and Hub models.
        core = self.__dict__.get("core_config", {})
        if name in core:
            return core[name]
        raise AttributeError(name)

    def validate(self) -> None:
        """Reject incomplete, diagnostic, or inconsistent final-model metadata."""
        for name in (
            "core_config", "vision_config", "language_config", "detector_config",
            "image_processor_config", "lora_config", "runtime_config",
            "checkpoint_metadata",
        ):
            if not getattr(self, name):
                raise ValueError(f"FALCON package requires {name}")
        metadata = self.checkpoint_metadata
        validate_config(self.runtime_config)
        if metadata.get("stage") != 3 or metadata.get("training_complete") is not True:
            raise ValueError("FALCON packages require a completed Stage 3 checkpoint")
        if metadata.get("official_result_eligible") is False or metadata.get(
            "diagnostic_training_reasons"
        ):
            raise ValueError("Diagnostic checkpoints cannot be packaged as final FALCON models")
        if metadata.get("partial_initialization_allowed") or metadata.get(
            "legacy_initialization_allowed"
        ) or metadata.get("oracle_detector", "none") != "none":
            raise ValueError("FALCON packages cannot use partial initialization or oracle proposals")
        if metadata.get("ssa_ablation", "none") != "none" or self.runtime_config.get(
            "experiment", {}
        ).get("ssa_ablation", "none") != "none":
            raise ValueError("Final FALCON packages cannot use SSA ablations")
        capabilities = SafetyCapabilities.from_dict(metadata.get("safety_capabilities"))
        core_capabilities = SafetyCapabilities.from_dict(
            self.core_config.get("safety_capabilities")
        )
        if core_capabilities != capabilities:
            raise ValueError("Core and checkpoint safety capabilities differ")
        supervision = metadata.get("supervision_capabilities")
        if supervision is not None and apply_ablation(
            SafetyCapabilities.from_dict(supervision), "none"
        ) != capabilities:
            raise ValueError("Checkpoint capabilities disagree with supervision")
        require_observed = metadata.get("require_observed_supervision", False)
        if not isinstance(require_observed, bool):
            raise ValueError("require_observed_supervision must be boolean")
        observed = metadata.get("observed_supervision_coverage")
        if require_observed and observed is None:
            raise ValueError("Checkpoint lacks required observed supervision coverage")
        if observed is not None and (
            not isinstance(observed, dict)
            or set(observed) != {"images", "risk", "presence", "links"}
        ):
            raise ValueError("Observed coverage requires images, risk, presence, and links")
        if observed is not None and capabilities.restrict(
            capabilities_from_coverage(observed)
        ) != capabilities:
            raise ValueError("Checkpoint enables a head without observed supervision")
        if metadata.get("config") != self.runtime_config:
            raise ValueError("Checkpoint and runtime configurations differ")
        runtime_model = self.runtime_config["model"]
        runtime_training = resolve_stage_config(self.runtime_config, 3)
        for key in ("image_size", "patch_size", "region_dim", "roi_size", "max_regions"):
            if self.core_config.get(key) != runtime_model[key]:
                raise ValueError(f"Core {key} differs from the training configuration")
        for key in (
            "max_text_tokens", "text_overflow_policy", "risk_loss_weight",
            "presence_loss_weight", "link_loss_weight",
        ):
            if self.core_config.get(key) != runtime_training[key]:
                raise ValueError(f"Core {key} differs from the training configuration")
        for key, runtime_key in (("rank", "lora_rank"), ("alpha", "lora_alpha"),
                                 ("dropout", "lora_dropout")):
            if self.lora_config.get(key) != runtime_model[runtime_key]:
                raise ValueError(f"LoRA {key} differs from the training configuration")
        runtime_detector = self.runtime_config["detector"]
        if self.detector_config.get("variant") != runtime_detector["variant"]:
            raise ValueError("Detector variant differs from the training configuration")
        policy = self.detector_config.get("proposal_policy", {})
        for key in ("score_threshold", "nms_threshold", "max_regions", "mask_threshold"):
            if policy.get(key) != runtime_detector.get(key, 0.5 if key == "mask_threshold" else None):
                raise ValueError(f"Detector {key} differs from the training configuration")
        resolution = runtime_detector.get("resolution")
        if resolution is not None and self.detector_config.get("architecture", {}).get(
            "resolution"
        ) != resolution:
            raise ValueError("Detector resolution differs from the training configuration")
        mask_threshold = self.core_config.get("mask_probability_threshold")
        if isinstance(mask_threshold, bool) or not isinstance(mask_threshold, int | float) or not (
            0 <= mask_threshold <= 1
        ):
            raise ValueError("Core mask_probability_threshold must lie in [0, 1]")
        for config_name, dimension in (
            ("vision_config", "vision_dim"), ("language_config", "language_dim")
        ):
            backbone = getattr(self, config_name)
            if not isinstance(backbone.get("model_type"), str):
                raise ValueError(f"{config_name} requires its architecture model_type")
            if backbone.get("hidden_size") != self.core_config.get(dimension):
                raise ValueError(f"{config_name} hidden_size differs from {dimension}")


FalconConfig.register_for_auto_class("AutoConfig")
