from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .capabilities import SSA_ABLATIONS

COMPONENT_ORDER = ("detonator", "explosive", "battery")
LINK_ORDER = (
    ("battery", "detonator"),
    ("battery", "explosive"),
    ("detonator", "explosive"),
)
DETECTOR_VARIANTS = {
    "seg-nano",
    "seg-small",
    "seg-medium",
    "seg-large",
    "seg-xlarge",
    "seg-2xlarge",
}


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "falcon-x",
        "protocol": "dataset-v1",
        "backbone_layout": "independent",
        "world_size": None,
        "ssa_ablation": "none",
    },
    "model": {
        "vision_model": "facebook/dinov2-large",
        "language_model": "lmsys/vicuna-7b-v1.5",
        "image_size": 448,
        "patch_size": 14,
        "region_dim": 1024,
        "roi_size": 4,
        "max_regions": 100,
        "component_order": list(COMPONENT_ORDER),
        "link_order": [list(pair) for pair in LINK_ORDER],
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    },
    "detector": {
        "variant": "seg-2xlarge",
        "resolution": None,
        "score_threshold": 0.15,
        "nms_threshold": 0.6,
        "mask_threshold": 0.5,
        "max_regions": 100,
    },
    "training": {
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "max_text_tokens": 256,
        "text_overflow_policy": "error",
        "risk_loss_weight": 1.0,
        "presence_loss_weight": 0.5,
        "link_loss_weight": 0.5,
        "seed": 42,
        "precision": "bf16",
        "tf32": True,
        "sampling": "all",
        "epoch_size": None,
    },
}

# Per-device batches and gradient accumulation are configured separately.
DEFAULT_STAGES: dict[str, dict[str, Any]] = {
    "stage1": {
        "epochs": 12,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "learning_rate": 1e-4,
        "encoder_learning_rate": 1.5e-4,
        "weight_decay": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 0.0,
        "multi_scale": False,
        "expanded_scales": False,
        "precision": "backend_mixed",
        "tf32": True,
    },
    "stage2": deepcopy(DEFAULT_CONFIG["training"]),
    "stage3": {
        **deepcopy(DEFAULT_CONFIG["training"]),
        "learning_rate": 1e-5,
        "precision": "bf16",
    },
}
DEFAULT_CONFIG["stages"] = deepcopy(DEFAULT_STAGES)


def _merge(
    base: dict[str, Any], update: dict[str, Any], prefix: str = ""
) -> dict[str, Any]:
    if not isinstance(update, dict):
        raise ValueError(f"{prefix or 'Configuration'} must be a mapping")
    for key, value in update.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown configuration key: {name}")
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value, name)
        else:
            base[key] = value
    return base


def _positive_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not minimum <= number < float("inf"):
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def validate_config(config: dict[str, Any]) -> None:
    """Reject architecture drift and common experiment-configuration mistakes."""

    for section in ("model", "detector", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
    model = config["model"]
    detector = config["detector"]
    experiment = config.get("experiment", DEFAULT_CONFIG["experiment"])
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a mapping")
    if not isinstance(experiment["name"], str) or not experiment["name"].strip():
        raise ValueError("experiment.name must be a non-empty string")
    if experiment["protocol"] not in ("dataset-v1", "paper-v2"):
        raise ValueError("experiment.protocol must be dataset-v1 or paper-v2")
    if experiment["backbone_layout"] != "independent":
        raise ValueError("Only the independent-backbone implementation is available")
    if experiment.get("ssa_ablation", "none") not in SSA_ABLATIONS:
        raise ValueError(f"experiment.ssa_ablation must be one of {SSA_ABLATIONS}")
    if experiment["world_size"] is not None:
        _positive_integer(experiment["world_size"], "experiment.world_size")

    for key in ("vision_model", "language_model"):
        if not isinstance(model[key], str) or not model[key].strip():
            raise ValueError(
                f"model.{key} must be a non-empty path or model identifier"
            )
    for key in ("image_size", "patch_size", "region_dim", "roi_size", "max_regions"):
        _positive_integer(model[key], f"model.{key}")
    if model["image_size"] % (2 * model["patch_size"]):
        raise ValueError("model.image_size must be divisible by twice model.patch_size")
    if tuple(model["component_order"]) != COMPONENT_ORDER:
        raise ValueError(f"model.component_order must be {list(COMPONENT_ORDER)!r}")
    if tuple(tuple(pair) for pair in model["link_order"]) != LINK_ORDER:
        raise ValueError(
            f"model.link_order must be {[list(pair) for pair in LINK_ORDER]!r}"
        )
    _positive_integer(model["lora_rank"], "model.lora_rank")
    _positive_integer(model["lora_alpha"], "model.lora_alpha")
    dropout = _finite_number(model["lora_dropout"], "model.lora_dropout")
    if dropout >= 1.0:
        raise ValueError("model.lora_dropout must be smaller than 1")

    if detector["variant"] not in DETECTOR_VARIANTS:
        raise ValueError(
            f"detector.variant must be one of {sorted(DETECTOR_VARIANTS)!r}"
        )
    for key in ("score_threshold", "nms_threshold", "mask_threshold"):
        if key not in detector:
            continue
        value = _finite_number(detector[key], f"detector.{key}")
        if value > 1.0:
            raise ValueError(f"detector.{key} must not exceed 1")
    _positive_integer(detector["max_regions"], "detector.max_regions")
    if detector["max_regions"] > model["max_regions"]:
        raise ValueError("detector.max_regions must not exceed model.max_regions")

    resolution = detector.get("resolution")
    if resolution is not None:
        _positive_integer(resolution, "detector.resolution")
        # Pinned segmentation variants use patch size 12. Nano uses one local
        # window; the other supported variants use two. No model load is needed.
        divisor = 12 if detector["variant"] == "seg-nano" else 24
        if resolution % divisor:
            raise ValueError(
                f"detector.resolution must be divisible by {divisor} for pinned RF-DETR"
            )

    _validate_training(config["training"], "training")
    stages = config.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("stages must be a mapping")
    for stage, values in stages.items():
        if stage not in DEFAULT_STAGES:
            raise ValueError(f"Unknown training stage: {stage}")
        if not isinstance(values, dict):
            raise ValueError(f"stages.{stage} must be a mapping")
        if stage == "stage1":
            for key in ("epochs", "batch_size", "gradient_accumulation"):
                _positive_integer(values[key], f"stages.stage1.{key}")
            for key in (
                "learning_rate",
                "encoder_learning_rate",
                "weight_decay",
                "warmup_epochs",
            ):
                _finite_number(values[key], f"stages.stage1.{key}")
            if values["learning_rate"] == 0 or values["encoder_learning_rate"] == 0:
                raise ValueError("Stage 1 learning rates must be positive")
            if values["lr_scheduler"] not in ("cosine", "step"):
                raise ValueError("stages.stage1.lr_scheduler must be cosine or step")
            for key in ("multi_scale", "expanded_scales"):
                if not isinstance(values[key], bool):
                    raise ValueError(f"stages.stage1.{key} must be boolean")
            if values.get("precision") not in ("backend_mixed", "fp32"):
                raise ValueError(
                    "stages.stage1.precision must be backend_mixed or fp32"
                )
            if not isinstance(values.get("tf32", True), bool):
                raise ValueError("stages.stage1.tf32 must be boolean")
        else:
            _validate_training(values, f"stages.{stage}")


def _validate_precision(values: dict[str, Any], prefix: str) -> None:
    if values.get("precision", "bf16") not in ("fp16", "bf16"):
        raise ValueError(f"{prefix}.precision must be fp16 or bf16")
    if not isinstance(values.get("tf32", True), bool):
        raise ValueError(f"{prefix}.tf32 must be boolean")


def _validate_training(training: dict[str, Any], prefix: str) -> None:
    for key in ("epochs", "batch_size", "gradient_accumulation", "max_text_tokens"):
        _positive_integer(training[key], f"{prefix}.{key}")
    if training["max_text_tokens"] < 3:
        raise ValueError("training.max_text_tokens must be at least 3")
    for key in (
        "learning_rate",
        "weight_decay",
        "risk_loss_weight",
        "presence_loss_weight",
        "link_loss_weight",
    ):
        _finite_number(training[key], f"{prefix}.{key}")
    if training["learning_rate"] == 0:
        raise ValueError("training.learning_rate must be greater than 0")
    warmup = _finite_number(training["warmup_ratio"], "training.warmup_ratio")
    if warmup >= 1.0:
        raise ValueError("training.warmup_ratio must be smaller than 1")
    if isinstance(training["seed"], bool) or not isinstance(training["seed"], int):
        raise ValueError("training.seed must be an integer")
    _validate_precision(training, prefix)
    if training.get("text_overflow_policy", "error") not in ("error", "truncate"):
        raise ValueError(f"{prefix}.text_overflow_policy must be error or truncate")
    if training.get("sampling", "all") not in ("all", "family_balanced"):
        raise ValueError(f"{prefix}.sampling must be all or family_balanced")
    if training.get("epoch_size") is not None:
        _positive_integer(training["epoch_size"], f"{prefix}.epoch_size")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    payload: dict[str, Any] = {}
    if path is not None:
        with Path(path).expanduser().open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("Configuration root must be a mapping")
        _merge(config, payload)
    # Explicit legacy/common training options still apply to both multimodal
    # stages. Explicit per-stage overrides win. Defaults, however, differ between
    # Stage 2 and Stage 3; a shared default LR must not erase that distinction.
    common = payload.get("training", {})
    stages = payload.get("stages", {})
    if not isinstance(common, dict) or not isinstance(stages, dict):
        raise ValueError("training and stages must be mappings")
    for key in ("stage2", "stage3"):
        merged = _merge(deepcopy(DEFAULT_STAGES[key]), common, "training")
        config["stages"][key] = _merge(merged, stages.get(key, {}), f"stages.{key}")
    validate_config(config)
    return config


def resolve_stage_config(config: dict[str, Any], stage: int) -> dict[str, Any]:
    """Return a detached, fully resolved stage configuration.

    The fallback preserves callers supplying the original configuration shape.
    Stage-specific configuration takes precedence in newly loaded files.
    """
    if stage not in (1, 2, 3):
        raise ValueError("Training stage must be 1, 2, or 3")
    key = f"stage{stage}"
    if key in config.get("stages", {}):
        return deepcopy(config["stages"][key])
    if stage == 1:
        return deepcopy(DEFAULT_STAGES[key])
    return deepcopy(config["training"])


def apply_stage_overrides(
    config: Mapping[str, Any],
    stage: int,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply stage overrides to both runtime and saved checkpoint configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if stage not in (1, 2, 3):
        raise ValueError("Training stage must be 1, 2, or 3")
    if not isinstance(overrides, Mapping):
        raise TypeError("stage overrides must be a mapping")

    effective = deepcopy(dict(config))
    validate_config(effective)
    resolved = resolve_stage_config(effective, stage)
    unknown = sorted(set(overrides).difference(resolved))
    if unknown:
        prefix = f"stages.stage{stage}"
        raise ValueError(
            "Unknown stage override"
            + ("s" if len(unknown) != 1 else "")
            + ": "
            + ", ".join(f"{prefix}.{name}" for name in unknown)
        )
    resolved.update(deepcopy(dict(overrides)))
    effective.setdefault("stages", {})[f"stage{stage}"] = resolved
    validate_config(effective)
    return effective


def paper_reproduction_issues(config: dict[str, Any]) -> list[str]:
    """Report configuration differences from the paper."""
    issues = [
        "Binary presence supervision uses BCE-with-logits instead of the paper's L1 loss."
    ]
    if (
        config.get("experiment", {}).get("backbone_layout", "independent")
        == "independent"
    ):
        issues.append(
            "The shared-feature detector described in the paper is not implemented; "
            "the independent RF-DETR/DINO topology is a documented deviation."
        )
    if config["detector"].get("resolution") != 448:
        issues.append(
            "The detector does not use the paper's stated 448-pixel resolution; "
            "448 is incompatible with the pinned segmentation backend."
        )
    return issues
