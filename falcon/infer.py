from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from collections.abc import Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import quote

import numpy as np
import torch
from PIL import Image


from .capabilities import (
    SSA_ABLATIONS,
    SafetyCapabilities,
    apply_ablation,
    capabilities_from_coverage,
)
from .config import load_config, resolve_stage_config, validate_config
from .detector import ProposalPolicy, RFDETRDetector
from .grounding import (
    PanopticPrediction,
    panoptic_id_map_to_rgb,
)
from .model import FalconModel
from .prediction import predict, predict_panoptic, predict_segmentation
from .tasks import TASK_REGISTRY

CHECKPOINT_FORMAT = "falcon-checkpoint"
LEGACY_CHECKPOINT_FORMAT = "falcon-checkpoint-v1"
FINAL_STAGE3_READINESS_FORMAT = "falcon-final-stage3-readiness-v2"


@dataclass(frozen=True)
class CheckpointSpec:
    """Validated checkpoint material read before any heavyweight allocation."""

    stage: int
    state_dict: dict[str, Any]
    metadata: dict[str, Any]
    config: dict[str, Any] | None
    safety_capabilities: SafetyCapabilities


def _read_checkpoint(path: str | Path) -> CheckpointSpec:
    """Read a CPU checkpoint; frozen backbones are loaded separately."""
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("checkpoint must be a regular file")
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    format_name = payload.get("format")
    if format_name not in (None, CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT):
        raise ValueError(f"unsupported checkpoint format {format_name!r}")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain a non-empty state_dict")
    stage = payload.get("stage")
    if isinstance(stage, bool) or stage not in (2, 3):
        raise ValueError("checkpoint stage must be 2 or 3")
    raw_metadata = payload.get("metadata")
    if not raw_metadata:
        if format_name == CHECKPOINT_FORMAT:
            raise ValueError("checkpoint metadata is required")
        warnings.warn(
            "Legacy checkpoint has no metadata; safety outputs are disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return CheckpointSpec(stage, state, {}, None, SafetyCapabilities.unavailable())
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")
    metadata = deepcopy(dict(raw_metadata))
    metadata.setdefault("checkpoint_format", format_name or LEGACY_CHECKPOINT_FORMAT)
    if metadata["checkpoint_format"] not in (
        CHECKPOINT_FORMAT,
        LEGACY_CHECKPOINT_FORMAT,
    ):
        raise ValueError("unsupported checkpoint metadata format")
    if metadata.get("stage", stage) != stage:
        raise ValueError("checkpoint stage disagrees with metadata")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint metadata.config must be a mapping")
    validate_config(config)
    capabilities = SafetyCapabilities.from_dict(metadata.get("safety_capabilities"))
    supervision = metadata.get("supervision_capabilities")
    ablation = metadata.get(
        "ssa_ablation", config.get("experiment", {}).get("ssa_ablation", "none")
    )
    if supervision is not None:
        if (
            apply_ablation(SafetyCapabilities.from_dict(supervision), ablation)
            != capabilities
        ):
            raise ValueError(
                "checkpoint capabilities disagree with supervision and ablation"
            )
    if ablation != config.get("experiment", {}).get("ssa_ablation", "none"):
        raise ValueError("checkpoint ablation disagrees with its configuration")
    require_observed = metadata.get("require_observed_supervision", False)
    if not isinstance(require_observed, bool):
        raise ValueError("require_observed_supervision must be boolean")
    observed = metadata.get("observed_supervision_coverage")
    if require_observed and observed is None:
        raise ValueError("checkpoint lacks required observed supervision coverage")
    if observed is not None:
        if not isinstance(observed, Mapping) or set(observed) != {
            "images",
            "risk",
            "presence",
            "links",
        }:
            raise ValueError(
                "observed supervision requires images, risk, presence and links"
            )
        if capabilities.restrict(capabilities_from_coverage(observed)) != capabilities:
            raise ValueError(
                "checkpoint enables a head without observed training supervision"
            )
    complete = metadata.get("training_complete")
    if complete is not None and not isinstance(complete, bool):
        raise ValueError("training_complete must be boolean")
    if format_name == CHECKPOINT_FORMAT and complete is None:
        raise ValueError("checkpoint metadata.training_complete is required")
    return CheckpointSpec(stage, state, metadata, config, capabilities)


def _critical_config_values(config: Mapping[str, Any], stage: int) -> dict[str, Any]:
    """Select values that affect model shape, proposal identity, or output semantics."""

    model = config["model"]
    detector = config["detector"]
    training = resolve_stage_config(dict(config), stage)
    return {
        "model": {
            key: deepcopy(model[key])
            for key in (
                "image_size",
                "patch_size",
                "region_dim",
                "roi_size",
                "max_regions",
                "component_order",
                "link_order",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
            )
        },
        "detector": {
            key: deepcopy(detector.get(key))
            for key in (
                "variant",
                "resolution",
                "score_threshold",
                "nms_threshold",
                "mask_threshold",
                "max_regions",
            )
        },
        "training": {
            key: deepcopy(training[key])
            for key in (
                "max_text_tokens",
                "text_overflow_policy",
                "risk_loss_weight",
                "presence_loss_weight",
                "link_loss_weight",
            )
        },
    }


def _effective_config(
    checkpoint: CheckpointSpec,
    runtime_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if runtime_config is not None:
        validate_config(runtime_config)
    if checkpoint.config is None:
        return load_config(None) if runtime_config is None else runtime_config
    if runtime_config is None:
        return deepcopy(checkpoint.config)
    expected = _critical_config_values(checkpoint.config, checkpoint.stage)
    received = _critical_config_values(runtime_config, checkpoint.stage)
    if received != expected:
        differences = []
        for section in expected:
            for key, value in expected[section].items():
                if received[section][key] != value:
                    differences.append(
                        f"{section}.{key}: checkpoint={value!r}, runtime="
                        f"{received[section][key]!r}"
                    )
        raise ValueError(
            "runtime configuration is incompatible with checkpoint metadata: "
            + "; ".join(differences)
        )
    return deepcopy(runtime_config)


def _resolve_backbone_locations(
    checkpoint: CheckpointSpec,
    args: argparse.Namespace,
    runtime_config: Mapping[str, Any] | None,
    effective_config: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve load locations separately from portable checkpoint identity."""

    locations = checkpoint.metadata.get("model_locations", {})
    if locations is None:
        locations = {}
    if not isinstance(locations, Mapping):
        raise ValueError("checkpoint metadata.model_locations must be a mapping")
    result = {}
    for config_name, metadata_name, argument_name in (
        ("vision_model", "vision_backbone", "vision_model"),
        ("language_model", "language_backbone", "language_model"),
    ):
        explicit = getattr(args, argument_name, None)
        runtime = (
            runtime_config.get("model", {}).get(config_name)
            if runtime_config is not None
            else None
        )
        hinted = locations.get(config_name)
        legacy = effective_config["model"].get(config_name)
        reference = next(
            (
                value
                for value in (explicit, runtime, hinted, legacy)
                if value is not None
            ),
            None,
        )
        if not isinstance(reference, str) or not reference.strip():
            option = argument_name.replace("_", "-")
            raise ValueError(
                f"No load location is available for {metadata_name}; provide --{option}"
            )
        if reference.startswith("sha256:"):
            raise ValueError(
                f"Checkpoint stores only the portable identity for {metadata_name}; provide "
                f"--{argument_name.replace('_', '-')} with the local model snapshot path"
            )
        result[metadata_name] = reference
    return result


def _require_final_stage3(checkpoint: CheckpointSpec) -> None:
    metadata = checkpoint.metadata
    if checkpoint.stage != 3 or metadata.get("training_complete") is not True:
        raise ValueError("Evaluation requires a completed Stage 3 checkpoint")
    if metadata.get("official_result_eligible") is False or metadata.get(
        "diagnostic_training_reasons"
    ):
        raise ValueError(
            "Diagnostic training checkpoints cannot be used for final evaluation"
        )
    config_ablation = (
        (checkpoint.config or {}).get("experiment", {}).get("ssa_ablation", "none")
    )
    if (
        metadata.get("ssa_ablation", config_ablation) != "none"
        or config_ablation != "none"
    ):
        raise ValueError("Final Stage 3 evaluation does not allow ablated checkpoints")
    if metadata.get("partial_initialization_allowed") or metadata.get(
        "legacy_initialization_allowed"
    ):
        raise ValueError(
            "Final Stage 3 evaluation does not allow partial/legacy initialization"
        )
    if metadata.get("oracle_detector", "none") != "none":
        raise ValueError("Final Stage 3 evaluation does not allow oracle checkpoints")


def verify_final_stage3_evaluation(
    *,
    model: str | Path | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    checkpoint: str | Path | None = None,
    detector_weights: str | Path | None = None,
    config: str | Path | Mapping[str, Any] | None = None,
    vision_model: str | None = None,
    language_model: str | None = None,
) -> dict[str, Any]:
    """Check final checkpoint metadata and local model paths without model inference."""
    if model is not None:
        if any(
            value is not None
            for value in (
                checkpoint,
                detector_weights,
                config,
                vision_model,
                language_model,
            )
        ):
            raise ValueError(
                "--model is self-contained; do not pass separate checkpoint, detector, backbone or config options"
            )
        return _verify_hf_model(
            model, revision=revision, local_files_only=local_files_only
        )
    if revision is not None or local_files_only:
        raise ValueError("--revision and --local-files-only require --model")
    if checkpoint is None or detector_weights is None:
        raise ValueError(
            "Use --model, or provide both --checkpoint and --detector-weights"
        )
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    spec = _read_checkpoint(checkpoint_path)
    _require_final_stage3(spec)
    runtime_config = (
        None
        if config is None
        else deepcopy(dict(config))
        if isinstance(config, Mapping)
        else load_config(config)
    )
    effective_config = _effective_config(spec, runtime_config)
    if effective_config.get("experiment", {}).get("ssa_ablation", "none") != "none":
        raise ValueError("Final Stage 3 evaluation cannot override SSA ablation")
    args = argparse.Namespace(vision_model=vision_model, language_model=language_model)
    locations = _resolve_backbone_locations(
        spec, args, runtime_config, effective_config
    )
    for name, location in locations.items():
        local = Path(location).expanduser()
        if not local.is_dir():
            raise ValueError(
                f"Final evaluation requires a local {name} directory: {local}"
            )
        locations[name] = str(local.resolve(strict=True))
    effective_config["model"]["vision_model"] = locations["vision_backbone"]
    effective_config["model"]["language_model"] = locations["language_backbone"]
    detector_path = Path(detector_weights).expanduser().resolve(strict=True)
    if not detector_path.is_file():
        raise ValueError("Final evaluation requires a local detector weights file")
    return {
        "format": FINAL_STAGE3_READINESS_FORMAT,
        "ready": True,
        "stage": 3,
        "checkpoint": str(checkpoint_path),
        "metadata": spec.metadata,
        "config": deepcopy(effective_config),
        "model_locations": {
            "vision_model": locations["vision_backbone"],
            "language_model": locations["language_backbone"],
        },
        "detector_weights": str(detector_path),
        "scope": "checkpoint metadata and local model paths",
    }


def _verify_hf_model(
    model: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    from .hf_support import resolve_model_package, validate_model_package

    package = resolve_model_package(
        model, revision=revision, local_files_only=local_files_only
    )
    verified = validate_model_package(package)
    metadata = verified["metadata"]
    runtime_config = verified["runtime_config"]
    _require_final_stage3(
        CheckpointSpec(
            stage=metadata.get("stage"),
            state_dict={},
            metadata=metadata,
            config=runtime_config,
            safety_capabilities=SafetyCapabilities.from_dict(
                metadata.get("safety_capabilities")
            ),
        )
    )
    reference = str(model)
    if Path(model).expanduser().is_dir():
        reference = str(Path(model).expanduser().resolve())
    resolved_revision = package.name if package.parent.name == "snapshots" else revision
    return {
        "format": FINAL_STAGE3_READINESS_FORMAT,
        "ready": True,
        "stage": 3,
        "model": {
            "reference": reference,
            "revision": resolved_revision,
            "requested_revision": revision,
            "path": str(package),
        },
        "metadata": deepcopy(metadata),
        "config": deepcopy(runtime_config),
        "scope": "self-contained Hugging Face model metadata and weight inventory",
    }


def _load_hf_model(args: argparse.Namespace, config: dict[str, Any] | None) -> Any:
    if any(
        value is not None
        for value in (
            getattr(args, "checkpoint", None),
            getattr(args, "detector_weights", None),
            getattr(args, "vision_model", None),
            getattr(args, "language_model", None),
            config,
        )
    ):
        raise ValueError(
            "--model cannot be combined with separate checkpoint, detector, backbone or config options"
        )
    if (
        getattr(args, "allow_partial_checkpoint", False)
        or getattr(args, "ssa_ablation", "none") != "none"
        or getattr(args, "oracle_detector", "none") != "none"
    ):
        raise ValueError(
            "Hugging Face inference uses the complete, unablated exported model"
        )
    readiness = getattr(args, "final_stage3_readiness", None)
    if readiness is None:
        readiness = _verify_hf_model(
            args.model,
            revision=getattr(args, "revision", None),
            local_files_only=getattr(args, "local_files_only", False),
        )
    if (
        not isinstance(readiness, Mapping)
        or readiness.get("format") != FINAL_STAGE3_READINESS_FORMAT
        or readiness.get("ready") is not True
        or not isinstance(readiness.get("model"), Mapping)
    ):
        raise ValueError("Invalid Hugging Face model readiness report")
    reference = str(args.model)
    if Path(args.model).expanduser().is_dir():
        reference = str(Path(args.model).expanduser().resolve())
    if reference != readiness["model"].get("reference") or getattr(
        args, "revision", None
    ) != readiness["model"].get("requested_revision"):
        raise ValueError("Requested model differs from Hugging Face readiness")
    # Load the exact snapshot inspected by preflight, not a moving Hub branch.
    current = _verify_hf_model(readiness["model"]["path"], local_files_only=True)
    if any(current[key] != readiness[key] for key in ("stage", "metadata", "config")):
        raise ValueError("Hugging Face model changed after readiness verification")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        readiness["model"]["path"],
        trust_remote_code=True,
        local_files_only=True,
        device=args.device,
    ).eval()
    if (
        getattr(model, "checkpoint_metadata", None) != readiness["metadata"]
        or getattr(model, "runtime_config", None) != readiness["config"]
    ):
        raise ValueError(
            "Loaded model metadata differs from the verified FALCON export"
        )
    model.hf_model_identity = deepcopy(readiness["model"])
    return model


def _validate_oracle_mode(mode: str) -> None:
    if mode != "none":
        raise ValueError("Falcon inference requires live detector proposals")


def _apply_inference_ablation(
    model: FalconModel,
    name: str,
    *,
    trained_capabilities: SafetyCapabilities | None = None,
) -> SafetyCapabilities:
    """Restrict a loaded model before its first token injection.

    Checkpoint completeness is validated first against the trained capability
    inventory.  The restriction then freezes additional heads and removes their
    tokens, without ever enabling a head absent from the checkpoint metadata.
    """

    if name not in SSA_ABLATIONS:
        raise ValueError(f"unknown SSA ablation {name!r}")
    trained = trained_capabilities
    if trained is None:
        trained = model.config.safety_capabilities
    if not isinstance(trained, SafetyCapabilities):
        raise TypeError("trained safety capabilities must be SafetyCapabilities")
    effective = apply_ablation(trained, name)
    if effective != trained:
        if not hasattr(model, "config") or not hasattr(model, "safety_adapter"):
            raise TypeError("SSA ablation requires a FalconModel safety adapter")
        model.config = replace(model.config, safety_capabilities=effective)
        model.safety_adapter.capabilities = effective
        model.safety_adapter.freeze_unavailable_heads()
    model.inference_ssa_ablation = name
    model.inference_safety_capabilities = effective
    return effective


def _load_model(args: argparse.Namespace, config: dict[str, Any] | None) -> FalconModel:
    if getattr(args, "model", None) is not None:
        return _load_hf_model(args, config)
    if getattr(args, "revision", None) is not None or getattr(
        args, "local_files_only", False
    ):
        raise ValueError("--revision and --local-files-only require --model")
    if not getattr(args, "checkpoint", None) or not getattr(
        args, "detector_weights", None
    ):
        raise ValueError(
            "Use --model, or provide both --checkpoint and --detector-weights"
        )
    # Checkpoint structure, metadata, capabilities, and configuration are all
    # validated before RF-DETR or either pretrained backbone is constructed.
    readiness = getattr(args, "final_stage3_readiness", None)
    if readiness is not None:
        if (
            not isinstance(readiness, Mapping)
            or readiness.get("format") != FINAL_STAGE3_READINESS_FORMAT
            or readiness.get("ready") is not True
            or readiness.get("stage") != 3
        ):
            raise ValueError("Invalid final Stage 3 evaluation readiness report")
        checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
        if str(checkpoint_path) != readiness.get("checkpoint"):
            raise ValueError("Checkpoint location differs from final Stage 3 readiness")
        if (
            getattr(args, "allow_partial_checkpoint", False)
            or getattr(args, "ssa_ablation", "none") != "none"
            or getattr(args, "oracle_detector", "none") != "none"
        ):
            raise ValueError(
                "Final Stage 3 evaluation cannot enable partial checkpoints, ablations, or oracles"
            )
    checkpoint = _read_checkpoint(args.checkpoint)
    if readiness is not None:
        _require_final_stage3(checkpoint)
        if checkpoint.metadata != readiness.get("metadata"):
            raise ValueError("Checkpoint metadata changed after Stage 3 readiness")

    allow_partial = getattr(args, "allow_partial_checkpoint", False)
    if not isinstance(allow_partial, bool):
        raise ValueError("allow_partial_checkpoint must be boolean")
    if checkpoint.metadata.get("training_complete") is False and not allow_partial:
        raise ValueError(
            "checkpoint training is incomplete; pass --allow-partial-checkpoint only for "
            "explicitly diagnostic inference"
        )
    runtime_config = config
    config = _effective_config(checkpoint, runtime_config)
    backbone_locations = _resolve_backbone_locations(
        checkpoint,
        args,
        runtime_config,
        config,
    )
    if readiness is not None:
        actual_locations = {
            name: str(Path(location).expanduser().resolve(strict=True))
            for name, location in backbone_locations.items()
        }
        expected_locations = readiness.get("model_locations", {})
        if actual_locations != {
            "vision_backbone": expected_locations.get("vision_model"),
            "language_backbone": expected_locations.get("language_model"),
        } or str(
            Path(args.detector_weights).expanduser().resolve(strict=True)
        ) != readiness.get("detector_weights"):
            raise ValueError(
                "Frozen artifact locations differ from final Stage 3 readiness"
            )
        # Absolute, existing snapshots cannot silently become Hub model IDs.
        backbone_locations = actual_locations
    ablation = getattr(args, "ssa_ablation", "none")
    if ablation not in SSA_ABLATIONS:
        raise ValueError(f"unknown SSA ablation {ablation!r}")
    _validate_oracle_mode(getattr(args, "oracle_detector", "none"))
    config["model"]["vision_model"] = backbone_locations["vision_backbone"]
    config["model"]["language_model"] = backbone_locations["language_backbone"]

    model_cfg, detector_cfg, training_cfg = (
        config["model"],
        config["detector"],
        resolve_stage_config(config, checkpoint.stage),
    )
    policy = ProposalPolicy(
        score_threshold=detector_cfg["score_threshold"],
        nms_threshold=detector_cfg["nms_threshold"],
        max_regions=detector_cfg["max_regions"],
        mask_threshold=detector_cfg.get("mask_threshold", 0.5),
    )
    detector = RFDETRDetector(
        checkpoint=args.detector_weights,
        variant=detector_cfg["variant"],
        device=args.device,
        resolution=detector_cfg.get("resolution"),
        proposal_policy=policy,
    )
    if getattr(args, "language_dtype", None) is not None:
        dtype = args.language_dtype
        if not isinstance(dtype, torch.dtype):
            raise TypeError("language_dtype must be a torch.dtype")
    elif args.device.startswith("cuda"):
        dtype = (
            torch.bfloat16 if training_cfg.get("precision") == "bf16" else torch.float16
        )
    else:
        dtype = torch.float32
    model = FalconModel.from_pretrained(
        vision_model_name_or_path=backbone_locations["vision_backbone"],
        language_model_name_or_path=backbone_locations["language_backbone"],
        detector=detector,
        region_dim=model_cfg["region_dim"],
        image_size=model_cfg["image_size"],
        patch_size=model_cfg["patch_size"],
        roi_size=model_cfg["roi_size"],
        max_regions=model_cfg["max_regions"],
        max_text_tokens=training_cfg["max_text_tokens"],
        text_overflow_policy=training_cfg.get("text_overflow_policy", "error"),
        risk_loss_weight=training_cfg["risk_loss_weight"],
        presence_loss_weight=training_cfg["presence_loss_weight"],
        link_loss_weight=training_cfg["link_loss_weight"],
        safety_capabilities=checkpoint.safety_capabilities,
        vision_kwargs={"local_files_only": True} if readiness is not None else None,
        language_kwargs={
            "torch_dtype": dtype,
            "attn_implementation": "sdpa",
            **({"local_files_only": True} if readiness is not None else {}),
        },
    )
    if checkpoint.stage == 3:
        model.enable_lora(
            rank=model_cfg["lora_rank"],
            alpha=model_cfg["lora_alpha"],
            dropout=model_cfg["lora_dropout"],
        )
    model.set_training_stage(checkpoint.stage)
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing = expected.difference(checkpoint.state_dict)
    if missing:
        raise ValueError(f"checkpoint is incomplete; missing {sorted(missing)[:5]}")
    incompatible = model.load_state_dict(checkpoint.state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"checkpoint contains unexpected keys: {incompatible.unexpected_keys[:5]}"
        )
    effective_capabilities = _apply_inference_ablation(
        model,
        ablation,
        trained_capabilities=checkpoint.safety_capabilities,
    )
    model = model.to(args.device).eval()

    model.checkpoint_metadata = deepcopy(checkpoint.metadata)
    model.runtime_config = deepcopy(config)
    model.backbone_locations = deepcopy(backbone_locations)
    model.trained_safety_capabilities = checkpoint.safety_capabilities
    model.inference_safety_capabilities = effective_capabilities
    return model


def _coco_rle(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_util

    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise ValueError("prediction masks must be two-dimensional boolean arrays")
    encoded = mask_util.encode(np.asfortranarray(array.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if not isinstance(counts, str):
        raise RuntimeError("pycocotools returned non-text RLE counts")
    return {"size": [int(value) for value in encoded["size"]], "counts": counts}


def _write_panoptic_artifact(
    prediction: PanopticPrediction,
    artifact_root: Path,
    identity: str,
) -> str:
    """Write an atomic PNG under a reversible, filesystem-safe task name."""
    encoded = quote(identity, safe="-_.")
    chunks = [
        encoded[index : index + 160] for index in range(0, len(encoded), 160)
    ] or ["unnamed"]
    relative = Path(
        "panoptic",
        *("part-" + chunk for chunk in chunks[:-1]),
        "task-" + chunks[-1] + ".png",
    )
    destination = artifact_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".panoptic-", suffix=".png", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            Image.fromarray(panoptic_id_map_to_rgb(prediction.id_map), mode="RGB").save(
                stream, format="PNG"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return relative.as_posix()


class FalconTaskPredictor:
    """Model-backed callable for the evaluator's ground-truth-free task view."""

    _VIEW_FIELDS = frozenset(
        {
            "split",
            "task_id",
            "family",
            "image_id",
            "image_path",
            "prompt",
            "question_type",
            "query_category_id",
        }
    )

    def __init__(
        self,
        model: FalconModel,
        artifact_root: str | Path,
        *,
        category_ids: Collection[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        policy = getattr(getattr(model, "detector", None), "proposal_policy", None)
        if not isinstance(policy, ProposalPolicy):
            # Hub custom code has its own module namespace; validate its policy
            # values instead of relying on Python class identity across modules.
            try:
                local_policy = ProposalPolicy(
                    **{
                        name: getattr(policy, name)
                        for name in (
                            "score_threshold",
                            "nms_threshold",
                            "max_regions",
                            "mask_threshold",
                            "reject_empty_masks",
                            "version",
                        )
                    }
                )
                if policy.to_dict() != local_policy.to_dict():
                    raise ValueError("detector proposal policy semantics differ")
                policy = local_policy
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "FalconTaskPredictor requires a detector with ProposalPolicy"
                ) from exc
        categories = tuple(category_ids)
        if (
            not categories
            or len(categories) != len(set(categories))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in categories
            )
        ):
            raise ValueError(
                "category_ids must be unique positive integers from the manifest"
            )
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        self.model = model
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.category_ids = categories
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.proposal_policy = policy

    @classmethod
    def _validate_view(cls, task: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(task, Mapping):
            raise ValueError("inference task view must be a mapping")
        unknown = sorted(set(task).difference(cls._VIEW_FIELDS))
        if unknown:
            raise ValueError(
                "inference task view contains fields outside the no-ground-truth contract: "
                f"{unknown}"
            )
        required = ("split", "task_id", "family", "image_id", "image_path", "prompt")
        missing = [name for name in required if name not in task]
        if missing:
            raise ValueError(f"inference task view is missing fields: {missing}")
        for name in ("split", "task_id", "image_path", "prompt"):
            if not isinstance(task[name], str) or not task[name]:
                raise ValueError(f"inference task {name} must be a non-empty string")
        if task["family"] not in TASK_REGISTRY:
            raise ValueError(f"unknown task family {task['family']!r}")
        return dict(task)

    def __call__(self, task: Mapping[str, Any]) -> dict[str, Any]:
        view = self._validate_view(task)
        kwargs = {
            "image": view["image_path"],
            "prompt": view["prompt"],
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        prediction_kind = TASK_REGISTRY[view["family"]].prediction_kind
        if prediction_kind in ("text", "answer"):
            result, _masks = _predict_model(self.model, "text", **kwargs)
            return {"answer": result["answer"]}
        if prediction_kind == "binary_mask":
            _result, prediction = _predict_model(self.model, "segmentation", **kwargs)
            return {"regions": [_coco_rle(mask) for mask in prediction.masks]}
        if prediction_kind == "panoptic":
            query_category_id = None
            if view["family"] == "referring_panoptic_segmentation":
                query_category_id = view.get("query_category_id")
                if query_category_id is None:
                    raise ValueError(
                        "referring panoptic inference requires query_category_id"
                    )
            _result, prediction = _predict_model(
                self.model,
                "panoptic",
                category_ids=self.category_ids,
                query_category_id=query_category_id,
                **kwargs,
            )
            file_name = _write_panoptic_artifact(
                prediction,
                self.artifact_root,
                f"{view['split']}\0{view['task_id']}",
            )
            return {
                "panoptic": {
                    "file_name": file_name,
                    "segments_info": [dict(item) for item in prediction.segments_info],
                },
            }
        raise AssertionError(f"unhandled prediction kind {prediction_kind!r}")


def _predict_model(model: Any, task: str, **kwargs: Any) -> Any:
    if getattr(getattr(model, "config", None), "model_type", None) == "falcon_x":
        return model.predict(task=task, **kwargs)
    method = {
        "text": predict,
        "segmentation": predict_segmentation,
        "panoptic": predict_panoptic,
    }[task]
    return method(model, **kwargs)


def _checkpoint_category_ids(metadata: Mapping[str, Any]) -> tuple[int, ...] | None:
    raw = metadata.get("dataset_categories")
    if raw is None:
        return None
    if isinstance(raw, str | bytes) or not isinstance(raw, Collection):
        raise ValueError("checkpoint dataset_categories must be an array")
    identifiers = []
    for index, category in enumerate(raw):
        if not isinstance(category, Mapping):
            raise ValueError(
                f"checkpoint dataset_categories[{index}] must be a mapping"
            )
        identifier = category.get("id")
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 1
        ):
            raise ValueError(
                f"checkpoint dataset_categories[{index}].id must be a positive integer"
            )
        identifiers.append(identifier)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("checkpoint dataset category IDs must be nonempty and unique")
    return tuple(identifiers)


def load_task_predictor(
    *,
    model: str | Path | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    checkpoint: str | Path | None = None,
    detector_weights: str | Path | None = None,
    artifact_root: str | Path,
    category_ids: Collection[int],
    config: str | Path | Mapping[str, Any] | None = None,
    vision_model: str | None = None,
    language_model: str | None = None,
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    ssa_ablation: str = "none",
    oracle_detector: str = "none",
    allow_partial_checkpoint: bool = False,
    final_stage3_readiness: Mapping[str, Any] | None = None,
) -> tuple[FalconTaskPredictor, Mapping[str, Any]]:
    """Build a final Stage 3 evaluator and return its run settings."""

    _validate_oracle_mode(oracle_detector)
    if not isinstance(allow_partial_checkpoint, bool):
        raise ValueError("allow_partial_checkpoint must be boolean")
    if allow_partial_checkpoint or ssa_ablation != "none":
        raise ValueError("Evaluation does not allow partial checkpoints or ablations")
    if final_stage3_readiness is None:
        final_stage3_readiness = verify_final_stage3_evaluation(
            model=model,
            revision=revision,
            local_files_only=local_files_only,
            checkpoint=checkpoint,
            detector_weights=detector_weights,
            config=config,
            vision_model=vision_model,
            language_model=language_model,
        )
    if config is None:
        runtime_config = None
    elif isinstance(config, Mapping):
        runtime_config = deepcopy(dict(config))
        validate_config(runtime_config)
    else:
        runtime_config = load_config(config)
    args = argparse.Namespace(
        model=None if model is None else str(model),
        revision=revision,
        local_files_only=local_files_only,
        checkpoint=None if checkpoint is None else str(checkpoint),
        detector_weights=None if detector_weights is None else str(detector_weights),
        vision_model=vision_model,
        language_model=language_model,
        device=device,
        ssa_ablation=ssa_ablation,
        oracle_detector=oracle_detector,
        allow_partial_checkpoint=allow_partial_checkpoint,
        final_stage3_readiness=final_stage3_readiness,
    )
    model = _load_model(args, runtime_config)

    metadata = dict(model.checkpoint_metadata)
    trained_categories = _checkpoint_category_ids(metadata)
    supplied_categories = tuple(category_ids)
    if trained_categories is not None and set(supplied_categories) != set(
        trained_categories
    ):
        raise ValueError(
            "predictor category_ids differ from checkpoint dataset taxonomy"
        )
    predictor = FalconTaskPredictor(
        model,
        artifact_root,
        category_ids=supplied_categories,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    metadata = model.checkpoint_metadata
    provenance = {
        "stage": 3 if final_stage3_readiness is not None else metadata.get("stage"),
        "training_complete": metadata.get("training_complete", False),
        "checkpoint_scope": {
            "training_complete": metadata.get("training_complete", False),
            "partial_checkpoint_allowed": allow_partial_checkpoint,
            "official_result_eligible": metadata.get("training_complete") is True
            and metadata.get("official_result_eligible", True) is True
            and not metadata.get("diagnostic_training_reasons"),
            "diagnostic_training_reasons": metadata.get(
                "diagnostic_training_reasons", []
            ),
        },
        "config": deepcopy(model.runtime_config),
        "safety_capabilities": model.inference_safety_capabilities.as_dict(),
        "ssa_ablation": ssa_ablation,
        "oracle_detector": oracle_detector,
        "final_stage3": final_stage3_readiness is not None,
    }
    if getattr(model, "hf_model_identity", None) is not None:
        provenance["model"] = deepcopy(model.hf_model_identity)
    else:
        provenance.update(
            {
                "checkpoint": str(Path(checkpoint).expanduser().resolve()),
                "detector_weights": str(Path(detector_weights).expanduser().resolve()),
                "model_locations": {
                    "vision_model": model.backbone_locations["vision_backbone"],
                    "language_model": model.backbone_locations["language_backbone"],
                },
            }
        )

    return predictor, MappingProxyType(provenance)


@torch.inference_mode()
def infer(args: argparse.Namespace) -> dict:
    prediction_kind = getattr(args, "output_kind", "text")
    query_category_id = getattr(args, "query_category_id", None)
    if query_category_id is not None and prediction_kind != "panoptic":
        raise ValueError("--query-category-id requires --output-kind panoptic")
    config_path = getattr(args, "config", None)
    model = _load_model(args, None if config_path is None else load_config(config_path))
    with Image.open(args.image) as opened:
        image = opened.convert("RGB")
    common = {
        "image": image,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    panoptic_prediction = None
    if prediction_kind == "seg":
        result, segmentation = _predict_model(model, "segmentation", **common)
        masks = segmentation.masks
    elif prediction_kind == "panoptic":
        category_ids = getattr(args, "category_id", None)
        if not category_ids:
            raise ValueError("--category-id is required for panoptic prediction")
        result, panoptic_prediction = _predict_model(
            model,
            "panoptic",
            category_ids=category_ids,
            query_category_id=query_category_id,
            **common,
        )
        masks = (
            np.stack([item.mask for item in panoptic_prediction.instances])
            if (panoptic_prediction.instances)
            else np.empty((0, *panoptic_prediction.id_map.shape), dtype=bool)
        )
    else:
        result, masks = _predict_model(model, "text", **common)
    result["image_id"] = args.image_id or Path(args.image).stem
    result["image"] = str(Path(args.image).expanduser().resolve())
    if args.output_dir:
        output = Path(args.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        if panoptic_prediction is not None:
            file_name = _write_panoptic_artifact(
                panoptic_prediction,
                output,
                f"infer\0{result['image_id']}",
            )
            result["panoptic"] = {
                "file_name": file_name,
                "segments_info": [
                    dict(item) for item in panoptic_prediction.segments_info
                ],
            }
        (output / "prediction.jsonl").write_text(
            json.dumps(result, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(output / "masks.npz", masks=masks)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live end-to-end Falcon inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id", help="Evaluation ID; defaults to the image stem")
    parser.add_argument("--prompt", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        help="Self-contained FALCON model directory or Hugging Face repository ID (loads its custom code)",
    )
    source.add_argument(
        "--checkpoint", help="Legacy training checkpoint; requires --detector-weights"
    )
    parser.add_argument("--revision", help="Hugging Face model commit, tag or branch")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local or cached Hugging Face files",
    )
    parser.add_argument(
        "--detector-weights", help="Detector checkpoint for legacy --checkpoint loading"
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--config", help="Optional YAML override; package defaults are portable"
    )
    parser.add_argument("--vision-model")
    parser.add_argument("--language-model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-kind", choices=("text", "seg", "panoptic"), default="text"
    )
    parser.add_argument(
        "--category-id",
        action="append",
        type=int,
        help="Manifest category ID; repeat for panoptic output",
    )
    parser.add_argument(
        "--query-category-id",
        type=int,
        help="Select a predicted category for referring panoptic output",
    )
    parser.add_argument(
        "--ssa-ablation",
        choices=SSA_ABLATIONS,
        default="none",
        help="Inference-only diagnostic head/token restriction",
    )
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Diagnostic only: allow a checkpoint explicitly marked training_complete=false",
    )
    return parser


def main() -> None:
    print(json.dumps(infer(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
