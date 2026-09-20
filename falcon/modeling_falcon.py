"""Self-contained Transformers wrapper around the FALCON architecture."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, PreTrainedModel
from transformers.models.auto.image_processing_auto import get_image_processor_class_from_name
from transformers.utils import ModelOutput

# Transformers 4.49 copies only direct relative imports for local model folders.
from .artifacts import atomic_json as atomic_json
from .capabilities import SafetyCapabilities
from .config import load_config as load_config
from .configuration_falcon import FalconConfig
from .detector import RFDETRDetector
from .grounding import resolve_segmentation as resolve_segmentation
from .model import FalconConfig as CoreConfig
from .model import FalconModel as CoreModel
from .prediction import predict, predict_panoptic, predict_segmentation
from .regions import MaskAwareRegionEncoder as MaskAwareRegionEncoder
from .safety import StructuredSafetyAdapter as StructuredSafetyAdapter


def _portable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _portable(item)
            for key, item in value.items()
            if key not in {
                "_name_or_path", "name_or_path", "auto_map", "custom_pipelines",
                "base_model_name_or_path",
            }
        }
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return deepcopy(value)


def _backbone_config(values: dict[str, Any]) -> Any:
    values = deepcopy(values)
    model_type = values.pop("model_type")
    return AutoConfig.for_model(model_type, **values)


def _dtype(value: str | torch.dtype | None) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value not in (None, "float32", "float16", "bfloat16"):
        raise ValueError(f"Unsupported backbone dtype: {value!r}")
    return getattr(torch, value or "float32")


@dataclass
class FalconOutput(ModelOutput):
    loss: Any = None
    language_loss: Any = None
    structured_losses: Any = None
    language_output: Any = None
    safety: Any = None
    region_embeddings: Any = None
    detector_output: Any = None
    prefix: Any = None


class FalconModel(PreTrainedModel):
    config_class = FalconConfig
    base_model_prefix = "core"
    main_input_name = "images"

    def __init__(self, config: FalconConfig, *, core: CoreModel | None = None) -> None:
        config.validate()
        super().__init__(config)
        if core is None:
            core_values = deepcopy(config.core_config)
            core_values["safety_capabilities"] = SafetyCapabilities.from_dict(
                core_values["safety_capabilities"]
            )
            core_config = CoreConfig(**core_values)
            vision = AutoModel.from_config(
                _backbone_config(config.vision_config),
                torch_dtype=_dtype(config.vision_config.get("torch_dtype")),
            )
            language = AutoModelForCausalLM.from_config(
                _backbone_config(config.language_config),
                torch_dtype=_dtype(config.language_config.get("torch_dtype")),
            )
            processor_values = deepcopy(config.image_processor_config)
            processor_class = get_image_processor_class_from_name(
                processor_values.pop("image_processor_type", "")
            )
            if processor_class is None:
                raise ValueError("Unknown image_processor_type in FALCON package")
            core = CoreModel(
                core_config,
                vision_encoder=vision,
                language_model=language,
                detector=RFDETRDetector.from_config(config.detector_config, device="cpu"),
                image_processor=processor_class.from_dict(processor_values),
            )
            core.enable_lora(**config.lora_config)
            core.set_training_stage(3)
            if config.language_generation_config:
                core.language_model.generation_config = GenerationConfig.from_dict(
                    config.language_generation_config
                )
                # This is an explicit saved generation policy, not a policy to
                # regenerate later from the language architecture's defaults.
                core.language_model.generation_config._from_model_config = False
        if core.training_stage != 3:
            raise ValueError("A Hugging Face FALCON model requires Stage 3 weights")
        self.core = core
        # RFDETRDetector is not an nn.Module. Register its actual backend module
        # explicitly so saving, loading, and device movement include every weight.
        self.detector_model = core.detector.torch_model
        self.detector_model.requires_grad_(False).eval()
        self.core.checkpoint_metadata = deepcopy(config.checkpoint_metadata)
        self.core.runtime_config = deepcopy(config.runtime_config)
        self.core.trained_safety_capabilities = core.config.safety_capabilities
        self.core.inference_safety_capabilities = core.config.safety_capabilities
        self.core.inference_ssa_ablation = "none"

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            core = self.__dict__.get("_modules", {}).get("core")
            if core is not None and hasattr(core, name):
                return getattr(core, name)
            raise

    @classmethod
    def from_falcon(cls, core: CoreModel) -> FalconModel:
        """Wrap loaded final weights without reallocating the frozen backbones."""
        source = deepcopy(getattr(core, "checkpoint_metadata", {}))
        if core.training_stage != 3 or source.get("training_complete") is not True:
            raise ValueError("Export requires a completed Stage 3 model")
        if source.get("partial_initialization_allowed") or source.get(
            "legacy_initialization_allowed"
        ) or source.get("oracle_detector", "none") != "none":
            raise ValueError("Cannot export partial initialization or oracle models")
        runtime = _portable(core.runtime_config)
        runtime["model"]["vision_model"] = "facebook/dinov2-large"
        runtime["model"]["language_model"] = "lmsys/vicuna-7b-v1.5"
        runtime["experiment"]["name"] = "falcon-x"
        retained = {
            "stage", "training_complete", "checkpoint_format", "safety_capabilities",
            "supervision_capabilities", "ssa_ablation", "require_observed_supervision",
            "observed_supervision_coverage", "dataset_categories",
            "diagnostic_training_reasons", "official_result_eligible", "precision",
            "structured_loss_recipe", "grounding_policy", "grounding_coverage",
        }
        metadata = {key: _portable(value) for key, value in source.items() if key in retained}
        metadata["stage"] = 3
        metadata["config"] = runtime
        vision_config = _portable(core.vision_encoder.config.to_dict())
        language_config = _portable(core.language_model.config.to_dict())
        for values, model in ((vision_config, core.vision_encoder),
                              (language_config, core.language_model)):
            values["torch_dtype"] = str(next(model.parameters()).dtype).removeprefix("torch.")
            values["attn_implementation"] = model.config._attn_implementation
        adapters = core.language_model.peft_config
        if set(adapters) != {"default"}:
            raise ValueError("FALCON export requires exactly one default LoRA adapter")
        lora = adapters["default"]
        if lora.bias != "none" or lora.modules_to_save or lora.use_dora or lora.use_rslora:
            raise ValueError("Unsupported LoRA variant in FALCON export")
        config = FalconConfig(
            core_config=asdict(core.config),
            vision_config=vision_config,
            language_config=language_config,
            detector_config=core.detector.export_config(),
            image_processor_config=_portable(core.image_processor.to_dict()),
            lora_config={
                "rank": lora.r, "alpha": lora.lora_alpha,
                "dropout": lora.lora_dropout, "target_modules": sorted(lora.target_modules),
            },
            runtime_config=runtime,
            checkpoint_metadata=metadata,
            language_generation_config=_portable(core.language_model.generation_config.to_dict()),
        )
        return cls(config, core=core)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        if kwargs.pop("torch_dtype", None) not in (None, "auto"):
            raise ValueError("FALCON preserves each component's saved dtype; omit torch_dtype")
        for name in ("quantization_config", "load_in_8bit", "load_in_4bit", "ignore_mismatched_sizes"):
            if kwargs.get(name):
                raise ValueError(f"FALCON does not support {name}")
        device = kwargs.pop("device", None)
        device_map = kwargs.pop("device_map", None)
        if device_map is not None:
            if isinstance(device_map, dict) and set(device_map) == {""}:
                mapped_device = device_map[""]
            elif isinstance(device_map, str | torch.device | int) and device_map not in (
                "auto", "balanced", "balanced_low_0", "sequential", "disk",
            ):
                mapped_device = device_map
            else:
                raise ValueError("FALCON supports one device; use device='cuda' or .to('cuda')")
            if isinstance(mapped_device, int):
                mapped_device = f"cuda:{mapped_device}"
            if device is not None and torch.device(device) != torch.device(mapped_device):
                raise ValueError("device and device_map disagree")
            device = mapped_device
        tokenizer_kwargs = {
            name: kwargs[name] for name in (
                "cache_dir", "force_download", "local_files_only", "token", "revision",
            ) if name in kwargs
        }
        package_subfolder = kwargs.get("subfolder", "")
        tokenizer_kwargs["subfolder"] = "/".join(
            part for part in (package_subfolder, "tokenizer") if part
        )
        requested_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(
            pretrained_model_name_or_path, *args, output_loading_info=True, **kwargs
        )
        issues = {name: info[name] for name in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
        ) if info.get(name)}
        if issues:
            raise ValueError(f"Incomplete or incompatible FALCON weights: {issues}")
        resolved_revision = getattr(model.config, "_commit_hash", None)
        if resolved_revision:
            tokenizer_kwargs["revision"] = resolved_revision
        model.core.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, use_fast=False, trust_remote_code=False,
            **tokenizer_kwargs,
        )
        if model.core.tokenizer.pad_token_id is None:
            model.core.tokenizer.pad_token = model.core.tokenizer.eos_token
        if device is not None:
            model.to(device)
        model.eval()
        return (model, info) if requested_info else model

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        if kwargs.get("push_to_hub"):
            raise ValueError("Save the complete folder first, then use push_to_hub() or upload_folder()")
        if self.core.tokenizer is None:
            raise ValueError("A complete FALCON package requires its tokenizer")
        self.config.validate()
        super().save_pretrained(save_directory, **kwargs)
        self.core.tokenizer.save_pretrained(Path(save_directory) / "tokenizer")
        self.core.image_processor.save_pretrained(save_directory)

    def _apply(self, fn: Any, recurse: bool = True):
        for dtype in (torch.float32, torch.bfloat16):
            if fn(torch.empty(0, dtype=dtype)).dtype != dtype:
                raise ValueError("FALCON uses mixed component dtypes; move devices without casting")
        result = super()._apply(fn, recurse=recurse)
        detector = self.__dict__.get("_modules", {}).get("detector_model")
        if detector is not None:
            self.core.detector.to(next(detector.parameters()).device)
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        self.detector_model.eval()
        return self

    def get_input_embeddings(self) -> nn.Module:
        return self.core.language_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.core.language_model.get_output_embeddings()

    def forward(self, **kwargs: Any) -> FalconOutput:
        return FalconOutput(**vars(self.core(**kwargs)))

    def generate(self, **kwargs: Any):
        return self.core.generate(**kwargs)

    @torch.inference_mode()
    def predict(self, image: Any, prompt: str, *, task: str = "text", **kwargs: Any):
        methods = {
            "text": predict, "segmentation": predict_segmentation, "panoptic": predict_panoptic,
        }
        if task not in methods:
            raise ValueError("task must be text, segmentation, or panoptic")
        return methods[task](self.core, image=image, prompt=prompt, **kwargs)

    @torch.inference_mode()
    def predict_segmentation(self, image: Any, prompt: str, **kwargs: Any):
        return predict_segmentation(self.core, image=image, prompt=prompt, **kwargs)

    @torch.inference_mode()
    def predict_panoptic(self, image: Any, prompt: str, **kwargs: Any):
        return predict_panoptic(self.core, image=image, prompt=prompt, **kwargs)


FalconModel.register_for_auto_class("AutoModel")
