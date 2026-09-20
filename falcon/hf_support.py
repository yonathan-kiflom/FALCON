"""Resolve and inspect a self-contained FALCON model without allocating weights."""

from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath
from typing import Any


def resolve_model_package(
    model: str | Path,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError("--model must name a model directory or a Hub model ID")
        if revision is not None:
            raise ValueError("--revision applies only to Hub models, not local directories")
        return path.resolve()
    if path.is_absolute() or str(model).startswith(("./", "../", "~")):
        raise FileNotFoundError(path)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(model),
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=["*.json", "*.py", "*.safetensors", "tokenizer/*"],
        )
    ).resolve()


def _asset(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Invalid model asset path: {name!r}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing model asset: {path}")
    return path


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_model_package(path: str | Path) -> dict[str, Any]:
    """Validate configuration, executable-code inventory and safetensors headers.

    Architecture-specific tensor shapes are additionally checked by the strict
    model loader. This inspection never executes code from the model repository.
    """
    from safetensors import safe_open

    from .configuration_falcon import FalconConfig

    root = Path(path).expanduser().resolve(strict=True)
    raw = _read_object(_asset(root, "config.json"))
    if raw.get("model_type") != "falcon_x":
        raise ValueError("The model must be a FALCON falcon_x package")
    config = FalconConfig.from_dict(raw)
    config.validate()
    expected_auto_map = {
        "AutoConfig": "configuration_falcon.FalconConfig",
        "AutoModel": "modeling_falcon.FalconModel",
    }
    for key, value in expected_auto_map.items():
        if raw.get("auto_map", {}).get(key) != value:
            raise ValueError(f"FALCON config requires auto_map.{key}={value!r}")

    pending = ["configuration_falcon.py", "modeling_falcon.py"]
    source_files: set[str] = set()
    while pending:
        name = pending.pop()
        if name in source_files:
            continue
        tree = ast.parse(_asset(root, name).read_text(encoding="utf-8"), filename=name)
        source_files.add(name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or not node.module or "." in node.module:
                    raise ValueError("FALCON custom model code must use flat relative imports")
                pending.append(node.module + ".py")

    _read_object(_asset(root, "tokenizer/tokenizer_config.json"))
    if not any((root / "tokenizer" / name).is_file() for name in ("tokenizer.model", "tokenizer.json")):
        raise ValueError("FALCON package lacks tokenizer vocabulary")
    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if single.exists() == index.exists():
        raise ValueError("Expected either model.safetensors or its shard index")
    weight_map = None
    if index.exists():
        weight_map = _read_object(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Model shard index lacks a nonempty weight_map")
        if not all(isinstance(k, str) and isinstance(v, str) and v.endswith(".safetensors") for k, v in weight_map.items()):
            raise ValueError("Invalid safetensors weight map")
        names = sorted(set(weight_map.values()))
    else:
        names = ["model.safetensors"]
    actual: dict[str, str] = {}
    for name in names:
        with safe_open(str(_asset(root, name)), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in actual:
                    raise ValueError(f"Duplicate tensor in model shards: {key}")
                actual[key] = name
    if weight_map is not None and actual != weight_map:
        raise ValueError("Safetensors shard inventory differs from its weight map")
    for component in (
        "core.vision_encoder.", "core.language_model.", "core.patch_aggregator.",
        "core.region_encoder.", "core.safety_adapter.", "core.image_projection.",
        "core.region_projection.", "core.region_index_embedding.", "detector_model.",
    ):
        if not any(key.startswith(component) for key in actual):
            raise ValueError(f"Model package is missing component: {component}")
    if not any(".lora_A." in key for key in actual) or not any(".lora_B." in key for key in actual):
        raise ValueError("Model package is missing the trained Stage-3 LoRA tensors")
    return {
        "config": raw,
        "metadata": config.checkpoint_metadata,
        "runtime_config": config.runtime_config,
        "weight_files": [str(root / name) for name in names],
        "source_files": sorted(source_files),
        "tensor_count": len(actual),
    }
