"""Private, persistent inference process for the local demo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from typing import Any


_REQUEST_FIELDS = {
    "id", "image", "mode", "prompt", "max_new_tokens", "query_category_id",
}
# Uploads are capped at 20 MiB before conversion. A valid JPEG can expand to a
# larger RGB PNG on this private transport; 16 MP bounds its decoded size.
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_PIXELS = 16_000_000
_MAX_MESSAGE = 65_536
_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")


def _input_path(workspace: Path, request: dict[str, Any]) -> Path:
    request_id = request.get("id")
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise ValueError("Request id must be a lowercase UUID hex string")
    image = request.get("image")
    if not isinstance(image, str) or "\\" in image:
        raise ValueError("Image must be a relative workspace path")
    relative = PurePosixPath(image)
    if (
        relative.is_absolute() or image != relative.as_posix()
        or len(relative.parts) != 2 or relative.parts[0] != request_id
        or relative.name not in {"input.png", "input.jpg", "input.jpeg"}
    ):
        raise ValueError("Image must be request-id/input.png, input.jpg, or input.jpeg")
    path = workspace
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symbolic links are not allowed in request paths")
    if not path.is_file() or not path.resolve().is_relative_to(workspace):
        raise ValueError("Image must be a regular file inside the worker workspace")
    if path.stat().st_size > _MAX_IMAGE_BYTES:
        raise ValueError("Internal image exceeds the 64 MiB transport limit")
    return path


def _request_input(
    request: Any, workspace: Path, model: Any, category_ids: set[int],
) -> tuple[Path, Any]:
    from PIL import Image

    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    unknown = set(request) - _REQUEST_FIELDS
    if unknown:
        raise ValueError(f"Unsupported request fields: {', '.join(sorted(unknown))}")
    required = _REQUEST_FIELDS - {"query_category_id"}
    if required - set(request):
        raise ValueError(f"Missing request fields: {', '.join(sorted(required - set(request)))}")
    path = _input_path(workspace, request)
    if request["mode"] not in ("text", "segmentation", "panoptic"):
        raise ValueError("Mode must be text, segmentation, or panoptic")
    if type(request["max_new_tokens"]) is not int or request["max_new_tokens"] not in (
        64, 128, 256,
    ):
        raise ValueError("max_new_tokens must be 64, 128, or 256")
    prompt = request["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32_768:
        raise ValueError("Prompt must be nonempty text of at most 32768 characters")
    query = request.get("query_category_id")
    if "query_category_id" in request and (
        request["mode"] != "panoptic" or type(query) is not int or query not in category_ids
    ):
        raise ValueError("query_category_id must be a checkpoint category in panoptic mode")
    count = model.core.text_token_counts(f"USER: {prompt}\nASSISTANT:")["prompt_tokens"]
    budget = model.core.config.max_text_tokens
    if count > budget:
        raise ValueError(f"Prompt needs {count} tokens; the model allows {budget}. Shorten it.")
    with Image.open(path) as opened:
        if opened.format not in {"JPEG", "PNG"}:
            raise ValueError("Only JPEG and PNG images are supported")
        if opened.width * opened.height > _MAX_PIXELS:
            raise ValueError("Image exceeds the 16 megapixel limit")
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Animated images are not supported")
        opened.load()
        image = opened.convert("RGB").copy()
    return path, image


def _load_model(path: Path, device: str) -> tuple[Any, list[dict[str, Any]]]:
    import torch
    from transformers import AutoModel

    if not path.is_dir():
        raise ValueError("--model must point to a local exported FALCON directory")
    with (path / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    metadata = config.get("checkpoint_metadata", {})
    if (
        config.get("model_type") != "falcon_x"
        or metadata.get("stage") != 3 or metadata.get("training_complete") is not True
    ):
        raise ValueError("The demo requires an exported, completed Stage 3 FALCON model")
    categories = metadata.get("dataset_categories", [])
    expected_names = {1: "detonator", 2: "explosive", 3: "battery"}
    if (
        not isinstance(categories, list) or len(categories) != 3
        or any(
            not isinstance(item, dict) or type(item.get("id")) is not int
            or expected_names.get(item["id"]) != item.get("name")
            for item in categories
        )
        or {item["id"] for item in categories} != set(expected_names)
    ):
        raise ValueError("Checkpoint must define detonator, explosive, and battery categories 1–3")
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The local demo requires one available CUDA GPU")
    torch.cuda.set_device(selected)
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(
        str(path.resolve()), trust_remote_code=True, local_files_only=True,
    ).to(selected).eval()
    model.config.validate()
    return model, sorted(categories, key=lambda item: item["id"])


def _predict(
    request: dict[str, Any], workspace: Path, model: Any, category_ids: set[int], device: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    path, image = _request_input(request, workspace, model, category_ids)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    kwargs = {
        "image": image,
        "prompt": request["prompt"],
        "max_new_tokens": request["max_new_tokens"],
        "temperature": 0.0,
    }
    with torch.inference_mode():
        if request["mode"] == "panoptic":
            result, prediction = model.predict_panoptic(
                **kwargs, category_ids=category_ids,
                query_category_id=request.get("query_category_id"),
            )
            array = np.asarray(prediction.id_map)
            if array.shape != (image.height, image.width) or array.dtype.kind not in "iu":
                raise ValueError("Panoptic prediction must be a native-size integer id map")
            if np.any(array < 0):
                raise ValueError("Panoptic prediction contains negative segment IDs")
            artifact_name, array_key = "panoptic", "id_map"
        elif request["mode"] == "segmentation":
            result, prediction = model.predict_segmentation(**kwargs)
            result["region_ids"] = list(prediction.region_indices)
            array = np.asarray(prediction.masks)
            artifact_name, array_key = "masks", "masks"
        else:
            result, masks = model.predict(**kwargs)
            array = np.asarray(masks)
            artifact_name, array_key = "masks", "masks"
    if array_key == "masks" and (
        array.ndim != 3 or array.shape[1:] != (image.height, image.width)
        or array.dtype != np.bool_ or len(result.get("region_ids", [])) != len(array)
    ):
        raise ValueError("Segmentation prediction must contain native-size binary masks and region IDs")
    # Verify JSON portability before creating an output file.
    json.dumps(result, allow_nan=False)
    artifacts = {}
    if request["mode"] != "text" or len(array):
        artifact = path.parent / f"{artifact_name}.npz"
        with artifact.open("xb") as handle:
            np.savez_compressed(handle, **{array_key: array})
        artifacts[artifact_name] = artifact.relative_to(workspace).as_posix()
    torch.cuda.synchronize(device)
    return {
        "id": request["id"], "status": "ok", "result": result, "artifacts": artifacts,
        "metrics": {
            "seconds": round(time.monotonic() - start, 3),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
        },
    }


def _fatal_cuda(error: Exception) -> bool:
    import torch

    return isinstance(error, torch.cuda.OutOfMemoryError) or any(
        phrase in str(error).lower() for phrase in (
            "cuda error", "cuda out of memory", "device-side assert", "illegal memory access",
            "cublas_status", "cudnn_status", "nccl error",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Preserve one protocol descriptor; Python and native-library logs go to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "DO_NOT_TRACK"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def emit(value: dict[str, Any]) -> None:
        protocol.write(json.dumps(value, allow_nan=False) + "\n")
        protocol.flush()

    try:
        import torch

        if args.workspace.is_symlink() or not args.workspace.is_dir():
            raise ValueError("Workspace must be an existing directory, not a symbolic link")
        workspace = args.workspace.resolve()
        start = time.monotonic()
        model, categories = _load_model(args.model, args.device)
        torch.cuda.synchronize(args.device)
        emit({"event": "ready", "metadata": {
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(args.device),
            "model": str(args.model.resolve()),
            "load_seconds": round(time.monotonic() - start, 3),
            "allocated_mib": round(torch.cuda.memory_allocated(args.device) / 2**20, 1),
            "reserved_mib": round(torch.cuda.memory_reserved(args.device) / 2**20, 1),
            "categories": categories,
            "capabilities": model.core.config.safety_capabilities.as_dict(),
            "max_text_tokens": model.core.config.max_text_tokens,
        }})
    except Exception as error:
        emit({"event": "fatal", "error": f"{type(error).__name__}: {error}"})
        return

    while True:
        line = sys.stdin.readline(_MAX_MESSAGE + 1)
        if not line:
            break
        request: Any = None
        try:
            if len(line) > _MAX_MESSAGE:
                emit({"event": "fatal", "error": "Worker request exceeds the message limit"})
                break
            request = json.loads(line)
            emit(_predict(request, workspace, model, {item["id"] for item in categories}, args.device))
        except Exception as error:
            fatal = _fatal_cuda(error)
            response = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "status": "error", "error": f"{type(error).__name__}: {error}",
            }
            if fatal:
                response["fatal"] = True
                response["error"] += "; worker stopped. Restart the demo before retrying."
            emit(response)
            if fatal:
                break
    protocol.close()


if __name__ == "__main__":
    main()
