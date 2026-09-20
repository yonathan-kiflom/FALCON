"""One resumable checkpoint per training stage."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate.utils import broadcast_object_list, gather_object

FORMAT = "falcon-checkpoint"


def rng_state() -> dict[str, Any]:
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("resume CUDA RNG topology differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in names
    }


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Checkpoint destination must be a regular file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_training_state(
    path: Path,
    *,
    accelerator: Any,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    metadata: dict[str, Any],
    cursor: dict[str, Any],
) -> None:
    """Replace last.pt atomically; DDP ranks share weights but retain their own RNG."""
    if not accelerator.sync_gradients:
        raise ValueError("training state can only be saved at an accumulation boundary")
    rank = getattr(accelerator, "process_index", 0)
    scaler = getattr(accelerator, "scaler", None)
    rank_states = gather_object(
        [
            {
                "rank": rank,
                "cursor": cursor,
                "rng": rng_state(),
                "scaler": None if scaler is None else scaler.state_dict(),
            }
        ]
    )
    error = [None]
    if accelerator.is_main_process:
        try:
            atomic_torch(
                path,
                {
                    "format": FORMAT,
                    "stage": metadata["stage"],
                    "state_dict": trainable_state(accelerator.unwrap_model(model)),
                    "metadata": metadata,
                    "training": {
                        "world_size": accelerator.num_processes,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "rank_states": rank_states,
                    },
                },
            )
        except (OSError, RuntimeError, ValueError) as exc:
            error[0] = str(exc)
    broadcast_object_list(error)
    if error[0] is not None:
        raise RuntimeError(f"Could not save checkpoint: {error[0]}")


def inspect_training_state(
    path: str | Path, *, metadata: dict[str, Any], world_size: int, rank: int = 0
) -> dict[str, Any]:
    """Read the shared weights and this rank's resume state."""
    path = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Resume requires a last.pt checkpoint saved by this trainer")
    saved = payload.get("metadata")
    if not isinstance(saved, dict) or payload.get("stage") != metadata["stage"]:
        raise ValueError("Resume checkpoint has missing metadata or a different stage")
    if saved.get("training_complete") is True:
        raise ValueError("This training stage is already complete")
    for key, value in metadata.items():
        if key != "training_complete" and saved.get(key) != value:
            raise ValueError(f"Resume training settings differ at {key}")
    training = payload.get("training")
    if not isinstance(training, dict) or training.get("world_size") != world_size:
        raise ValueError(
            "Resume training state is missing or distributed world size differs"
        )
    if not 0 <= rank < world_size:
        raise ValueError("Resume rank is outside world size")
    rank_states = training.get("rank_states")
    if (
        not isinstance(rank_states, list)
        or len(rank_states) != world_size
        or any(not isinstance(item, dict) for item in rank_states)
        or {item.get("rank") for item in rank_states} != set(range(world_size))
    ):
        raise ValueError("Resume rank inventory is incomplete")
    local = next(item for item in rank_states if item["rank"] == rank)
    cursor = local.get("cursor")
    if not isinstance(cursor, dict) or any(
        isinstance(cursor.get(key), bool)
        or not isinstance(cursor.get(key), int)
        or cursor[key] < 0
        for key in ("epoch", "next_batch", "batches_seen", "updates")
    ):
        raise ValueError("Resume cursor is malformed")
    for key in ("optimizer", "scheduler"):
        if not isinstance(training.get(key), dict):
            raise ValueError(f"Resume {key} state is missing")
    if not isinstance(local.get("rng"), dict) or "scaler" not in local:
        raise ValueError("Resume RNG or scaler state is missing")
    return {
        "state_dict": payload.get("state_dict"),
        "optimizer": training["optimizer"],
        "scheduler": training["scheduler"],
        **local,
    }


def restore_training_state(
    payload: dict[str, Any],
    *,
    accelerator: Any,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
) -> None:
    unwrapped = accelerator.unwrap_model(model)
    expected = {
        name
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }
    state = payload.get("state_dict")
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError(
            "Resume trainable parameter inventory differs from the active model"
        )
    if any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("Resume checkpoint contains invalid parameter values")
    incompatible = unwrapped.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("Resume contains unexpected model parameters")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler = getattr(accelerator, "scaler", None)
    if (scaler is None) != (payload["scaler"] is None):
        raise ValueError("Resume precision/scaler topology differs")
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"])
