"""Cache frozen detector and vision features between training stages."""

from __future__ import annotations

import copy
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch

from .artifacts import atomic_json
from .checkpoint import atomic_torch
from .detector import DetectorOutput
from .model import VisionFeatures


class FrozenFeatureCache:
    """One cache directory for a fixed dataset and model configuration."""

    def __init__(self, root: str | Path, identity: dict[str, Any]):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = copy.deepcopy(identity)
        self.identity["model_files"] = {
            name: self._file_info(Path(identity[name]).expanduser())
            for name in ("vision_model", "detector_weights")
            if name in identity
        }
        with self._lock():
            path = self.root / "config.json"
            if path.exists():
                if json.loads(path.read_text()) != self.identity:
                    raise ValueError(
                        "Feature cache configuration changed; choose a new cache directory"
                    )
            else:
                atomic_json(path, self.identity)

    @staticmethod
    def _file_info(path: Path) -> list:
        if not path.exists():
            return []
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        return [
            [str(item.resolve()), item.stat().st_size, item.stat().st_mtime_ns]
            for item in files
            if item.is_file()
        ]

    @contextmanager
    def _lock(self):
        with (self.root / ".cache.lock").open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def _path(self, image_key: str) -> Path:
        if not isinstance(image_key, str) or not image_key:
            raise ValueError("Feature cache image key must be a nonempty string")
        name = quote(image_key, safe="._-")
        # Nested chunks avoid filesystem filename limits for long image IDs.
        parts = [name[index : index + 120] for index in range(0, len(name), 120)]
        path = self.root.joinpath("images", *parts[:-1], parts[-1] + ".pt")
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Feature cache path must stay inside its directory")
        return path

    @staticmethod
    def _validate(
        payload: Any, image_hw: tuple[int, int]
    ) -> tuple[DetectorOutput, VisionFeatures]:
        required = {
            "boxes",
            "scores",
            "masks",
            "proposal_ids",
            "image_hw",
            "patch_tokens",
            "grid_size",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("frozen cache payload has unexpected fields")
        if (
            not isinstance(image_hw, tuple)
            or len(image_hw) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in image_hw
            )
        ):
            raise ValueError("frozen cache expected image dimensions are invalid")
        payload_hw = payload["image_hw"]
        if (
            isinstance(payload_hw, str | bytes)
            or not isinstance(payload_hw, list | tuple)
            or tuple(payload_hw) != image_hw
        ):
            raise ValueError("frozen cache image dimensions differ")
        boxes, scores, masks = (payload[key] for key in ("boxes", "scores", "masks"))
        if any(not isinstance(value, torch.Tensor) for value in (boxes, scores, masks)):
            raise ValueError("frozen cache detector outputs must be tensors")
        n = boxes.shape[0]
        if (
            boxes.shape != (n, 4)
            or scores.shape != (n,)
            or masks.shape != (n, *image_hw)
        ):
            raise ValueError("frozen cache detector shapes differ")
        height, width = image_hw
        valid_boxes = (
            torch.is_floating_point(boxes)
            and bool(torch.isfinite(boxes).all())
            and bool((boxes[:, 0] >= 0).all())
            and bool((boxes[:, 1] >= 0).all())
            and bool((boxes[:, 2] <= width).all())
            and bool((boxes[:, 3] <= height).all())
            and bool((boxes[:, 2] > boxes[:, 0]).all())
            and bool((boxes[:, 3] > boxes[:, 1]).all())
        )
        if (
            masks.dtype != torch.bool
            or not valid_boxes
            or not torch.is_floating_point(scores)
            or not bool(torch.isfinite(scores).all())
        ):
            raise ValueError("frozen cache has invalid box/score/mask values")
        features = payload["patch_tokens"]
        grid = tuple(payload["grid_size"])
        if (
            len(grid) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in grid)
            or not isinstance(features, torch.Tensor)
            or features.ndim != 3
            or features.shape[:2] != (1, grid[0] * grid[1])
            or not torch.is_floating_point(features)
            or not bool(torch.isfinite(features).all())
        ):
            raise ValueError("frozen cache has invalid DINO patch features")
        proposal_ids = payload["proposal_ids"]
        if proposal_ids is not None:
            if (
                not isinstance(proposal_ids, torch.Tensor)
                or proposal_ids.shape != (n,)
                or proposal_ids.dtype != torch.long
                or bool((proposal_ids < 0).any())
                or len(torch.unique(proposal_ids)) != n
            ):
                raise ValueError("frozen cache proposal identities are invalid")
        return (
            DetectorOutput(boxes, scores, masks, image_hw, proposal_ids),
            VisionFeatures(features, grid),
        )

    def get(
        self, image_key: str, image_hw: tuple[int, int]
    ) -> tuple[DetectorOutput, VisionFeatures] | None:
        path = self._path(image_key)
        with self._lock():
            if not path.exists():
                return None
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("image_key") != image_key:
                raise ValueError("Feature cache entry belongs to another image")
            if saved.get("image_file") != self._file_info(Path(image_key)):
                raise ValueError("Cached image changed; choose a new cache directory")
            return self._validate(saved["features"], image_hw)

    def put(self, image_key: str, detected: Any, features: VisionFeatures) -> None:
        path = self._path(image_key)
        proposal_ids = getattr(detected, "proposal_ids", None)
        payload = {
            "boxes": detected.boxes_xyxy.detach().cpu(),
            "scores": detected.scores.detach().cpu(),
            "masks": detected.masks.detach().cpu(),
            "image_hw": list(detected.original_hw),
            "proposal_ids": None
            if proposal_ids is None
            else proposal_ids.detach().cpu(),
            "patch_tokens": features.patch_tokens.detach().cpu(),
            "grid_size": list(features.grid_size),
        }
        self._validate(payload, tuple(detected.original_hw))
        with self._lock():
            atomic_torch(
                path,
                {
                    "image_key": image_key,
                    "image_file": self._file_info(Path(image_key)),
                    "features": payload,
                },
            )
