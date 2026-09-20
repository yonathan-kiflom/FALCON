"""File output and panoptic image utilities."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def panoptic_ids_from_image(image):
    """Decode integer or RGB(A) panoptic IDs without rescaling."""

    import numpy as np

    array = np.asarray(image)
    if array.ndim == 2:
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("panoptic map must contain integer IDs")
        return array.astype(np.int64)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[..., :3].astype(np.int64)
        return rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    raise ValueError("panoptic PNG must be integer or RGB")


def external_output(
    path: str | Path,
    dataset_root: str | Path,
    *,
    protected_roots: Iterable[str | Path] = (),
) -> Path:
    """Keep generated files outside dataset and annotation directories."""

    output = Path(path).expanduser().resolve()
    requested_roots = (dataset_root, *tuple(protected_roots))
    immutable_roots: list[Path] = []
    for raw_root in requested_roots:
        root = Path(raw_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Protected input root is not a directory: {root}")
        if root not in immutable_roots:
            immutable_roots.append(root)
    for root in immutable_roots:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError(
                "Derived output must be outside immutable input roots and not an ancestor of one"
            )
    return output


def atomic_json(path: str | Path, value: Any) -> None:
    """Write a complete UTF-8 JSON document, then atomically replace its target."""
    path = Path(path)
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
