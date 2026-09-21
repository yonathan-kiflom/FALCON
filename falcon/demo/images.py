"""Input validation and native-resolution prediction rendering."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from ..data import CANONICAL_CATEGORY_IDS, CANONICAL_COMPONENT_ORDER

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
_COLORS = (
    (0, 174, 239), (255, 170, 0), (219, 74, 134), (58, 201, 139),
    (166, 120, 255), (255, 104, 77), (0, 207, 207), (221, 207, 50),
)
_CATEGORY_NAMES = {value: key for key, value in CANONICAL_CATEGORY_IDS.items()}


def load_image(path: str | Path) -> Image.Image:
    """Decode one PNG/JPEG as RGB without resizing or EXIF rotation."""
    requested = Path(path)
    try:
        if not requested.is_file() or requested.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Use a JPEG or PNG file no larger than 20 MB.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(requested) as opened:
                if opened.format not in ("JPEG", "PNG"):
                    raise ValueError("Only JPEG and PNG images are supported.")
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Images must contain no more than 16 megapixels.")
                if getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported.")
                opened.verify()
            with Image.open(requested) as opened:
                normalized = opened.convert("RGB").copy()
                normalized.info.clear()
                return normalized
    except (
        OSError, UnidentifiedImageError, SyntaxError,
        Image.DecompressionBombError, Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("The image is corrupt, unreadable, or too large.") from exc


def _array(workspace: Path, relative: str, key: str) -> np.ndarray:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("Invalid prediction artifact path")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or parsed.as_posix() != relative or ".." in parsed.parts:
        raise ValueError("Prediction artifacts must stay inside the demo workspace")
    root = Path(workspace).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.suffix != ".npz":
        raise ValueError("Invalid prediction artifact")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != [key]:
            raise ValueError(f"Prediction artifact must contain only {key!r}")
        return archive[key]


def _color(index: int) -> tuple[int, int, int]:
    return _COLORS[index % len(_COLORS)]


def _result(response: dict) -> dict:
    result = response.get("result", response)
    if not isinstance(result, dict):
        raise ValueError("Prediction result must be an object")
    return result


def _overlay(image: Image.Image, regions: list[tuple[np.ndarray, int, str]]) -> Image.Image:
    pixels = np.array(image.convert("RGB"), copy=True)
    labels = []
    for mask, index, label in regions:
        if not mask.any():
            continue
        color = np.asarray(_color(index), dtype=np.float32)
        pixels[mask] = (pixels[mask].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)
        interior = mask.copy()
        interior[1:] &= mask[:-1]
        interior[:-1] &= mask[1:]
        interior[:, 1:] &= mask[:, :-1]
        interior[:, :-1] &= mask[:, 1:]
        interior[[0, -1], :] = False
        interior[:, [0, -1]] = False
        pixels[mask & ~interior] = color.astype(np.uint8)
        ys, xs = np.nonzero(mask)
        labels.append((int(xs.min()), int(ys.min()), label, tuple(color.astype(int))))
    result = Image.fromarray(pixels)
    draw = ImageDraw.Draw(result)
    for x, y, label, color in labels:
        box = draw.textbbox((x, y), label)
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((x, y), label, fill=color)
    return result


def render_prediction(
    image: Image.Image, response: dict, workspace: Path,
) -> tuple[Image.Image | None, str]:
    """Render only worker prediction artifacts; never substitute dataset masks."""
    height, width = image.height, image.width
    result = _result(response)
    artifacts = response.get("artifacts", response)
    if not isinstance(artifacts, dict):
        raise ValueError("Prediction artifacts must be an object")
    if artifacts.get("masks") is not None and artifacts.get("panoptic") is not None:
        raise ValueError("A prediction cannot contain both mask and panoptic artifacts")
    regions: list[tuple[np.ndarray, int, str]] = []
    legend: list[str] = []
    if artifacts.get("panoptic") is not None:
        id_map = _array(workspace, artifacts["panoptic"], "id_map")
        if (
            id_map.shape != (height, width) or id_map.dtype.kind not in "iu"
            or np.any(id_map < 0)
        ):
            raise ValueError("Panoptic prediction must be a native-size nonnegative integer map")
        segments = result.get("segments_info")
        if not isinstance(segments, list):
            raise ValueError("Panoptic prediction lacks its segment metadata")
        seen: set[int] = set()
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("Invalid panoptic segment metadata")
            identifier, category = segment.get("id"), segment.get("category_id")
            if (
                isinstance(identifier, bool) or not isinstance(identifier, int)
                or identifier < 1 or identifier in seen
                or isinstance(category, bool) or not isinstance(category, int)
                or category not in _CATEGORY_NAMES
            ):
                raise ValueError("Invalid panoptic segment ID or category")
            seen.add(identifier)
            name = _CATEGORY_NAMES[category]
            regions.append((id_map == identifier, identifier, f"{name} {identifier}"))
            legend.append(f"Instance {identifier}: {name} — #{'%02x%02x%02x' % _color(identifier)}")
        if set(np.unique(id_map)).difference({0}) != seen:
            raise ValueError("Panoptic segment metadata does not match the predicted map")
    elif artifacts.get("masks") is not None:
        masks = _array(workspace, artifacts["masks"], "masks")
        if (
            masks.ndim != 3 or masks.shape[1:] != (height, width)
            or masks.dtype.kind not in "bui" or np.any((masks != 0) & (masks != 1))
        ):
            raise ValueError("Segmentation prediction must contain native-size binary masks")
        identifiers = result.get("region_ids")
        if (
            not isinstance(identifiers, list) or len(identifiers) != len(masks)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("Segmentation region IDs do not match the predicted masks")
        for mask, identifier in zip(masks, identifiers, strict=True):
            if not mask.any():
                raise ValueError("A nonempty region reference has an empty predicted mask")
            regions.append((mask.astype(bool), identifier, f"Region {identifier}"))
            legend.append(f"Region {identifier} — #{'%02x%02x%02x' % _color(identifier)}")
    else:
        return None, ""
    return _overlay(image, regions), "\n".join(legend) if regions else "No regions predicted."


def presence_rows(response: dict) -> list[list]:
    """Return only explicitly enabled, finite component-head probabilities."""
    safety = _result(response).get("safety")
    safety = safety if isinstance(safety, dict) else {}
    capabilities = safety.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    enabled = capabilities.get("presence")
    probabilities = safety.get("presence")
    enabled = enabled if isinstance(enabled, list) and len(enabled) == 3 else [False] * 3
    probabilities = probabilities if isinstance(probabilities, list) and len(probabilities) == 3 else [None] * 3
    rows = []
    for name, flag, value in zip(CANONICAL_COMPONENT_ORDER, enabled, probabilities, strict=True):
        available = (
            flag is True and not isinstance(value, bool) and isinstance(value, int | float)
            and math.isfinite(value) and 0 <= value <= 1
        )
        rows.append([name, round(float(value), 4) if available else "Unavailable"])
    return rows


def parse_components(missing_response: dict, complete_response: dict) -> str:
    missing_raw = _result(missing_response).get("answer", "")
    complete_raw = _result(complete_response).get("answer", "")
    try:
        missing = json.loads(missing_raw)
        valid_missing = (
            isinstance(missing, list)
            and all(isinstance(name, str) and name in CANONICAL_COMPONENT_ORDER for name in missing)
            and len(missing) == len(set(missing))
        )
    except (ValueError, TypeError):
        valid_missing = False
    if valid_missing:
        lines = ["Missing categories: " + (", ".join(missing) or "none")]
    else:
        lines = [f"Warning: invalid missing-category answer. Raw answer: {missing_raw}"]
    complete = complete_raw.strip() if isinstance(complete_raw, str) else None
    if complete in ("0", "1"):
        lines.append(f"All three component categories present: {'Yes (1)' if complete == '1' else 'No (0)'}")
    else:
        lines.append(f"Warning: invalid completeness answer. Raw answer: {complete_raw}")
    return "\n".join(lines)
