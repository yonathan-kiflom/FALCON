"""Explicit structured-head capabilities, independent of model dependencies.

Reference availability, training coverage and prediction availability are separate
facts. A capability records which trained heads may be used; it does not certify
accuracy. Missing checkpoint metadata must not be interpreted as all-enabled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SafetyCapabilities:
    risk: bool = True
    presence: tuple[bool, bool, bool] = (True, True, True)
    links: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        if not isinstance(self.risk, bool):
            raise ValueError("risk capability must be boolean")
        for name in ("presence", "links"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or len(values) != 3
                or any(not isinstance(value, bool) for value in values)
            ):
                raise ValueError(f"{name} capabilities must be a tuple of three booleans")

    @property
    def token_indices(self) -> tuple[int, ...]:
        flags = (self.risk, *self.presence, *self.links)
        return tuple(index for index, enabled in enumerate(flags) if enabled)

    def as_dict(self) -> dict[str, Any]:
        return {"risk": self.risk, "presence": list(self.presence), "links": list(self.links)}

    @classmethod
    def unavailable(cls) -> SafetyCapabilities:
        return cls(False, (False, False, False), (False, False, False))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SafetyCapabilities:
        if not isinstance(value, Mapping) or set(value) != {"risk", "presence", "links"}:
            raise ValueError("safety capabilities require exactly risk, presence, and links")
        if not isinstance(value["presence"], list | tuple) or not isinstance(
            value["links"], list | tuple
        ):
            raise ValueError("presence and links capabilities must be arrays")
        return cls(value["risk"], tuple(value["presence"]), tuple(value["links"]))

    def restrict(self, allowed: SafetyCapabilities) -> SafetyCapabilities:
        """Ablations can disable capabilities, never enable untrained heads."""
        return SafetyCapabilities(
            self.risk and allowed.risk,
            tuple(a and b for a, b in zip(self.presence, allowed.presence, strict=True)),
            tuple(a and b for a, b in zip(self.links, allowed.links, strict=True)),
        )


ALL_SAFETY_HEADS = SafetyCapabilities()
SSA_ABLATIONS = ("none", "no_ssa", "no_presence", "no_links", "no_risk")


def apply_ablation(capabilities: SafetyCapabilities, name: str) -> SafetyCapabilities:
    """Disable named heads before token construction; never invent capability."""
    if name not in SSA_ABLATIONS:
        raise ValueError(f"unknown SSA ablation {name!r}")
    mask = {
        "none": ALL_SAFETY_HEADS,
        "no_ssa": SafetyCapabilities.unavailable(),
        "no_presence": SafetyCapabilities(True, (False, False, False), (True, True, True)),
        "no_links": SafetyCapabilities(True, (True, True, True), (False, False, False)),
        "no_risk": SafetyCapabilities(False, (True, True, True), (True, True, True)),
    }[name]
    return capabilities.restrict(mask)


def supervision_coverage(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count authoritative labels per image, without substituting missing values.

    Callers must provide unique training images and validated provenance. This
    function validates numeric domains, not the authority of the source itself.
    """
    coverage = {"images": 0, "risk": 0, "presence": [0, 0, 0], "links": [0, 0, 0]}

    def observed(value: Any, *, binary: bool = False) -> bool:
        if value is None:
            return False
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("structured labels must be numeric or null")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("structured labels must be finite and in [0, 1]")
        if binary and value not in (0, 1):
            raise ValueError("presence labels must be binary")
        return True

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("structured targets must be objects")
        coverage["images"] += 1
        coverage["risk"] += int(observed(row.get("risk")))
        for name in ("presence", "links"):
            values = row.get(name, [None, None, None])
            if not isinstance(values, list | tuple) or len(values) != 3:
                raise ValueError(f"{name} targets must have three entries")
            for index, value in enumerate(values):
                coverage[name][index] += int(observed(value, binary=name == "presence"))
    return coverage


def capabilities_from_coverage(coverage: Mapping[str, Any]) -> SafetyCapabilities:
    """An observed label enables training, not a claim of calibrated predictions."""
    for name in ("images", "risk"):
        value = coverage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} coverage must be a nonnegative integer")
    counts = []
    for name in ("presence", "links"):
        values = coverage.get(name)
        if (
            not isinstance(values, list | tuple)
            or len(values) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
        ):
            raise ValueError(f"{name} coverage must contain three nonnegative integers")
        counts.extend(values)
    if any(value > coverage["images"] for value in [coverage["risk"], *counts]):
        raise ValueError("label coverage cannot exceed training image count")
    return SafetyCapabilities(
        coverage["risk"] > 0,
        tuple(value > 0 for value in coverage["presence"]),
        tuple(value > 0 for value in coverage["links"]),
    )
