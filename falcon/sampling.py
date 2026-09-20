"""Deterministic task-family sampling."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskEpochPlan:
    indices: tuple[int, ...]
    report: dict[str, Any]


def plan_epoch(
    examples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    epoch: int,
    mode: str = "all",
    weights: Mapping[str, float] | None = None,
    epoch_size: int | None = None,
) -> TaskEpochPlan:
    """All-row coverage or exact family quotas with explicit replacement.

    Family-balanced quotas use largest remainders. Each family is shuffled in
    complete cycles before a repeated example is drawn, maximizing unique
    coverage for its quota. The final selected rows are grouped by image for
    frozen-feature reuse; this changes order, not quotas or coverage.
    """
    if not examples:
        raise ValueError("cannot sample an empty task dataset")
    for name, value in (("seed", seed), ("epoch", epoch)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if mode not in ("all", "family_balanced"):
        raise ValueError("sampling mode must be all or family_balanced")
    rng = random.Random(f"falcon-task-sampling/v1:{seed}:{epoch}")
    families: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        family = example.get("family", example.get("task"))
        if not isinstance(family, str) or not family:
            raise ValueError("every example must declare a family or legacy task")
        if "image_id" not in example:
            raise ValueError("every example must declare image_id")
        families[family].append(index)
    if epoch_size is not None and (
        isinstance(epoch_size, bool) or not isinstance(epoch_size, int) or epoch_size < 1
    ):
        raise ValueError("epoch_size must be a positive integer")
    if mode == "all":
        if epoch_size not in (None, len(examples)):
            raise ValueError(
                "sampling=all requires epoch_size equal to the number of tasks or null"
            )
        quotas = {family: len(pool) for family, pool in families.items()}
    else:
        if weights is None or not isinstance(weights, Mapping):
            raise ValueError("family_balanced requires explicit family weights")
        missing = set(families).difference(weights)
        if missing:
            raise ValueError(f"missing weights for selected task families: {sorted(missing)}")
        selected_weights = {family: weights[family] for family in families}
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0
            for value in selected_weights.values()
        ):
            raise ValueError("selected family weights must be finite and positive")
        budget = len(examples) if epoch_size is None else epoch_size
        total_weight = sum(selected_weights.values())
        exact = {
            family: budget * weight / total_weight for family, weight in selected_weights.items()
        }
        quotas = {family: math.floor(value) for family, value in exact.items()}
        order = sorted(exact, key=lambda family: (-(exact[family] - quotas[family]), family))
        for family in order[: budget - sum(quotas.values())]:
            quotas[family] += 1

    selected = []
    unique_counts = {}
    for family in sorted(families):
        pool = families[family]
        remaining = quotas[family]
        unique_counts[family] = min(remaining, len(pool))
        while remaining:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            take = min(remaining, len(shuffled))
            selected.extend(shuffled[:take])
            remaining -= take
    groups = defaultdict(list)
    for index in selected:
        image_id = examples[index]["image_id"]
        # int 1 and string '1' must remain distinct split-scoped identities.
        groups[(type(image_id).__name__, image_id)].append(index)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    ordered = []
    for group in grouped:
        rng.shuffle(group)
        ordered.extend(group)
    report = {
        "version": "falcon-task-sampling/v1",
        "mode": mode,
        "seed": seed,
        "epoch": epoch,
        "draws": len(ordered),
        "unique_tasks": len(set(ordered)),
        "repeated_draws": len(ordered) - len(set(ordered)),
        "dataset_tasks": len(examples),
        "unique_images": len(groups),
        "coverage_fraction": len(set(ordered)) / len(examples),
        "families": {
            family: {
                "available": len(families[family]),
                "draws": quotas[family],
                "unique_tasks": unique_counts[family],
                "coverage_fraction": unique_counts[family] / len(families[family]),
            }
            for family in sorted(families)
        },
    }
    return TaskEpochPlan(tuple(ordered), report)
