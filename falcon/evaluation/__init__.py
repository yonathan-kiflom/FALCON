"""Offline, task-level evaluation for the compact Falcon dataset."""

from .runner import build_run_manifest, run_predictions
from .schemas import (
    PREDICTION_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    EvaluationError,
    ModelOutputError,
    PredictionCoverageError,
    PredictionFormatError,
    PredictionSet,
    ReferenceValidationError,
    load_prediction_set,
    load_run_manifest,
)
from .scorer import score_dataset, validate_evaluation

__all__ = [
    "PREDICTION_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "EvaluationError",
    "ModelOutputError",
    "PredictionCoverageError",
    "PredictionFormatError",
    "PredictionSet",
    "ReferenceValidationError",
    "load_prediction_set",
    "load_run_manifest",
    "build_run_manifest",
    "run_predictions",
    "score_dataset",
    "validate_evaluation",
]
