"""Stage 3 task evaluation, readiness checks, and offline scoring."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _task_parser(command: str) -> argparse.ArgumentParser:
    from .tasks import DATASET_PROTOCOL, PAPER_PROTOCOL

    parser = argparse.ArgumentParser(
        prog=f"falcon evaluate {command}",
        description=f"{command.capitalize()} saved task predictions for falcon-x",
    )
    parser.add_argument(
        "--dataset", required=True, help="falcon-x dataset root or dataset.json"
    )
    parser.add_argument(
        "--annotations",
        action="append",
        default=[],
        help="Safety or panoptic annotation package (repeatable)",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--partition",
        help="Train partition JSON (requires --split train)",
    )
    parser.add_argument(
        "--partition-role",
        choices=("validation", "calibration"),
        default="validation",
        help="Partition role to select when --partition is supplied",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="'all' or a comma-separated list of stable task family names",
    )
    parser.add_argument(
        "--protocol",
        choices=(DATASET_PROTOCOL, PAPER_PROTOCOL),
        default=DATASET_PROTOCOL,
    )
    if command in ("validate", "score"):
        parser.add_argument(
            "--predictions", help="Saved run directory or prediction JSONL"
        )
    parser.add_argument("--output", help="Optional JSON report path")
    if command == "validate":
        parser.add_argument(
            "--full",
            action="store_true",
            help="Run all selected expensive dataset checks",
        )
    elif command == "score":
        parser.set_defaults(full=False)
    elif command in ("preflight", "run"):
        parser.set_defaults(full=False)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--model",
            help="Self-contained FALCON model directory or Hugging Face repository ID (loads its custom code)",
        )
        source.add_argument(
            "--checkpoint",
            help="Legacy training checkpoint; requires --detector-weights",
        )
        parser.add_argument(
            "--revision", help="Hugging Face model commit, tag or branch"
        )
        parser.add_argument(
            "--local-files-only",
            action="store_true",
            help="Use only local or cached Hugging Face files",
        )
        parser.add_argument(
            "--detector-weights",
            help="Detector checkpoint for legacy --checkpoint loading",
        )
        parser.add_argument("--run-dir", required=True)
        parser.add_argument("--config", help="Optional YAML config override")
        parser.add_argument("--vision-model")
        parser.add_argument("--language-model")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--max-new-tokens", type=int, default=256)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--retry-errors", action="store_true")
    return parser


def _write_report(report: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is not None:
        from .artifacts import atomic_json

        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"report output already exists; choose a new path: {output}"
            )
        atomic_json(output, report)
    print(text, end="")


def _require_final_stage3_predictions(predictions: Any) -> None:
    from .evaluation.scorer import require_final_stage3_provenance

    manifest = predictions.manifest
    require_final_stage3_provenance(
        manifest.get("provenance") if isinstance(manifest, Mapping) else None
    )


def _run_task_command(command: str, argv: Sequence[str]) -> None:
    from .artifacts import external_output
    from .data import open_dataset
    from .evaluation import (
        EvaluationError,
        load_prediction_set,
        run_predictions,
        score_dataset,
        validate_evaluation,
    )

    parser = _task_parser(command)
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            requested_output = Path(args.output).expanduser()
            if requested_output.exists() or requested_output.is_symlink():
                raise FileExistsError(
                    f"report output already exists; choose a new path: {requested_output}"
                )
        dataset = open_dataset(args.dataset, annotations=args.annotations)
        protected_roots_accessor = getattr(dataset, "protected_roots", None)
        protected_roots = (
            protected_roots_accessor() if callable(protected_roots_accessor) else ()
        )
        report_output = (
            external_output(
                args.output,
                dataset.root,
                protected_roots=protected_roots,
            )
            if args.output is not None
            else None
        )
        run_dir = None
        if command in ("preflight", "run"):
            if args.max_new_tokens < 1:
                raise ValueError("--max-new-tokens must be positive")
            if args.temperature != 0.0:
                raise ValueError(
                    "Stage 3 evaluation requires greedy decoding (--temperature 0)"
                )
            if args.retry_errors and not args.resume:
                raise ValueError("--retry-errors requires --resume")
            run_dir = external_output(
                args.run_dir,
                dataset.root,
                protected_roots=protected_roots,
            )
            if run_dir.exists() and not args.resume:
                raise FileExistsError(
                    f"run directory already exists; use --resume: {run_dir}"
                )
        partition = None
        if command != "validate" and args.split == "train" and not args.partition:
            raise ValueError(
                "training-split evaluation requires a held-out --partition"
            )
        if args.partition:
            if args.split != "train":
                parser.error("--partition requires --split train")
            from .partitions import load_partition

            partition = load_partition(
                dataset, args.partition, role=args.partition_role
            )
        predictions_name = getattr(args, "predictions", None)
        require_manifest = True
        predictions = (
            load_prediction_set(predictions_name, require_manifest=require_manifest)
            if predictions_name
            else None
        )
        if predictions is not None and report_output is not None:
            report_output = external_output(
                report_output,
                dataset.root,
                protected_roots=(*protected_roots, predictions.artifact_root),
            )
        if command == "score":
            if predictions is None:
                parser.error("--predictions is required for score")
            _require_final_stage3_predictions(predictions)
            report = score_dataset(
                dataset,
                predictions,
                split=args.split,
                tasks=args.tasks,
                protocol=args.protocol,
                require_manifest=require_manifest,
                partition=partition,
                partition_role=args.partition_role,
            )
        elif command == "validate":
            report = validate_evaluation(
                dataset,
                split=args.split,
                tasks=args.tasks,
                predictions=predictions,
                protocol=args.protocol,
                full=args.full,
                partition=partition,
                partition_role=args.partition_role,
            )
        else:
            preflight = validate_evaluation(
                dataset,
                split=args.split,
                tasks=args.tasks,
                protocol=args.protocol,
                # Check reference assets before constructing the model stack.
                full=True,
                partition=partition,
                partition_role=args.partition_role,
            )
            if not preflight["valid"]:
                raise EvaluationError(
                    "run preflight failed: " + "; ".join(preflight["errors"])
                )
            from .infer import verify_final_stage3_evaluation

            readiness = verify_final_stage3_evaluation(
                model=args.model,
                revision=args.revision,
                local_files_only=args.local_files_only,
                checkpoint=args.checkpoint,
                detector_weights=args.detector_weights,
                config=args.config,
                vision_model=args.vision_model,
                language_model=args.language_model,
            )
            if (
                readiness["metadata"].get("dataset_categories")
                != dataset.manifest["categories"]
            ):
                raise EvaluationError(
                    "dataset taxonomy differs from the Stage 3 training taxonomy"
                )
            if args.model is not None:
                model_identity = {"model": readiness["model"]}
                model_roots = (Path(readiness["model"]["path"]),)
            else:
                model_identity = {
                    key: readiness[key]
                    for key in ("checkpoint", "detector_weights", "model_locations")
                }
                model_roots = (
                    Path(readiness["checkpoint"]).parent,
                    Path(readiness["detector_weights"]).parent,
                    *(Path(value) for value in readiness["model_locations"].values()),
                )
            input_roots = (*model_roots, *protected_roots)
            run_dir = external_output(
                run_dir, dataset.root, protected_roots=input_roots
            )
            if report_output is not None:
                report_output = external_output(
                    report_output, dataset.root, protected_roots=input_roots
                )
                if report_output == run_dir or run_dir in report_output.parents:
                    raise ValueError(
                        "report output must be outside the prediction run directory"
                    )
            from .evaluation.scorer import expected_run_manifest, select_dataset_tasks

            execution = {"device": args.device}
            decoding = {"max_new_tokens": args.max_new_tokens, "temperature": 0.0}
            expected_provenance = {
                **model_identity,
                "stage": 3,
                "training_complete": True,
                "config": readiness["config"],
                "final_stage3": True,
                "seed": args.seed,
                "decoding": decoding,
                "execution": execution,
            }

            selected = select_dataset_tasks(
                dataset,
                args.split,
                preflight["families"],
                image_ids=partition.image_ids if partition is not None else None,
            )
            expected_manifest = expected_run_manifest(
                dataset,
                args.split,
                selected,
                preflight["families"],
                args.protocol,
                partition=partition,
                partition_role=args.partition_role,
            )
            resume_status = None
            if args.resume:
                from .evaluation.runner import validate_resume_run

                resume_status = validate_resume_run(
                    run_dir,
                    expected_manifest=expected_manifest,
                    expected_provenance=expected_provenance,
                )
            if command == "preflight":
                report = {
                    "schema_version": "falcon-stage3-preflight-v1",
                    "ready": True,
                    "model_initialized": False,
                    "evaluation_started": False,
                    "dataset_validation": preflight,
                    "checkpoint": {
                        key: value
                        for key, value in readiness.items()
                        if key not in {"metadata", "config"}
                    },
                    "run_dir": str(run_dir),
                    "input_manifest": expected_manifest,
                    "execution": execution,
                    "resume": resume_status,
                    "decoding": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": 0.0,
                        "seed": args.seed,
                    },
                    "runtime_status": "Model loading, GPU memory, and generation are not exercised.",
                }
                _write_report(report, report_output)
                return
            from transformers import set_seed

            from .infer import load_task_predictor

            set_seed(args.seed)
            category_ids = tuple(row["id"] for row in dataset.manifest["categories"])
            predictor, loaded_provenance = load_task_predictor(
                model=args.model,
                revision=args.revision,
                local_files_only=args.local_files_only,
                checkpoint=args.checkpoint,
                detector_weights=args.detector_weights,
                artifact_root=run_dir,
                category_ids=category_ids,
                config=args.config,
                vision_model=args.vision_model,
                language_model=args.language_model,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                final_stage3_readiness=readiness,
            )
            provenance = {
                **dict(loaded_provenance),
                "seed": args.seed,
                "decoding": decoding,
                "execution": execution,
            }
            if any(
                provenance.get(key) != value
                for key, value in expected_provenance.items()
            ):
                raise EvaluationError(
                    "loaded predictor differs from its Stage 3 readiness identity"
                )
            report = run_predictions(
                dataset,
                predictor,
                run_dir=run_dir,
                split=args.split,
                tasks=args.tasks,
                protocol=args.protocol,
                provenance=provenance,
                resume=args.resume,
                retry_errors=args.retry_errors,
                partition=partition,
                partition_role=args.partition_role,
            )
    except (EvaluationError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    _write_report(report, report_output)
    if command == "validate" and not report["valid"]:
        raise SystemExit(1)
    if command == "run" and (not report["complete"] or report["model_invalid"]):
        raise SystemExit(1)


def _print_command_help() -> None:
    parser = argparse.ArgumentParser(
        prog="falcon evaluate",
        description="Prepare and evaluate the final Stage 3 model on falcon-x tasks",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.add_parser(
        "validate", help="validate dataset references and saved predictions"
    )
    commands.add_parser(
        "preflight", help="verify data and final Stage 3 artifacts without inference"
    )
    commands.add_parser(
        "run", help="run model inference into a resumable prediction ledger"
    )
    commands.add_parser(
        "score", help="score a complete saved prediction ledger offline"
    )
    parser.print_help()


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch final Stage 3 evaluation and reference validation."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments in (["-h"], ["--help"]):
        _print_command_help()
        return
    if arguments[0] not in ("validate", "preflight", "run", "score"):
        argparse.ArgumentParser(prog="falcon evaluate").error(
            f"unknown command {arguments[0]!r}; choose validate, preflight, run or score"
        )
    _run_task_command(arguments[0], arguments[1:])


if __name__ == "__main__":
    main()
