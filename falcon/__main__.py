"""Command-line entry point for FALCON."""

from __future__ import annotations

import argparse
import importlib
import sys


def main() -> None:
    commands = {
        "prepare": "Prepare dataset tasks and training partitions",
        "detector": "Train the Stage 1 detector",
        "train": "Train Stage 2 adapters or Stage 3 LoRA",
        "export": "Package the final model for Hugging Face",
        "infer": "Run the model on one image",
        "evaluate": "Validate, run, or score task evaluations",
    }
    parser = argparse.ArgumentParser(
        prog="falcon", description="FALCON training and evaluation"
    )
    parser.add_argument("command", choices=commands, help="command to run")
    parser.epilog = "\n".join(
        f"  {name}: {description}" for name, description in commands.items()
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return
    command = parser.parse_args(args[:1]).command
    sys.argv = [f"falcon {command}", *args[1:]]
    importlib.import_module(f"falcon.{command}").main()


if __name__ == "__main__":
    main()
