"""Internal Python transcription runner."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from app.audio import validate_supported_audio_path
from app.backends import FAKE_BACKEND_NAME, SUPPORTED_BACKENDS, transcribe_with_backend
from app.config import get_device_config
from app.errors import FailureReport
from app.output import write_all_outputs
from app.ui import format_failure_summary, format_finish_summary, format_start_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    stem = args.stem or input_path.stem

    try:
        validate_supported_audio_path(input_path)
    except ValueError as exc:
        print(
            format_failure_summary(FailureReport(stage="input validation", error=str(exc))),
            file=sys.stderr,
        )
        return 2

    config = get_device_config(args.device)
    mode = args.mode or args.device
    print(
        format_start_summary(
            input_path=input_path,
            mode=mode,
            selected_device=config.device,
            model=config.model,
        )
    )

    start_time = time.monotonic()
    try:
        segments = transcribe_with_backend(args.backend, input_path, config)
    except RuntimeError as exc:
        print(
            format_failure_summary(FailureReport(stage="transcription", error=str(exc))),
            file=sys.stderr,
        )
        return 1
    elapsed_seconds = time.monotonic() - start_time
    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "device": config.device,
        "model": config.model,
        "backend": args.backend,
    }
    log_lines = [
        f"input_path={input_path}",
        f"output_dir={output_dir}",
        f"device={config.device}",
        f"model={config.model}",
        f"backend={args.backend}",
        f"segments={len(segments)}",
    ]

    paths = write_all_outputs(
        segments=segments,
        output_dir=output_dir,
        stem=stem,
        metadata=metadata,
        log_lines=log_lines,
    )

    print(
        format_finish_summary(
            duration_seconds=_transcribed_duration_seconds(segments),
            elapsed_seconds=elapsed_seconds,
            device=config.device,
            model=config.model,
            files=paths,
        )
    )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local transcription inside the runtime.")
    parser.add_argument("input_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--mode", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--stem")
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=FAKE_BACKEND_NAME)
    return parser


def _transcribed_duration_seconds(segments) -> float:
    if not segments:
        return 0.0
    return max(segment.end for segment in segments)


if __name__ == "__main__":
    raise SystemExit(main())
