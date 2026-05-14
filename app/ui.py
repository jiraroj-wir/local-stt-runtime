"""Plain terminal UI formatting helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from app.errors import FailureReport

SEPARATOR = "────────────────────────────────────────"


def format_start_summary(
    *,
    input_path: Path,
    mode: str,
    selected_device: str,
    model: str,
    duration_seconds: float | None = None,
    audio: str | None = None,
) -> str:
    return "\n".join(
        [
            SEPARATOR,
            "LOCAL-STT-RUNTIME",
            "",
            "Input",
            f"  File       : {input_path}",
            f"  Duration   : {_duration_or_unknown(duration_seconds)}",
            f"  Audio      : {audio or 'not inspected'}",
            "",
            "Runtime",
            f"  Mode       : {mode}",
            f"  Selected   : {selected_device.upper()}",
            f"  Model      : {model}",
            SEPARATOR,
        ]
    )


def format_runtime_fallback(reason: str) -> str:
    return "\n".join(
        [
            SEPARATOR,
            "Runtime fallback",
            f"  Reason     : {reason}",
            "  Continuing : CPU",
            SEPARATOR,
        ]
    )


def format_progress_line(
    *,
    current_seconds: float,
    total_seconds: float,
    elapsed_seconds: float,
    device: str,
    model: str,
) -> str:
    percent = _percent(current_seconds, total_seconds)
    return (
        f"[{format_clock(current_seconds)} / {format_clock(total_seconds)}] "
        f"{percent}% | elapsed {format_clock(elapsed_seconds)} | "
        f"{device.upper()} {_short_model_name(model)}"
    )


def format_finish_summary(
    *,
    duration_seconds: float,
    elapsed_seconds: float,
    device: str,
    model: str,
    files: Mapping[str, Path],
) -> str:
    return "\n".join(
        [
            SEPARATOR,
            "Transcription complete",
            "",
            "Finish",
            f"  Duration   : {format_clock(duration_seconds)}",
            f"  Elapsed    : {format_clock(elapsed_seconds)}",
            f"  Realtime   : {_realtime_factor(duration_seconds, elapsed_seconds)}x",
            f"  Device     : {device.upper()}",
            f"  Model      : {model}",
            "",
            "Files written",
            *[f"  {label:<10} : {path}" for label, path in files.items()],
            SEPARATOR,
        ]
    )


def format_failure_summary(report: FailureReport) -> str:
    lines = [
        SEPARATOR,
        "Transcription failed",
        "",
        "Failure",
        f"  Stage      : {report.stage}",
        f"  Error      : {report.error}",
    ]
    if report.log_path is not None:
        lines.append(f"  Log        : {report.log_path}")
    lines.append(SEPARATOR)
    return "\n".join(lines)


def format_clock(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, timestamp_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{timestamp_seconds:02d}"


def _duration_or_unknown(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return format_clock(seconds)


def _percent(current_seconds: float, total_seconds: float) -> int:
    if total_seconds <= 0:
        return 0
    return max(0, min(100, int(current_seconds / total_seconds * 100)))


def _realtime_factor(duration_seconds: float, elapsed_seconds: float) -> str:
    if elapsed_seconds <= 0:
        return "0.00"
    return f"{duration_seconds / elapsed_seconds:.2f}"


def _short_model_name(model: str) -> str:
    return model.rsplit("/", maxsplit=1)[-1].removeprefix("faster-whisper-")
