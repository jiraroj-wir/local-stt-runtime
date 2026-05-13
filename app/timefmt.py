"""Timestamp formatting helpers."""

from __future__ import annotations

import math


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    return _format_timestamp(seconds, millisecond_separator=",")


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp."""
    return _format_timestamp(seconds, millisecond_separator=".")


def _format_timestamp(seconds: float, *, millisecond_separator: str) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be a non-negative finite number")

    total_milliseconds = int(seconds * 1000 + 0.5)
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    total_minutes, timestamp_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)

    return (
        f"{hours:02d}:{minutes:02d}:{timestamp_seconds:02d}"
        f"{millisecond_separator}{milliseconds:03d}"
    )
