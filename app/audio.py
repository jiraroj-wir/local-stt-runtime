"""Audio metadata and preprocessing helper functions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

SUPPORTED_AUDIO_EXTENSIONS = frozenset({".aac", ".m4a", ".mp3", ".wav", ".flac", ".ogg"})


@dataclass(frozen=True)
class AudioMetadata:
    duration_seconds: float
    format_name: str | None
    sample_rate_hz: int | None
    channels: int | None
    codec_name: str | None


def is_supported_audio_path(path: str | PathLike[str]) -> bool:
    """Return whether a path has a supported audio extension."""
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def validate_supported_audio_path(path: str | PathLike[str]) -> None:
    """Raise ValueError if a path does not have a supported audio extension."""
    if not is_supported_audio_path(path):
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio file extension for {path!s}; supported: {supported}")


def build_ffprobe_command(input_path: Path) -> list[str]:
    """Build an ffprobe command for JSON audio metadata."""
    return [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]


def parse_ffprobe_metadata(ffprobe_json: dict[str, Any]) -> AudioMetadata:
    """Parse ffprobe JSON output into AudioMetadata."""
    format_section = ffprobe_json.get("format")
    if not isinstance(format_section, dict):
        format_section = {}

    audio_stream = _first_audio_stream(ffprobe_json.get("streams"))
    duration_seconds = _parse_required_duration(format_section, audio_stream)

    return AudioMetadata(
        duration_seconds=duration_seconds,
        format_name=_optional_str(format_section.get("format_name")),
        sample_rate_hz=_optional_int(audio_stream.get("sample_rate") if audio_stream else None),
        channels=_optional_int(audio_stream.get("channels") if audio_stream else None),
        codec_name=_optional_str(audio_stream.get("codec_name") if audio_stream else None),
    )


def build_preprocess_command(input_path: Path, output_wav_path: Path) -> list[str]:
    """Build an ffmpeg command to preprocess audio to 16 kHz mono WAV."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav_path),
    ]


def _first_audio_stream(streams: Any) -> dict[str, Any] | None:
    if not isinstance(streams, list):
        return None

    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            return stream
    return None


def _parse_required_duration(
    format_section: dict[str, Any],
    audio_stream: dict[str, Any] | None,
) -> float:
    if _has_value(format_section.get("duration")):
        return _required_float(format_section["duration"], "format.duration")

    if audio_stream and _has_value(audio_stream.get("duration")):
        return _required_float(audio_stream["duration"], "audio stream duration")

    raise ValueError("ffprobe metadata does not include a usable duration")


def _required_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ffprobe duration in {field_name}: {value!r}") from exc

    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"Invalid ffprobe duration in {field_name}: {value!r}")
    return parsed


def _optional_int(value: Any) -> int | None:
    if not _has_value(value):
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    return parsed


def _optional_str(value: Any) -> str | None:
    if not _has_value(value):
        return None
    return str(value)


def _has_value(value: Any) -> bool:
    return value is not None and value != ""
