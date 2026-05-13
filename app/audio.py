from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = {".aac", ".m4a", ".mp3", ".wav", ".flac", ".ogg"}


@dataclass(frozen=True)
class AudioMetadata:
    duration_seconds: float
    format_name: str | None
    sample_rate_hz: int | None
    channels: int | None
    codec_name: str | None


def is_supported_audio_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def validate_supported_audio_path(path: str | Path) -> None:
    if not is_supported_audio_path(path):
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio extension for '{path}'. Supported extensions: {supported}"
        )


def build_ffprobe_command(input_path: Path) -> list[str]:
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


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_ffprobe_metadata(ffprobe_json: dict) -> AudioMetadata:
    format_data = ffprobe_json.get("format")
    if not isinstance(format_data, dict):
        format_data = {}

    streams = ffprobe_json.get("streams")
    if not isinstance(streams, list):
        streams = []

    audio_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        None,
    )

    duration_seconds = _parse_float(format_data.get("duration"))
    if duration_seconds is None and isinstance(audio_stream, dict):
        duration_seconds = _parse_float(audio_stream.get("duration"))

    if duration_seconds is None:
        raise ValueError("Unable to determine audio duration from ffprobe metadata")

    return AudioMetadata(
        duration_seconds=duration_seconds,
        format_name=format_data.get("format_name")
        if isinstance(format_data.get("format_name"), str)
        else None,
        sample_rate_hz=_parse_int(audio_stream.get("sample_rate"))
        if isinstance(audio_stream, dict)
        else None,
        channels=_parse_int(audio_stream.get("channels"))
        if isinstance(audio_stream, dict)
        else None,
        codec_name=audio_stream.get("codec_name")
        if isinstance(audio_stream, dict)
        and isinstance(audio_stream.get("codec_name"), str)
        else None,
    )


def build_preprocess_command(input_path: Path, output_wav_path: Path) -> list[str]:
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
