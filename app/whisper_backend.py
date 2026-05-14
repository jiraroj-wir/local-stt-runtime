"""faster-whisper transcription backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import TranscriptionConfig
from app.segments import TranscriptionSegment

MISSING_FASTER_WHISPER_MESSAGE = (
    "faster-whisper is not installed; run inside the container or install runtime deps."
)


def transcribe_faster_whisper(
    input_path: Path,
    config: TranscriptionConfig,
) -> list[TranscriptionSegment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(MISSING_FASTER_WHISPER_MESSAGE) from exc

    model = WhisperModel(
        config.model,
        **_model_kwargs(config),
    )
    segments, _info = model.transcribe(
        str(input_path),
        language=config.language,
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
        condition_on_previous_text=config.condition_on_previous_text,
        word_timestamps=config.word_timestamps if config.word_timestamps is not None else False,
    )

    return [
        TranscriptionSegment(start=segment.start, end=segment.end, text=segment.text)
        for segment in segments
    ]


def _model_kwargs(config: TranscriptionConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "device": config.device,
        "compute_type": config.compute_type,
    }
    if config.device == "cpu" and config.cpu_threads not in (None, "auto"):
        kwargs["cpu_threads"] = config.cpu_threads
    return kwargs
