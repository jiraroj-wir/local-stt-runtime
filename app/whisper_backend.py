"""faster-whisper transcription backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import TranscriptionConfig
from app.segments import TranscriptionSegment

MISSING_FASTER_WHISPER_MESSAGE = (
    "faster-whisper is not installed; run inside the container or install runtime deps."
)


class FasterWhisperSession:
    """Reusable faster-whisper model session for one transcription run."""

    def __init__(self, config: TranscriptionConfig) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(MISSING_FASTER_WHISPER_MESSAGE) from exc

        self.config = config
        self.model = WhisperModel(
            config.model,
            **_model_kwargs(config),
        )

    def transcribe(self, input_path: Path) -> list[TranscriptionSegment]:
        segments, _info = self.model.transcribe(
            str(input_path),
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            condition_on_previous_text=self.config.condition_on_previous_text,
            word_timestamps=(
                self.config.word_timestamps if self.config.word_timestamps is not None else False
            ),
        )

        return [
            TranscriptionSegment(start=segment.start, end=segment.end, text=segment.text)
            for segment in segments
        ]


def create_faster_whisper_session(config: TranscriptionConfig) -> FasterWhisperSession:
    return FasterWhisperSession(config)


def transcribe_faster_whisper(
    input_path: Path,
    config: TranscriptionConfig,
) -> list[TranscriptionSegment]:
    return create_faster_whisper_session(config).transcribe(input_path)


def _model_kwargs(config: TranscriptionConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "device": config.device,
        "compute_type": config.compute_type,
    }
    if config.device == "cpu" and config.cpu_threads not in (None, "auto"):
        kwargs["cpu_threads"] = config.cpu_threads
    return kwargs
