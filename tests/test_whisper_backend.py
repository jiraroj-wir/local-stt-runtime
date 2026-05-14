import builtins
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.config import cpu_config, cuda_config
from app.segments import TranscriptionSegment
from app.whisper_backend import MISSING_FASTER_WHISPER_MESSAGE, transcribe_faster_whisper


@dataclass(frozen=True)
class FakeWhisperSegment:
    start: float
    end: float
    text: str


class FakeWhisperModel:
    init_calls: list[dict[str, object]] = []
    transcribe_calls: list[dict[str, object]] = []

    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.init_calls.append({"model_name": model_name, **kwargs})

    def transcribe(
        self,
        input_path: str,
        **kwargs: object,
    ) -> tuple[list[FakeWhisperSegment], object]:
        self.transcribe_calls.append({"input_path": input_path, **kwargs})
        return (
            [
                FakeWhisperSegment(start=0.0, end=1.25, text="Hello."),
                FakeWhisperSegment(start=1.25, end=2.5, text="World."),
            ],
            object(),
        )


def test_faster_whisper_constructs_model_with_cuda_config(monkeypatch) -> None:
    install_fake_faster_whisper(monkeypatch)
    config = cuda_config()

    segments = transcribe_faster_whisper(Path("lecture.m4a"), config)

    assert FakeWhisperModel.init_calls == [
        {
            "model_name": "Systran/faster-whisper-large-v3",
            "device": "cuda",
            "compute_type": "int8",
        }
    ]
    assert segments == [
        TranscriptionSegment(start=0.0, end=1.25, text="Hello."),
        TranscriptionSegment(start=1.25, end=2.5, text="World."),
    ]


def test_faster_whisper_transcribe_receives_expected_options(monkeypatch) -> None:
    install_fake_faster_whisper(monkeypatch)

    transcribe_faster_whisper(Path("lecture.m4a"), cuda_config())

    assert FakeWhisperModel.transcribe_calls == [
        {
            "input_path": "lecture.m4a",
            "language": "en",
            "beam_size": 5,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }
    ]


def test_faster_whisper_cpu_model_omits_auto_cpu_threads(monkeypatch) -> None:
    install_fake_faster_whisper(monkeypatch)

    transcribe_faster_whisper(Path("lecture.m4a"), cpu_config())

    assert FakeWhisperModel.init_calls == [
        {
            "model_name": "Systran/faster-whisper-medium.en",
            "device": "cpu",
            "compute_type": "int8",
        }
    ]


def test_missing_faster_whisper_raises_clear_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "faster_whisper":
            raise ImportError("missing faster_whisper")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match=MISSING_FASTER_WHISPER_MESSAGE):
        transcribe_faster_whisper(Path("lecture.m4a"), cpu_config())


def install_fake_faster_whisper(monkeypatch) -> None:
    FakeWhisperModel.init_calls = []
    FakeWhisperModel.transcribe_calls = []
    module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
