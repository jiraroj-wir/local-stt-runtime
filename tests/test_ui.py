from pathlib import Path

from app.errors import FailureReport
from app.ui import (
    format_failure_summary,
    format_finish_summary,
    format_progress_line,
    format_runtime_fallback,
    format_start_summary,
)


def test_start_summary_includes_expected_fields() -> None:
    summary = format_start_summary(
        input_path=Path("audio/lecture.m4a"),
        duration_seconds=8022,
        audio="AAC, 44.1 kHz, stereo",
        mode="auto",
        selected_device="cuda",
        model="Systran/faster-whisper-large-v3",
    )

    assert "LOCAL-STT-RUNTIME" in summary
    assert "File       : audio/lecture.m4a" in summary
    assert "Duration   : 02:13:42" in summary
    assert "Audio      : AAC, 44.1 kHz, stereo" in summary
    assert "Mode       : auto" in summary
    assert "Selected   : CUDA" in summary
    assert "Model      : Systran/faster-whisper-large-v3" in summary


def test_runtime_fallback_summary_includes_reason_and_cpu_continuation() -> None:
    summary = format_runtime_fallback("CUDA/CDI GPU check failed")

    assert "Runtime fallback" in summary
    assert "Reason     : CUDA/CDI GPU check failed" in summary
    assert "Continuing : CPU" in summary


def test_progress_line_formats_percent_timestamps_and_runtime_label() -> None:
    line = format_progress_line(
        current_seconds=2538,
        total_seconds=8022,
        elapsed_seconds=1690,
        device="cuda",
        model="Systran/faster-whisper-large-v3",
    )

    assert line == "[00:42:18 / 02:13:42] 31% | elapsed 00:28:10 | CUDA large-v3"


def test_finish_summary_includes_elapsed_realtime_and_output_files() -> None:
    summary = format_finish_summary(
        duration_seconds=120,
        elapsed_seconds=30,
        device="cpu",
        model="Systran/faster-whisper-medium.en",
        files={
            "txt": Path("lecture.txt"),
            "srt": Path("lecture.srt"),
        },
    )

    assert "Transcription complete" in summary
    assert "Duration   : 00:02:00" in summary
    assert "Elapsed    : 00:00:30" in summary
    assert "Realtime   : 4.00x" in summary
    assert "Device     : CPU" in summary
    assert "Model      : Systran/faster-whisper-medium.en" in summary
    assert "txt        : lecture.txt" in summary
    assert "srt        : lecture.srt" in summary


def test_failure_summary_includes_stage_error_and_log_path() -> None:
    summary = format_failure_summary(
        FailureReport(
            stage="transcription",
            error="backend failed",
            log_path=Path("run.log"),
        )
    )

    assert "Transcription failed" in summary
    assert "Stage      : transcription" in summary
    assert "Error      : backend failed" in summary
    assert "Log        : run.log" in summary
