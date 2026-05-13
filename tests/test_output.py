import json

from app.output import (
    write_all_outputs,
    write_json,
    write_metadata,
    write_run_log,
    write_srt,
    write_txt,
)
from app.segments import TranscriptionSegment


def test_write_txt_strips_whitespace_and_skips_empty_text(tmp_path) -> None:
    output_path = tmp_path / "lecture.txt"
    segments = [
        TranscriptionSegment(start=0.0, end=1.0, text="  Hello everyone.  "),
        TranscriptionSegment(start=1.0, end=2.0, text="   "),
        TranscriptionSegment(start=2.0, end=3.0, text="Welcome back."),
    ]

    write_txt(segments, output_path)

    assert output_path.read_text(encoding="utf-8") == "Hello everyone.\nWelcome back.\n"


def test_write_srt_numbers_cues_formats_timestamps_and_skips_empty_text(tmp_path) -> None:
    output_path = tmp_path / "lecture.srt"
    segments = [
        TranscriptionSegment(start=0.0, end=2.5, text="  Hello everyone.  "),
        TranscriptionSegment(start=2.5, end=3.0, text=""),
        TranscriptionSegment(start=3.0, end=4.25, text="Welcome back."),
    ]

    write_srt(segments, output_path)

    assert output_path.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:02,500\n"
        "Hello everyone.\n"
        "\n"
        "2\n"
        "00:00:03,000 --> 00:00:04,250\n"
        "Welcome back.\n"
    )


def test_write_json_writes_segments_and_metadata(tmp_path) -> None:
    output_path = tmp_path / "lecture.json"
    segments = [TranscriptionSegment(start=0.0, end=2.5, text="Hello everyone.")]
    metadata = {"model": "fake-model", "language": "en"}

    write_json(segments, output_path, metadata=metadata)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(data) == {"segments", "metadata"}
    assert data["segments"] == [{"start": 0.0, "end": 2.5, "text": "Hello everyone."}]
    assert data["metadata"] == metadata


def test_write_json_defaults_metadata_to_empty_object(tmp_path) -> None:
    output_path = tmp_path / "lecture.json"

    write_json([], output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "segments": [],
        "metadata": {},
    }


def test_write_metadata_writes_json_file_with_expected_fields(tmp_path) -> None:
    output_path = tmp_path / "metadata.json"
    metadata = {"language": "en", "duration_seconds": 2.5}

    write_metadata(metadata, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == metadata


def test_write_run_log_writes_one_line_per_input_line_with_trailing_newline(tmp_path) -> None:
    output_path = tmp_path / "run.log"

    write_run_log(["starting", "finished"], output_path)

    assert output_path.read_text(encoding="utf-8") == "starting\nfinished\n"


def test_write_all_outputs_creates_expected_files_and_returns_paths(tmp_path) -> None:
    output_dir = tmp_path / "transcripts"
    segments = [TranscriptionSegment(start=0.0, end=2.5, text="Hello everyone.")]
    metadata = {"language": "en"}

    paths = write_all_outputs(
        segments=segments,
        output_dir=output_dir,
        stem="lecture",
        metadata=metadata,
        log_lines=["done"],
    )

    assert set(paths) == {"txt", "srt", "json", "metadata", "log"}
    assert paths == {
        "txt": output_dir / "lecture.txt",
        "srt": output_dir / "lecture.srt",
        "json": output_dir / "lecture.json",
        "metadata": output_dir / "metadata.json",
        "log": output_dir / "run.log",
    }
    assert all(path.exists() for path in paths.values())
