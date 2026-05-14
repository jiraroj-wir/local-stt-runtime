import json
import sys

from app.transcribe import main


def test_fake_runner_writes_outputs_with_default_stem(tmp_path, capsys) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "lecture"

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--backend",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Transcription complete" in captured.out
    assert (output_dir / "lecture.txt").read_text(encoding="utf-8") == (
        "Hello everyone.\n"
        "Today we discuss local transcription.\n"
    )
    assert (output_dir / "lecture.srt").exists()
    assert (output_dir / "lecture.json").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "run.log").exists()


def test_fake_runner_uses_custom_stem(tmp_path) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "transcripts"

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--stem",
            "custom-name",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "custom-name.txt").exists()
    assert (output_dir / "custom-name.srt").exists()
    assert (output_dir / "custom-name.json").exists()


def test_fake_runner_json_output_contains_fake_segments(tmp_path) -> None:
    input_path = tmp_path / "lecture.wav"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"

    exit_code = main([str(input_path), "--output-dir", str(output_dir), "--device", "cuda"])

    data = json.loads((output_dir / "lecture.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert data["segments"] == [
        {"start": 0.0, "end": 2.5, "text": "Hello everyone."},
        {
            "start": 2.5,
            "end": 5.0,
            "text": "Today we discuss local transcription.",
        },
    ]


def test_fake_runner_metadata_contains_runtime_contract_fields(tmp_path) -> None:
    input_path = tmp_path / "lecture.mp3"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda",
            "--backend",
            "fake",
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metadata["input_path"] == str(input_path)
    assert metadata["output_dir"] == str(output_dir)
    assert metadata["device"] == "cuda"
    assert metadata["model"] == "Systran/faster-whisper-large-v3"
    assert metadata["backend"] == "fake"


def test_invalid_extension_exits_nonzero_and_writes_no_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "notes.txt"
    input_path.write_text("not audio", encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = main([str(input_path), "--output-dir", str(output_dir), "--device", "cpu"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Unsupported audio file extension" in captured.err
    assert not output_dir.exists()


def test_runner_does_not_import_faster_whisper(tmp_path) -> None:
    input_path = tmp_path / "lecture.flac"
    input_path.write_bytes(b"fake audio")

    exit_code = main([str(input_path), "--output-dir", str(tmp_path / "out"), "--device", "cpu"])

    assert exit_code == 0
    assert "faster_whisper" not in sys.modules


def test_runner_accepts_faster_whisper_backend_when_available(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"

    def fake_transcribe_with_backend(backend, backend_input_path, config):
        assert backend == "faster-whisper"
        assert backend_input_path == input_path
        assert config.device == "cpu"
        return []

    monkeypatch.setattr("app.transcribe.transcribe_with_backend", fake_transcribe_with_backend)

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--backend",
            "faster-whisper",
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metadata["backend"] == "faster-whisper"


def test_runner_reports_missing_faster_whisper_error(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")

    def fake_transcribe_with_backend(_backend, _backend_input_path, _config):
        raise RuntimeError(
            "faster-whisper is not installed; run inside the container or install runtime deps."
        )

    monkeypatch.setattr("app.transcribe.transcribe_with_backend", fake_transcribe_with_backend)

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--backend",
            "faster-whisper",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "faster-whisper is not installed" in captured.err
