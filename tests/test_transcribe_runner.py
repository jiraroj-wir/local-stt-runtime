import json
import sys
from pathlib import Path

from app.audio import AudioInspection, AudioMetadata, ChunkPlanItem, VolumeStats
from app.segments import TranscriptionSegment
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
    assert "input_duration_seconds" in metadata
    assert "transcribed_duration_seconds" in metadata
    assert "elapsed_seconds" in metadata
    assert "realtime_factor" in metadata
    assert "audio" in metadata
    assert metadata["chunking_enabled"] is True
    assert metadata["chunk_minutes"] == 20
    assert metadata["chunks_count"] == 0
    assert metadata["preprocess_mode"] == "auto"
    assert metadata["preprocessing_applied"] is False


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

    class FakeSession:
        def transcribe(self, backend_input_path):
            assert backend_input_path == input_path
            return []

    def fake_create_backend_session(backend, config):
        assert backend == "faster-whisper"
        assert config.device == "cpu"
        return FakeSession()

    monkeypatch.setattr("app.transcribe.create_backend_session", fake_create_backend_session)

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

    def fake_create_backend_session(_backend, _config):
        raise RuntimeError(
            "faster-whisper is not installed; run inside the container or install runtime deps."
        )

    monkeypatch.setattr("app.transcribe.create_backend_session", fake_create_backend_session)

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


def test_runner_uses_ffprobe_duration_in_start_finish_and_metadata(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        "app.transcribe.inspect_audio",
        lambda _path: AudioInspection(
            AudioMetadata(
                duration_seconds=1200.0,
                codec_name="aac",
                sample_rate=44100,
                channels=2,
                bit_rate=128000,
                format_name="mov,mp4",
            )
        ),
    )
    monkeypatch.setattr(
        "app.transcribe.detect_volume",
        lambda _path: VolumeStats(mean_volume_db=-20, max_volume_db=-3),
    )

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
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Duration   : 00:20:00" in captured.out
    assert "Audio      : aac, 44.1 kHz, stereo, 128 kbps" in captured.out
    assert "Audio      : 00:20:00" in captured.out
    assert metadata["input_duration_seconds"] == 1200.0
    assert metadata["audio"]["codec_name"] == "aac"
    assert metadata["mean_volume_db"] == -20
    assert metadata["max_volume_db"] == -3


def test_runner_chunks_long_audio_and_offsets_segments(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        "app.transcribe.inspect_audio",
        lambda _path: AudioInspection(
            AudioMetadata(
                duration_seconds=45 * 60,
                codec_name="aac",
                sample_rate=44100,
                channels=1,
                bit_rate=None,
                format_name="mov,mp4",
            )
        ),
    )
    monkeypatch.setattr(
        "app.transcribe.detect_volume",
        lambda _path: VolumeStats(mean_volume_db=-20, max_volume_db=-3),
    )
    created_chunks: list[Path] = []

    def fake_create_chunk(_source_path: Path, chunk: ChunkPlanItem) -> None:
        created_chunks.append(chunk.path)

    class FakeSession:
        def transcribe(self, backend_input_path: Path) -> list[TranscriptionSegment]:
            return [
                TranscriptionSegment(
                    start=5.0,
                    end=10.0,
                    text=f"text from {backend_input_path.name}",
                )
            ]

    monkeypatch.setattr("app.transcribe.create_chunk", fake_create_chunk)
    monkeypatch.setattr(
        "app.transcribe.create_backend_session",
        lambda _backend, _config: FakeSession(),
    )

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

    data = json.loads((output_dir / "lecture.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(created_chunks) == 3
    assert data["segments"][0]["start"] == 5.0
    assert data["segments"][1]["start"] == 1205.0
    assert data["segments"][2]["start"] == 2405.0
    assert metadata["chunks_count"] == 3
    assert metadata["chunks"][1]["offset_seconds"] == 1200.0


def test_runner_no_chunk_disables_long_audio_chunking(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        "app.transcribe.inspect_audio",
        lambda _path: AudioInspection(
            AudioMetadata(
                duration_seconds=45 * 60,
                codec_name=None,
                sample_rate=None,
                channels=None,
                bit_rate=None,
                format_name=None,
            )
        ),
    )
    monkeypatch.setattr(
        "app.transcribe.detect_volume",
        lambda _path: VolumeStats(mean_volume_db=-20, max_volume_db=-3),
    )

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--backend",
            "fake",
            "--no-chunk",
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metadata["chunking_enabled"] is False
    assert metadata["chunks_count"] == 0


def test_runner_preprocess_auto_normalizes_quiet_audio(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    normalized_paths: list[Path] = []

    monkeypatch.setattr(
        "app.transcribe.detect_volume",
        lambda _path: VolumeStats(mean_volume_db=-36, max_volume_db=-10),
    )

    def fake_normalize(_input_path: Path, output_wav_path: Path) -> None:
        normalized_paths.append(output_wav_path)

    monkeypatch.setattr("app.transcribe.normalize_audio", fake_normalize)

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

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(normalized_paths) == 1
    assert metadata["preprocessing_applied"] is True
    assert metadata["preprocessed_path"] == "lecture.normalized.wav"
