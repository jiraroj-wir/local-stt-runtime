import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.audio import AudioInspection, AudioMetadata, ChunkPlanItem, VolumeStats
from app.segments import TranscriptionSegment
from app.transcribe import _build_parser, main


@pytest.fixture(autouse=True)
def fake_source_audio_creation(monkeypatch) -> None:
    def fake_create_source_audio(
        _input_path: Path,
        output_wav_path: Path,
        *,
        normalize: bool,
    ) -> None:
        _ = normalize
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        output_wav_path.write_bytes(b"fake source wav")

    monkeypatch.setattr("app.transcribe.create_source_audio", fake_create_source_audio)


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
    assert "source_duration_seconds" in metadata
    assert "transcribed_duration_seconds" in metadata
    assert "elapsed_seconds" in metadata
    assert "realtime_factor" in metadata
    assert "audio" in metadata
    assert "original_audio" in metadata
    assert "source_audio" in metadata
    assert metadata["canonical_source_path"] == "lecture.source.wav"
    assert metadata["source_path"] == "lecture.source.wav"
    assert metadata["chunking_enabled"] is True
    assert metadata["chunk_minutes"] == 20
    assert metadata["chunks_count"] == 0
    assert metadata["preprocess_mode"] == "auto"
    assert metadata["preprocessing_applied"] is False
    assert metadata["audio_preparation_applied"] is True


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
            assert backend_input_path == output_dir / ".work" / "source" / "lecture.source.wav"
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
    assert metadata["source_duration_seconds"] == 1200.0
    assert metadata["audio"]["codec_name"] == "aac"
    assert metadata["original_audio"]["codec_name"] == "aac"
    assert metadata["source_audio"]["codec_name"] == "aac"
    assert metadata["mean_volume_db"] == -20
    assert metadata["max_volume_db"] == -3


def test_runner_chunks_long_audio_and_offsets_segments(tmp_path, monkeypatch, capsys) -> None:
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
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Chunk 1/3 offset 00:00:00 duration 00:20:00" in captured.out
    assert len(created_chunks) == 3
    assert data["segments"][0]["start"] == 5.0
    assert data["segments"][1]["start"] == 1205.0
    assert data["segments"][2]["start"] == 2405.0
    assert metadata["chunks_count"] == 3
    assert metadata["chunks"][1]["offset_seconds"] == 1200.0


def test_runner_parser_accepts_isolate_chunks() -> None:
    args = _build_parser().parse_args(
        [
            "lecture.wav",
            "--output-dir",
            "out",
            "--device",
            "cuda",
            "--isolate-chunks",
        ]
    )

    assert args.isolate_chunks is True


def test_runner_isolate_chunks_has_no_effect_without_planned_chunks(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    transcribed_paths: list[Path] = []

    monkeypatch.setattr(
        "app.transcribe.inspect_audio",
        lambda _path: AudioInspection(
            AudioMetadata(
                duration_seconds=10 * 60,
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

    class FakeSession:
        def transcribe(self, backend_input_path: Path) -> list[TranscriptionSegment]:
            transcribed_paths.append(backend_input_path)
            return []

    def fail_subprocess_run(*_args, **_kwargs):
        raise AssertionError("isolated child process should not run without chunks")

    monkeypatch.setattr(
        "app.transcribe.create_backend_session",
        lambda _backend, _config: FakeSession(),
    )
    monkeypatch.setattr("app.transcribe.subprocess.run", fail_subprocess_run)

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda",
            "--backend",
            "fake",
            "--isolate-chunks",
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert transcribed_paths == [output_dir / ".work" / "source" / "lecture.source.wav"]
    assert metadata["chunks_count"] == 0
    assert metadata["isolated_chunks"] is False
    assert metadata["child_processes_count"] == 0
    assert metadata["chunk_output_dirs"] == []


def test_runner_isolated_chunks_spawn_children_and_merge_offsets(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    commands: list[list[str]] = []

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
    monkeypatch.setattr("app.transcribe.create_chunk", lambda _source, _chunk: None)

    def fake_run(command, **_kwargs):
        commands.append(command)
        child_output_dir = Path(command[command.index("--output-dir") + 1])
        child_stem = command[command.index("--stem") + 1]
        child_output_dir.mkdir(parents=True, exist_ok=True)
        (child_output_dir / f"{child_stem}.json").write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start": 5.0,
                            "end": 10.0,
                            "text": f"text from {Path(command[3]).name}",
                        }
                    ],
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="child ok", stderr="")

    monkeypatch.setattr("app.transcribe.subprocess.run", fake_run)

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda",
            "--mode",
            "gpu",
            "--backend",
            "fake",
            "--isolate-chunks",
        ]
    )

    captured = capsys.readouterr()
    data = json.loads((output_dir / "lecture.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    run_log = (output_dir / "run.log").read_text(encoding="utf-8")

    assert exit_code == 0
    assert len(commands) == 3
    assert "Chunk 1/3 offset 00:00:00 duration 00:20:00 isolated" in captured.out
    first_command = commands[0]
    assert first_command[:3] == [sys.executable, "-m", "app.transcribe"]
    assert first_command[3].endswith("lecture.chunk-0001.wav")
    assert first_command[first_command.index("--output-dir") + 1] == str(
        output_dir / ".work" / "chunk_outputs" / "chunk-0001"
    )
    assert first_command[first_command.index("--device") + 1] == "cuda"
    assert first_command[first_command.index("--backend") + 1] == "fake"
    assert "--no-chunk" in first_command
    assert first_command[first_command.index("--preprocess") + 1] == "off"
    assert first_command[first_command.index("--stem") + 1] == "lecture.chunk-0001"
    assert first_command[first_command.index("--mode") + 1] == "gpu"
    assert "--isolate-chunks" not in first_command
    assert data["segments"][0]["start"] == 5.0
    assert data["segments"][1]["start"] == 1205.0
    assert data["segments"][2]["start"] == 2405.0
    assert metadata["isolated_chunks"] is True
    assert metadata["child_processes_count"] == 3
    assert metadata["chunk_output_dirs"] == [
        ".work/chunk_outputs/chunk-0001",
        ".work/chunk_outputs/chunk-0002",
        ".work/chunk_outputs/chunk-0003",
    ]
    assert "isolated_chunks=true" in run_log
    assert "child_processes_count=3" in run_log


def test_runner_isolated_chunk_failure_returns_transcription_error(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    commands: list[list[str]] = []

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
    monkeypatch.setattr("app.transcribe.create_chunk", lambda _source, _chunk: None)

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            7,
            stdout="loading model\n",
            stderr="CUDA out of memory\nlast stderr line\n",
        )

    monkeypatch.setattr("app.transcribe.subprocess.run", fake_run)

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda",
            "--backend",
            "fake",
            "--isolate-chunks",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Stage      : transcription" in captured.err
    assert "isolated chunk 1 failed at offset 00:00:00 with return code 7" in captured.err
    assert "stderr tail:" in captured.err
    assert "CUDA out of memory" in captured.err
    assert "stdout tail:" in captured.err
    assert commands[0][commands[0].index("--mode") + 1] == "gpu"
    assert not (output_dir / "lecture.json").exists()


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
    source_calls: list[tuple[Path, bool]] = []

    monkeypatch.setattr(
        "app.transcribe.detect_volume",
        lambda _path: VolumeStats(mean_volume_db=-36, max_volume_db=-10),
    )

    def fake_create_source_audio(
        _input_path: Path,
        output_wav_path: Path,
        *,
        normalize: bool,
    ) -> None:
        source_calls.append((output_wav_path, normalize))

    monkeypatch.setattr("app.transcribe.create_source_audio", fake_create_source_audio)

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
    assert source_calls == [(output_dir / ".work" / "source" / "lecture.source.wav", True)]
    assert metadata["preprocessing_applied"] is True
    assert metadata["audio_preparation_applied"] is True
    assert metadata["canonical_source_path"] == "lecture.source.wav"
    assert metadata["preprocessed_path"] == "lecture.source.wav"


def test_runner_preprocess_auto_normal_audio_creates_source_without_loudnorm(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    source_calls: list[tuple[Path, bool]] = []

    monkeypatch.setattr(
        "app.transcribe.detect_volume",
        lambda _path: VolumeStats(mean_volume_db=-20, max_volume_db=-3),
    )

    def fake_create_source_audio(
        _input_path: Path,
        output_wav_path: Path,
        *,
        normalize: bool,
    ) -> None:
        source_calls.append((output_wav_path, normalize))

    monkeypatch.setattr("app.transcribe.create_source_audio", fake_create_source_audio)

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
    assert source_calls == [(output_dir / ".work" / "source" / "lecture.source.wav", False)]
    assert metadata["preprocessing_applied"] is False
    assert metadata["audio_preparation_applied"] is True
    assert metadata["preprocessed_path"] is None


def test_runner_preprocess_off_still_creates_canonical_source(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "lecture.m4a"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    source_calls: list[tuple[Path, bool]] = []

    def fail_detect_volume(_path: Path) -> VolumeStats:
        raise AssertionError("volumedetect should not run when preprocessing is off")

    def fake_create_source_audio(
        _input_path: Path,
        output_wav_path: Path,
        *,
        normalize: bool,
    ) -> None:
        source_calls.append((output_wav_path, normalize))

    monkeypatch.setattr("app.transcribe.detect_volume", fail_detect_volume)
    monkeypatch.setattr("app.transcribe.create_source_audio", fake_create_source_audio)

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--backend",
            "fake",
            "--preprocess",
            "off",
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert source_calls == [(output_dir / ".work" / "source" / "lecture.source.wav", False)]
    assert metadata["preprocess_mode"] == "off"
    assert metadata["preprocessing_applied"] is False
    assert metadata["audio_preparation_applied"] is True
    assert metadata["canonical_source_path"] == "lecture.source.wav"


def test_runner_uses_canonical_source_duration_for_chunk_planning_when_preprocess_off(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "lecture.aac"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    raw_aac_warning = (
        "ffprobe warning: Estimating duration from bitrate, this may be inaccurate"
    )
    created_chunks: list[ChunkPlanItem] = []
    source_calls: list[tuple[Path, bool]] = []

    def fake_inspect_audio(path: Path) -> AudioInspection:
        if path.name == "lecture.source.wav":
            return AudioInspection(
                AudioMetadata(
                    duration_seconds=4416.0,
                    codec_name="pcm_s16le",
                    sample_rate=16000,
                    channels=1,
                    bit_rate=None,
                    format_name="wav",
                )
            )
        return AudioInspection(
            AudioMetadata(
                duration_seconds=4245.0,
                codec_name="aac",
                sample_rate=44100,
                channels=2,
                bit_rate=128000,
                format_name="aac",
            ),
            warning=raw_aac_warning,
        )

    def fake_create_chunk(_source_path: Path, chunk: ChunkPlanItem) -> None:
        created_chunks.append(chunk)

    def fake_create_source_audio(
        _input_path: Path,
        output_wav_path: Path,
        *,
        normalize: bool,
    ) -> None:
        source_calls.append((output_wav_path, normalize))

    monkeypatch.setattr("app.transcribe.inspect_audio", fake_inspect_audio)
    monkeypatch.setattr("app.transcribe.create_source_audio", fake_create_source_audio)
    monkeypatch.setattr("app.transcribe.create_chunk", fake_create_chunk)

    exit_code = main(
        [
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--backend",
            "fake",
            "--chunk-minutes",
            "72",
            "--preprocess",
            "off",
        ]
    )

    captured = capsys.readouterr()
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    run_log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert exit_code == 0
    assert source_calls == [(output_dir / ".work" / "source" / "lecture.source.wav", False)]
    assert len(created_chunks) == 2
    assert created_chunks[1].offset_seconds == 4320.0
    assert created_chunks[1].duration_seconds == 96.0
    assert metadata["input_duration_seconds"] is None
    assert metadata["source_duration_seconds"] == 4416.0
    assert metadata["original_audio"]["duration_seconds"] == 4245.0
    assert metadata["source_audio"]["duration_seconds"] == 4416.0
    assert metadata["preprocess_mode"] == "off"
    assert metadata["preprocessing_applied"] is False
    assert metadata["audio_preparation_applied"] is True
    assert raw_aac_warning in metadata["warnings"]
    assert f"warning={raw_aac_warning}" in run_log
    assert "Audio      : 01:13:36" in captured.out


def test_runner_keeps_raw_aac_duration_warning_as_informational(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "lecture.aac"
    input_path.write_bytes(b"fake audio")
    output_dir = tmp_path / "out"
    raw_aac_warning = (
        "ffprobe warning: Estimating duration from bitrate, this may be inaccurate"
    )

    def fake_inspect_audio(path: Path) -> AudioInspection:
        if path.name == "lecture.source.wav":
            return AudioInspection(
                AudioMetadata(
                    duration_seconds=4416.0,
                    codec_name="pcm_s16le",
                    sample_rate=16000,
                    channels=1,
                    bit_rate=None,
                    format_name="wav",
                )
            )
        return AudioInspection(
            AudioMetadata(
                duration_seconds=4245.0,
                codec_name="aac",
                sample_rate=44100,
                channels=2,
                bit_rate=128000,
                format_name="aac",
            ),
            warning=raw_aac_warning,
        )

    monkeypatch.setattr("app.transcribe.inspect_audio", fake_inspect_audio)
    monkeypatch.setattr(
        "app.transcribe.detect_volume",
        lambda _path: VolumeStats(mean_volume_db=-20, max_volume_db=-3),
    )
    monkeypatch.setattr("app.transcribe.create_chunk", lambda _source, _chunk: None)

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
    run_log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert exit_code == 0
    assert raw_aac_warning in metadata["warnings"]
    assert metadata["source_duration_seconds"] == 4416.0
    assert not any(
        "chunk planning may be inaccurate" in warning for warning in metadata["warnings"]
    )
    assert f"warning={raw_aac_warning}" in run_log
