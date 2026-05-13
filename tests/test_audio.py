from pathlib import Path

import pytest

from app.audio import (
    AudioMetadata,
    build_ffprobe_command,
    build_preprocess_command,
    is_supported_audio_path,
    parse_ffprobe_metadata,
    validate_supported_audio_path,
)


@pytest.mark.parametrize("extension", [".aac", ".m4a", ".mp3", ".wav", ".flac", ".ogg"])
def test_supported_extensions_pass(extension: str) -> None:
    path = Path(f"lecture{extension}")

    assert is_supported_audio_path(path)
    validate_supported_audio_path(path)


def test_uppercase_extension_passes() -> None:
    assert is_supported_audio_path(Path("lecture.MP3"))


@pytest.mark.parametrize("path", [Path("notes.txt"), Path("lecture")])
def test_unsupported_or_missing_extension_fails(path: Path) -> None:
    assert not is_supported_audio_path(path)

    with pytest.raises(ValueError, match="Unsupported audio file extension"):
        validate_supported_audio_path(path)


def test_build_ffprobe_command() -> None:
    input_path = Path("audio/lecture.m4a")

    command = build_ffprobe_command(input_path)

    assert command[0] == "ffprobe"
    assert ["-v", "error"] == command[1:3]
    assert ["-print_format", "json"] == command[3:5]
    assert "-show_format" in command
    assert "-show_streams" in command
    assert command[-1] == str(input_path)


def test_parse_metadata_duration_from_format() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {"duration": "12.5"},
            "streams": [{"codec_type": "audio", "duration": "99.0"}],
        }
    )

    assert metadata.duration_seconds == 12.5


def test_parse_metadata_duration_from_first_audio_stream_when_format_missing() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {},
            "streams": [
                {"codec_type": "video", "duration": "1.0"},
                {"codec_type": "audio", "duration": "44.25"},
            ],
        }
    )

    assert metadata.duration_seconds == 44.25


def test_parse_metadata_prefers_format_duration_over_stream_duration() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {"duration": "33.0"},
            "streams": [{"codec_type": "audio", "duration": "44.0"}],
        }
    )

    assert metadata.duration_seconds == 33.0


def test_parse_metadata_ignores_video_stream_and_uses_audio_stream() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {"duration": "60.0"},
            "streams": [
                {
                    "codec_type": "video",
                    "sample_rate": "99999",
                    "channels": 99,
                    "codec_name": "h264",
                },
                {
                    "codec_type": "audio",
                    "sample_rate": "44100",
                    "channels": 2,
                    "codec_name": "aac",
                },
            ],
        }
    )

    assert metadata.sample_rate_hz == 44100
    assert metadata.channels == 2
    assert metadata.codec_name == "aac"


def test_parse_metadata_parses_audio_and_format_fields() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {"duration": "120.75", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
            "streams": [
                {
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": "1",
                    "codec_name": "aac",
                }
            ],
        }
    )

    assert metadata == AudioMetadata(
        duration_seconds=120.75,
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        sample_rate_hz=48000,
        channels=1,
        codec_name="aac",
    )


def test_parse_metadata_raises_clear_value_error_when_no_duration_available() -> None:
    with pytest.raises(ValueError, match="usable duration"):
        parse_ffprobe_metadata({"format": {}, "streams": [{"codec_type": "audio"}]})


def test_parse_metadata_missing_optional_fields_return_none() -> None:
    metadata = parse_ffprobe_metadata({"format": {"duration": "1.0"}, "streams": []})

    assert metadata == AudioMetadata(
        duration_seconds=1.0,
        format_name=None,
        sample_rate_hz=None,
        channels=None,
        codec_name=None,
    )


def test_parse_metadata_invalid_optional_sample_rate_and_channels_return_none() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {"duration": "1.0"},
            "streams": [
                {
                    "codec_type": "audio",
                    "sample_rate": "not-a-number",
                    "channels": "not-a-number",
                }
            ],
        }
    )

    assert metadata.sample_rate_hz is None
    assert metadata.channels is None


@pytest.mark.parametrize("duration", ["not-a-number", "nan", "-1"])
def test_parse_metadata_invalid_duration_raises_value_error(duration: str) -> None:
    with pytest.raises(ValueError, match="Invalid ffprobe duration"):
        parse_ffprobe_metadata({"format": {"duration": duration}, "streams": []})


def test_build_preprocess_command() -> None:
    input_path = Path("audio/lecture.m4a")
    output_wav_path = Path("tmp/lecture.wav")

    command = build_preprocess_command(input_path, output_wav_path)

    assert command[0] == "ffmpeg"
    assert "-y" in command
    assert command[command.index("-i") + 1] == str(input_path)
    assert ["-ac", "1"] == command[command.index("-ac") : command.index("-ac") + 2]
    assert ["-ar", "16000"] == command[command.index("-ar") : command.index("-ar") + 2]
    assert ["-c:a", "pcm_s16le"] == command[command.index("-c:a") : command.index("-c:a") + 2]
    assert command[-1] == str(output_wav_path)
