from pathlib import Path

import pytest

from app.audio import (
    build_ffprobe_command,
    build_preprocess_command,
    is_supported_audio_path,
    parse_ffprobe_metadata,
    validate_supported_audio_path,
)


def test_supported_extensions_pass() -> None:
    for ext in (".aac", ".m4a", ".mp3", ".wav", ".flac", ".ogg"):
        assert is_supported_audio_path(Path(f"lecture{ext}"))


def test_supported_extension_is_case_insensitive() -> None:
    assert is_supported_audio_path(Path("lecture.MP3"))


def test_unsupported_extension_fails() -> None:
    with pytest.raises(ValueError, match="Unsupported audio extension"):
        validate_supported_audio_path(Path("notes.txt"))


def test_build_ffprobe_command_shape() -> None:
    input_path = Path("audio/input.m4a")

    command = build_ffprobe_command(input_path)

    assert command[0] == "ffprobe"
    assert "-print_format" in command
    assert "json" in command
    assert "-show_format" in command
    assert "-show_streams" in command
    assert command[-1] == str(input_path)


def test_parse_ffprobe_metadata_uses_format_duration() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {"duration": "12.5", "format_name": "mov,mp4,m4a"},
            "streams": [
                {
                    "codec_type": "audio",
                    "sample_rate": "44100",
                    "channels": 2,
                    "codec_name": "aac",
                    "duration": "10.0",
                }
            ],
        }
    )

    assert metadata.duration_seconds == pytest.approx(12.5)
    assert metadata.format_name == "mov,mp4,m4a"
    assert metadata.sample_rate_hz == 44100
    assert metadata.channels == 2
    assert metadata.codec_name == "aac"


def test_parse_ffprobe_metadata_uses_audio_stream_duration_fallback() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {"format_name": "ogg"},
            "streams": [{"codec_type": "audio", "duration": "33.25"}],
        }
    )

    assert metadata.duration_seconds == pytest.approx(33.25)


def test_parse_ffprobe_metadata_ignores_non_audio_streams() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {"duration": "101.0", "format_name": "matroska,webm"},
            "streams": [
                {
                    "codec_type": "video",
                    "sample_rate": "99999",
                    "channels": 9,
                    "codec_name": "h264",
                },
                {
                    "codec_type": "audio",
                    "sample_rate": "16000",
                    "channels": "1",
                    "codec_name": "opus",
                },
            ],
        }
    )

    assert metadata.sample_rate_hz == 16000
    assert metadata.channels == 1
    assert metadata.codec_name == "opus"


def test_parse_ffprobe_metadata_missing_duration_raises() -> None:
    with pytest.raises(ValueError, match="Unable to determine audio duration"):
        parse_ffprobe_metadata(
            {
                "format": {"format_name": "wav"},
                "streams": [{"codec_type": "audio"}],
            }
        )


def test_parse_ffprobe_metadata_missing_optional_fields_are_none() -> None:
    metadata = parse_ffprobe_metadata(
        {
            "format": {"duration": "5.0"},
            "streams": [{"codec_type": "audio"}],
        }
    )

    assert metadata.format_name is None
    assert metadata.sample_rate_hz is None
    assert metadata.channels is None
    assert metadata.codec_name is None


def test_build_preprocess_command_shape() -> None:
    input_path = Path("audio/input.flac")
    output_path = Path("audio/input/preprocessed.wav")

    command = build_preprocess_command(input_path, output_path)

    assert command[0] == "ffmpeg"
    assert "-y" in command
    i_index = command.index("-i")
    assert command[i_index + 1] == str(input_path)
    assert "-ac" in command
    assert command[command.index("-ac") + 1] == "1"
    assert "-ar" in command
    assert command[command.index("-ar") + 1] == "16000"
    assert "-c:a" in command
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert command[-1] == str(output_path)
