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


@pytest.mark.parametrize("name", ["a.aac", "a.m4a", "a.mp3", "a.wav", "a.flac", "a.ogg", "a.MP3"])
def test_supported_extensions(name: str) -> None:
    path = Path(name)
    assert is_supported_audio_path(path)
    validate_supported_audio_path(path)


def test_unsupported_extension_raises() -> None:
    path = Path("a.txt")
    assert not is_supported_audio_path(path)
    with pytest.raises(ValueError, match="Unsupported audio extension"):
        validate_supported_audio_path(path)


def test_build_ffprobe_command_shape() -> None:
    path = Path("audio/input.m4a")
    command = build_ffprobe_command(path)

    assert command[0] == "ffprobe"
    assert "-print_format" in command
    assert "json" in command
    assert "-show_format" in command
    assert "-show_streams" in command
    assert command[-1] == str(path)


def test_parse_ffprobe_uses_format_duration() -> None:
    payload = {
        "format": {"duration": "12.34", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [
            {
                "codec_type": "audio",
                "sample_rate": "16000",
                "channels": 1,
                "codec_name": "aac",
                "duration": "99.0",
            }
        ],
    }

    metadata = parse_ffprobe_metadata(payload)

    assert metadata == AudioMetadata(
        duration_seconds=12.34,
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        sample_rate_hz=16000,
        channels=1,
        codec_name="aac",
    )


def test_parse_ffprobe_uses_audio_stream_duration_fallback() -> None:
    payload = {
        "format": {"format_name": "mp3"},
        "streams": [{"codec_type": "audio", "duration": "45.5"}],
    }

    metadata = parse_ffprobe_metadata(payload)

    assert metadata.duration_seconds == 45.5


def test_parse_ffprobe_ignores_video_stream_and_uses_audio() -> None:
    payload = {
        "format": {"duration": "10.0"},
        "streams": [
            {"codec_type": "video", "sample_rate": "99999", "channels": 99, "codec_name": "h264"},
            {"codec_type": "audio", "sample_rate": "22050", "channels": "2", "codec_name": "mp3"},
        ],
    }

    metadata = parse_ffprobe_metadata(payload)

    assert metadata.sample_rate_hz == 22050
    assert metadata.channels == 2
    assert metadata.codec_name == "mp3"


def test_parse_ffprobe_raises_when_no_duration() -> None:
    payload = {"format": {}, "streams": [{"codec_type": "audio"}]}

    with pytest.raises(ValueError, match="Unable to determine audio duration"):
        parse_ffprobe_metadata(payload)


def test_parse_ffprobe_missing_optional_fields_returns_none() -> None:
    payload = {"format": {"duration": "3.0"}, "streams": [{"codec_type": "audio"}]}

    metadata = parse_ffprobe_metadata(payload)

    assert metadata.format_name is None
    assert metadata.sample_rate_hz is None
    assert metadata.channels is None
    assert metadata.codec_name is None


def test_build_preprocess_command_shape() -> None:
    input_path = Path("audio/lecture.m4a")
    output_path = Path("tmp/lecture.wav")

    command = build_preprocess_command(input_path, output_path)

    assert command[0] == "ffmpeg"
    assert "-y" in command
    assert command[command.index("-i") + 1] == str(input_path)
    assert "-ac" in command
    assert command[command.index("-ac") + 1] == "1"
    assert "-ar" in command
    assert command[command.index("-ar") + 1] == "16000"
    assert "-c:a" in command
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert command[-1] == str(output_path)
