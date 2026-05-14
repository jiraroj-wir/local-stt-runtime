import pytest

from app.config import cpu_config, cuda_config, get_device_config


def test_cpu_config_selects_medium_en_int8_cpu() -> None:
    config = cpu_config()

    assert config.model == "Systran/faster-whisper-medium.en"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.beam_size == 5
    assert config.language == "en"
    assert config.vad_filter is True
    assert config.condition_on_previous_text is False
    assert config.cpu_threads == "auto"


def test_cuda_config_selects_large_v3_int8_cuda() -> None:
    config = cuda_config()

    assert config.model == "Systran/faster-whisper-large-v3"
    assert config.device == "cuda"
    assert config.compute_type == "int8"
    assert config.beam_size == 5
    assert config.language == "en"
    assert config.vad_filter is True
    assert config.condition_on_previous_text is False
    assert config.word_timestamps is False
    assert config.batch_size == 1


def test_get_device_config_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unsupported transcription device"):
        get_device_config("gpu")
