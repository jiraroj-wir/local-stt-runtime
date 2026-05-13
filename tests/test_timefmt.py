import pytest

from app.timefmt import format_srt_timestamp


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00,000"),
        (1.234, "00:00:01,234"),
        (3661.5, "01:01:01,500"),
        (1.9996, "00:00:02,000"),
    ],
)
def test_format_srt_timestamp(seconds: float, expected: str) -> None:
    assert format_srt_timestamp(seconds) == expected


def test_format_srt_timestamp_rejects_negative_seconds() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        format_srt_timestamp(-0.001)
