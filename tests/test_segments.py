from app.segments import TranscriptionSegment, offset_segments


def test_offset_segments_adds_chunk_offset_to_timestamps() -> None:
    segments = [TranscriptionSegment(start=5.0, end=7.5, text="hello")]

    shifted = offset_segments(segments, 1200.0)

    assert shifted == [TranscriptionSegment(start=1205.0, end=1207.5, text="hello")]
