# tests/test_append_transcript.py
import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import db
from app import _parse_transcript, _split_segments


@pytest.fixture
def tmp_db(tmp_path):
    path = str(tmp_path / "test.db")
    asyncio.run(db.init_db(path))
    yield path
    asyncio.run(db.close_db())


def _run(coro):
    return asyncio.run(coro)


def _seed_external_session(sid: str, n_segments: int = 3):
    """Create an external session with n_segments and return max (seq, timestamp)."""
    async def _go():
        await db.create_session(sid, "Test Topic", source="external")
        await db.end_session(sid)
        base_ts = 1_000_000.0
        for i in range(1, n_segments + 1):
            await db.store_segment(
                session_id=sid,
                seq=i,
                text=f"Segment {i}",
                is_partial=False,
                timestamp=base_ts + i - 1,
                stt_backend="external",
                stt_language="",
            )
        return n_segments, base_ts + n_segments - 1
    return asyncio.run(_go())


def test_append_continues_seq_and_timestamp(tmp_db):
    """Appended segments have seq and timestamp continuing from the last existing segment."""
    sid = "ext-001"
    last_seq, last_ts = _seed_external_session(sid, n_segments=3)

    # Append 2 more segments directly via db (testing the DB layer)
    async def _go():
        await db.store_segment(sid, last_seq + 1, "Appended 1", False, last_ts + 1, stt_backend="external")
        await db.store_segment(sid, last_seq + 2, "Appended 2", False, last_ts + 2, stt_backend="external")

    asyncio.run(_go())

    segs = _run(db.get_session_transcript(sid))
    assert len(segs) == 5
    assert segs[3]["seq"] == 4
    assert segs[4]["seq"] == 5
    assert segs[3]["timestamp"] == pytest.approx(1_000_003.0)
    assert segs[4]["timestamp"] == pytest.approx(1_000_004.0)
    assert segs[3]["text"] == "Appended 1"
    assert segs[3]["stt_backend"] == "external"


def test_create_session_stores_source_correctly(tmp_db):
    """Sessions store their source correctly in the database."""
    async def _go():
        await db.create_session("live-001", "Live", source="live")

    asyncio.run(_go())
    session = _run(db.get_session("live-001"))
    assert session["source"] == "live"


def test_get_last_segment_info_returns_last(tmp_db):
    """get_last_segment_info returns the highest-seq segment's seq and timestamp."""
    sid = "ext-last"
    _seed_external_session(sid, n_segments=3)
    info = asyncio.run(db.get_last_segment_info(sid))
    assert info is not None
    assert info["seq"] == 3
    assert info["timestamp"] == pytest.approx(1_000_002.0)


def test_get_last_segment_info_none_for_empty(tmp_db):
    """get_last_segment_info returns None when session has no segments."""
    async def _go():
        await db.create_session("empty-001", "Empty", source="external")
        await db.end_session("empty-001")
    asyncio.run(_go())
    info = asyncio.run(db.get_last_segment_info("empty-001"))
    assert info is None


def test_append_empty_transcript_has_no_segments():
    """An empty text produces zero parts when split."""
    cleaned = _parse_transcript("   ")
    assert cleaned == ""


def test_parse_and_split_pipeline():
    """_parse_transcript + _split_segments correctly extract segments from raw text."""
    raw = "Hello world.\n\nThis is paragraph two.\n\nAnd paragraph three."
    parts = _split_segments(_parse_transcript(raw))
    assert len(parts) == 3
    assert parts[0] == "Hello world."
    assert parts[2] == "And paragraph three."
