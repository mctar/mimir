import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import _parse_transcript, _split_segments


def test_vtt_strips_header_timestamps_and_speaker_tags():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v John>Hello everyone.

00:00:04.000 --> 00:00:06.000
<v Mary>Good morning."""
    result = _parse_transcript(vtt)
    assert "Hello everyone." in result
    assert "Good morning." in result
    assert "WEBVTT" not in result
    assert "-->" not in result
    assert "<v " not in result


def test_vtt_with_cue_numbers_stripped():
    vtt = """WEBVTT

1
00:00:01.000 --> 00:00:02.000
First line.

2
00:00:03.000 --> 00:00:04.000
Second line."""
    result = _parse_transcript(vtt)
    assert "First line." in result
    assert "Second line." in result
    assert "1" not in result.splitlines()[0]


def test_plain_text_returned_unchanged():
    text = "Hello world.\n\nThis is a second paragraph."
    assert _parse_transcript(text) == text


def test_malformed_vtt_treated_as_plain_text():
    text = "Just some text without any VTT structure at all"
    assert _parse_transcript(text) == text


def test_empty_vtt_returns_empty_string():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n"
    assert _parse_transcript(vtt) == ""


def test_split_segments_by_double_newline():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    parts = _split_segments(text)
    assert parts == ["First paragraph.", "Second paragraph.", "Third paragraph."]


def test_split_segments_fallback_single_newline():
    text = "Line one.\nLine two.\nLine three."
    parts = _split_segments(text)
    assert parts == ["Line one.", "Line two.", "Line three."]


def test_split_segments_ignores_blank_parts():
    text = "First.\n\n\n\nSecond."
    parts = _split_segments(text)
    assert parts == ["First.", "Second."]


import db as db_module


@pytest.fixture
def tmp_db(tmp_path):
    path = str(tmp_path / "test.db")
    asyncio.run(db_module.init_db(path))
    yield path
    asyncio.run(db_module.close_db())


def test_import_creates_session_with_external_source(tmp_db):
    """Simulates the import endpoint: session gets source='external' and segments are inserted."""
    import time as time_mod

    raw = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    cleaned = _parse_transcript(raw)
    parts = _split_segments(cleaned)
    sid = "imp-test1"

    async def run():
        await db_module.create_session(sid, "Test Import", source="external")
        await db_module.end_session(sid)
        base = time_mod.time()
        for seq, text in enumerate(parts, start=1):
            await db_module.store_segment(
                sid, seq, text, False, base + seq - 1,
                stt_backend="external", stt_language="",
            )
        segments = await db_module.get_session_transcript(sid)
        return segments

    segments = asyncio.run(run())
    assert len(segments) == 3
    assert segments[0]["text"] == "First paragraph."
    assert segments[2]["text"] == "Third paragraph."
    assert segments[0]["stt_backend"] == "external"


def test_import_vtt_end_to_end(tmp_db):
    """VTT content flows correctly through parse → split → store."""
    import time as time_mod

    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Alice>We need to prioritize the roadmap.

00:00:04.000 --> 00:00:06.000
<v Bob>Agreed, let's focus on Q3 deliverables."""
    cleaned = _parse_transcript(vtt)
    parts = _split_segments(cleaned)
    sid = "imp-test2"

    async def run():
        await db_module.create_session(sid, "VTT Test", source="external")
        await db_module.end_session(sid)
        base = time_mod.time()
        for seq, text in enumerate(parts, start=1):
            await db_module.store_segment(
                sid, seq, text, False, base + seq - 1,
                stt_backend="external", stt_language="",
            )
        return await db_module.get_session_transcript(sid)

    segments = asyncio.run(run())
    assert len(segments) == 2
    assert "roadmap" in segments[0]["text"]
    assert "Q3" in segments[1]["text"]
