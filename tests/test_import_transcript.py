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
