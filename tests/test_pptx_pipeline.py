import asyncio
import os
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import db


@pytest.fixture
def tmp_db(tmp_path):
    path = str(tmp_path / "test.db")
    asyncio.run(db.init_db(path))
    yield path
    asyncio.run(db.close_db())


def test_recaps_has_pptx_columns(tmp_db):
    """Les 4 nouvelles colonnes existent après init_db."""
    import aiosqlite

    async def check():
        conn = await aiosqlite.connect(tmp_db)
        cursor = await conn.execute("PRAGMA table_info(recaps)")
        cols = {row[1] for row in await cursor.fetchall()}
        await conn.close()
        return cols

    cols = asyncio.run(check())
    assert "pptx_instructions" in cols
    assert "deck_spec_json" in cols
    assert "deck_spec_model" in cols
    assert "deck_spec_at" in cols


def test_get_pptx_data_returns_none_for_missing(tmp_db):
    """get_pptx_data retourne None si pas de recap."""
    result = asyncio.run(db.get_pptx_data("nonexistent-session"))
    assert result is None


def test_save_and_get_pptx_instructions(tmp_db):
    """Round-trip save/get des instructions."""
    sid = "test-session-001"

    async def setup():
        await db._db.execute(
            "INSERT INTO sessions (id, topic, created_at) VALUES (?, ?, ?)",
            (sid, "Test", 1000.0),
        )
        await db._db.execute(
            "INSERT INTO recaps (session_id, recap_json, model, created_at) VALUES (?, ?, ?, ?)",
            (sid, '{"elevator_pitch": "test"}', "test", 1000.0),
        )
        await db._db.commit()

    asyncio.run(setup())
    asyncio.run(db.save_pptx_instructions(sid, "Traduire en anglais"))
    result = asyncio.run(db.get_pptx_data(sid))
    assert result["instructions"] == "Traduire en anglais"
    assert result["deck_spec"] is None


def test_save_and_get_deck_spec(tmp_db):
    """Round-trip save/get du deck_spec."""
    sid = "test-session-002"
    spec = {"schema_version": 1, "slides": [{"layout": "cover", "slots": {"title": "Test"}}]}

    async def setup():
        await db._db.execute(
            "INSERT INTO sessions (id, topic, created_at) VALUES (?, ?, ?)",
            (sid, "Test", 1000.0),
        )
        await db._db.execute(
            "INSERT INTO recaps (session_id, recap_json, model, created_at) VALUES (?, ?, ?, ?)",
            (sid, '{"elevator_pitch": "test"}', "test", 1000.0),
        )
        await db._db.commit()

    asyncio.run(setup())
    asyncio.run(db.save_deck_spec(sid, spec, "gemini-2.5-flash"))
    result = asyncio.run(db.get_pptx_data(sid))
    assert result["deck_spec"]["schema_version"] == 1
    assert result["deck_spec"]["slides"][0]["layout"] == "cover"
    assert result["deck_spec_model"] == "gemini-2.5-flash"


def test_pptx_instructions_survive_recap_regeneration(tmp_db):
    """store_recap does not wipe pptx_instructions (UPSERT behaviour)."""
    sid = "test-session-003"

    async def setup():
        await db._db.execute(
            "INSERT INTO sessions (id, topic, created_at) VALUES (?, ?, ?)",
            (sid, "Test", 1000.0),
        )
        await db._db.commit()

    asyncio.run(setup())
    # First recap
    asyncio.run(db.store_recap(sid, {"elevator_pitch": "v1"}, "model-a"))
    # Save pptx instructions
    asyncio.run(db.save_pptx_instructions(sid, "Traduire en anglais"))
    # Regenerate recap (simulates user clicking "Regenerate Recap")
    asyncio.run(db.store_recap(sid, {"elevator_pitch": "v2"}, "model-b"))
    # pptx_instructions must survive
    result = asyncio.run(db.get_pptx_data(sid))
    assert result["instructions"] == "Traduire en anglais"


def test_assemble_pptx_creates_file(tmp_path):
    """_assemble_pptx génère un fichier .pptx sans lever d'exception."""
    import export as exp

    deck_spec = {
        "schema_version": 1,
        "slides": [
            {"layout": "cover",      "slots": {"title": "Test Session", "date": "23 April 2026", "duration": "45m"}},
            {"layout": "quote-large","slots": {"title": "Pitch", "body": "Une phrase de pitch."}},
            {"layout": "bullets",    "slots": {"title": "Points clés", "bullets": ["Point 1", "Point 2", "Point 3"]}},
            {"layout": "three-columns", "slots": {"title": "Connexions", "col1": "A ↔ B", "col2": "C ↔ D", "col3": "E ↔ F"}},
            {"layout": "concepts",   "slots": {"title": "Concepts", "terms": ["A", "B", "C"], "edges": ["A → B", "B → C"]}},
        ],
    }
    output = str(tmp_path / "test_out.pptx")
    exp._assemble_pptx(deck_spec, output)
    assert os.path.exists(output)
    assert os.path.getsize(output) > 1000


def test_assemble_pptx_unknown_layout_fallback(tmp_path):
    """Un layout inconnu ne lève pas d'exception (fallback bullets)."""
    import export as exp

    deck_spec = {
        "schema_version": 1,
        "slides": [
            {"layout": "invented-layout", "slots": {"title": "Fallback", "bullets": ["item"]}},
        ],
    }
    output = str(tmp_path / "fallback.pptx")
    exp._assemble_pptx(deck_spec, output)
    assert os.path.exists(output)


def test_assemble_pptx_all_layouts(tmp_path):
    """All 7 layouts render without exception."""
    import export as exp

    deck_spec = {
        "schema_version": 1,
        "slides": [
            {"layout": "text-large",   "slots": {"title": "Text", "body": "Some long body text here."}},
            {"layout": "two-columns",  "slots": {"title": "Two cols", "left": "Left content", "right": "Right content"}},
        ],
    }
    output = str(tmp_path / "all_layouts.pptx")
    exp._assemble_pptx(deck_spec, output)
    assert os.path.exists(output)
    assert os.path.getsize(output) > 1000


def test_generate_deck_spec_parses_valid_json():
    """generate_deck_spec parse correctement un JSON LLM valide."""
    import export as exp
    import unittest.mock as mock

    valid_spec = {
        "schema_version": 1,
        "slides": [
            {"layout": "cover", "slots": {"title": "Test", "date": "2026-04-23", "duration": "30m"}},
            {"layout": "bullets", "slots": {"title": "Points", "bullets": ["A", "B"]}},
        ],
    }
    import json

    async def fake_call(tier, system, user):
        return json.dumps(valid_spec)

    with mock.patch.object(exp, "_llm_call_slides", fake_call):
        chain = [{"provider": "test", "model": "test-model"}]
        result = asyncio.run(
            exp.generate_deck_spec(
                transcript="Texte de test",
                recap={"elevator_pitch": "pitch"},
                instructions=None,
                current_deck_spec=None,
                chain=chain,
            )
        )

    assert result["schema_version"] == 1
    assert len(result["slides"]) == 2
    assert result["slides"][0]["layout"] == "cover"


def test_generate_deck_spec_retries_on_invalid_json():
    """generate_deck_spec réessaie une fois si le JSON est invalide."""
    import export as exp
    import unittest.mock as mock
    import json

    valid_spec = {"schema_version": 1, "slides": [{"layout": "bullets", "slots": {"title": "T", "bullets": []}}]}
    call_count = {"n": 0}

    async def fake_call(tier, system, user):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "not valid json {{{"
        return json.dumps(valid_spec)

    with mock.patch.object(exp, "_llm_call_slides", fake_call):
        chain = [{"provider": "test", "model": "test-model"}]
        result = asyncio.run(
            exp.generate_deck_spec(
                transcript="Texte",
                recap={},
                instructions=None,
                current_deck_spec=None,
                chain=chain,
            )
        )

    assert call_count["n"] == 2
    assert result["slides"][0]["layout"] == "bullets"
