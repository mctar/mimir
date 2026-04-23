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
