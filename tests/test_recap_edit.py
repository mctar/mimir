import pytest
import db as db_module


@pytest.fixture(autouse=True)
async def fresh_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    await db_module.init_db(db_path)
    yield
    await db_module.close_db()


async def test_store_recap_manual_model():
    recap = {"elevator_pitch": "We decided to pivot.", "schema_version": 2}
    await db_module.store_recap("sess-001", recap, model="manual")
    result = await db_module.get_recap("sess-001")
    assert result is not None
    assert result["model"] == "manual"
    assert result["recap"]["elevator_pitch"] == "We decided to pivot."


async def test_store_recap_manual_overwrites_ai():
    await db_module.store_recap("sess-001", {"elevator_pitch": "AI version"}, model="hugin/gemma4:26b")
    await db_module.store_recap("sess-001", {"elevator_pitch": "Human version"}, model="manual")
    result = await db_module.get_recap("sess-001")
    assert result["model"] == "manual"
    assert result["recap"]["elevator_pitch"] == "Human version"
