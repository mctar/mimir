import pytest
import aiosqlite


@pytest.fixture
async def tmp_db(tmp_path):
    """Async aiosqlite connection with corpus_docs table."""
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE corpus_docs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT NOT NULL,
                source       TEXT,
                content      TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                created_at   REAL NOT NULL,
                active       INTEGER NOT NULL DEFAULT 1
            )
        """)
        await db.commit()
        yield db
