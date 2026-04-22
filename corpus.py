"""
Corpus management for ASE facilitator.
Stores document chunks; used for Q&A context (manual document selection).
"""

import hashlib, time
import aiosqlite


# ─── Storage ───────────────────────────────────────────────────────────────

async def store_doc(db: aiosqlite.Connection, title: str, source: str,
                    content: str) -> int:
    """Store a document chunk. Raises on duplicate content (hash collision)."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    cursor = await db.execute(
        "INSERT INTO corpus_docs (title, source, content, content_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (title, source, content, content_hash, time.time()),
    )
    await db.commit()
    return cursor.lastrowid


async def list_docs(db: aiosqlite.Connection) -> list[dict]:
    async with db.execute(
        "SELECT id, title, source, created_at, active FROM corpus_docs ORDER BY created_at DESC"
    ) as cur:
        rows = await cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "source": r[2], "created_at": r[3], "active": bool(r[4])}
        for r in rows
    ]


async def set_doc_active(db: aiosqlite.Connection, doc_id: int, active: bool):
    await db.execute("UPDATE corpus_docs SET active=? WHERE id=?", (int(active), doc_id))
    await db.commit()


async def delete_doc(db: aiosqlite.Connection, doc_id: int):
    await db.execute("DELETE FROM corpus_docs WHERE id=?", (doc_id,))
    await db.commit()


async def get_docs_by_ids(db: aiosqlite.Connection, ids: list[int]) -> list[dict]:
    """Return title + full content for the given IDs (active docs only)."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    async with db.execute(
        f"SELECT id, title, content FROM corpus_docs WHERE id IN ({placeholders}) AND active=1",
        ids,
    ) as cur:
        rows = await cur.fetchall()
    return [{"id": r[0], "title": r[1], "content": r[2]} for r in rows]
