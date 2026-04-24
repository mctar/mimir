"""
Corpus management for ASE facilitator.
Stores document chunks; used for Q&A context (manual document selection).
"""

import hashlib, json, time
import aiosqlite


# ─── Storage ───────────────────────────────────────────────────────────────

async def store_doc(db: aiosqlite.Connection, title: str, source: str,
                    content: str, label: str | None = None, role: str | None = None,
                    key_messages: list | None = None, usages: list | None = None) -> int:
    """Store a document chunk. Skips on duplicate content (hash collision) and updates metadata."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    km_json = json.dumps(key_messages, ensure_ascii=False) if key_messages is not None else None
    us_json = json.dumps(usages, ensure_ascii=False) if usages is not None else None
    cursor = await db.execute(
        "INSERT OR IGNORE INTO corpus_docs "
        "(title, source, content, content_hash, created_at, label, role, key_messages, usages) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (title, source, content, content_hash, time.time(), label, role, km_json, us_json),
    )
    # Update metadata even if chunk already existed (idempotent metadata sync)
    await db.execute(
        "UPDATE corpus_docs SET label=?, role=?, key_messages=?, usages=? WHERE content_hash=?",
        (label, role, km_json, us_json, content_hash),
    )
    await db.commit()
    return cursor.lastrowid or 0


async def list_docs(db: aiosqlite.Connection) -> list[dict]:
    async with db.execute(
        "SELECT id, title, source, created_at, active, label, role FROM corpus_docs ORDER BY created_at DESC"
    ) as cur:
        rows = await cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "source": r[2], "created_at": r[3], "active": bool(r[4]),
         "label": r[5], "role": r[6]}
        for r in rows
    ]


async def set_doc_active(db: aiosqlite.Connection, doc_id: int, active: bool):
    await db.execute("UPDATE corpus_docs SET active=? WHERE id=?", (int(active), doc_id))
    await db.commit()


async def delete_doc(db: aiosqlite.Connection, doc_id: int):
    await db.execute("DELETE FROM corpus_docs WHERE id=?", (doc_id,))
    await db.commit()


async def get_docs_by_ids(db: aiosqlite.Connection, ids: list[int]) -> list[dict]:
    """Return title + full content + metadata for the given IDs (active docs only)."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    async with db.execute(
        f"SELECT id, title, source, content, label, role, key_messages, usages "
        f"FROM corpus_docs WHERE id IN ({placeholders}) AND active=1",
        ids,
    ) as cur:
        rows = await cur.fetchall()
    return [{"id": r[0], "title": r[1], "source": r[2], "content": r[3],
             "label": r[4], "role": r[5], "key_messages": r[6], "usages": r[7]}
            for r in rows]


async def get_active_docs(db: aiosqlite.Connection) -> list[dict]:
    """Return all active chunks with their metadata, ordered by source then id."""
    async with db.execute(
        "SELECT id, title, source, content, label, role, key_messages, usages "
        "FROM corpus_docs WHERE active=1 ORDER BY source, id"
    ) as cur:
        rows = await cur.fetchall()
    return [{"id": r[0], "title": r[1], "source": r[2], "content": r[3],
             "label": r[4], "role": r[5], "key_messages": r[6], "usages": r[7]}
            for r in rows]
