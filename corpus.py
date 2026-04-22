"""
Corpus management for ASE facilitator.
Stores document chunks with embeddings; uses in-memory numpy search.
Corpus is small and fixed — loaded once at startup, no live updates during session.
"""

import hashlib, time, os
import numpy as np
import aiosqlite
import aiohttp

EMBED_DIM = 768  # nomic-embed-text (Ollama) output dimension

HUGIN_BASE_URL = os.environ.get("HUGIN_BASE_URL", "https://munin.btrbot.com")
HUGIN_CF_ID = os.environ.get("HUGIN_CF_ID", "")
HUGIN_CF_SECRET = os.environ.get("HUGIN_CF_SECRET", "")

# ─── Module-level cache (invalidated on write) ─────────────────────────────
_corpus_cache: list[dict] | None = None


def _invalidate_cache():
    global _corpus_cache
    _corpus_cache = None


# ─── Storage ───────────────────────────────────────────────────────────────

async def store_doc(db: aiosqlite.Connection, title: str, source: str,
                    content: str, embedding: np.ndarray) -> int:
    """Store a document chunk. Raises on duplicate content (hash collision)."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    emb_bytes = embedding.astype(np.float32).tobytes()
    cursor = await db.execute(
        "INSERT INTO corpus_docs (title, source, content, content_hash, embedding, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (title, source, content, content_hash, emb_bytes, time.time()),
    )
    await db.commit()
    _invalidate_cache()
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


async def load_corpus(db: aiosqlite.Connection) -> list[dict]:
    """Return active docs with embeddings. Cached until next write."""
    global _corpus_cache
    if _corpus_cache is not None:
        return _corpus_cache
    async with db.execute(
        "SELECT id, title, source, content, embedding FROM corpus_docs WHERE active=1"
    ) as cur:
        rows = await cur.fetchall()
    _corpus_cache = [
        {
            "id": r[0], "title": r[1], "source": r[2], "content": r[3],
            "embedding": np.frombuffer(r[4], dtype=np.float32).copy(),
        }
        for r in rows
    ]
    return _corpus_cache


async def set_doc_active(db: aiosqlite.Connection, doc_id: int, active: bool):
    await db.execute("UPDATE corpus_docs SET active=? WHERE id=?", (int(active), doc_id))
    await db.commit()
    _invalidate_cache()


async def delete_doc(db: aiosqlite.Connection, doc_id: int):
    await db.execute("DELETE FROM corpus_docs WHERE id=?", (doc_id,))
    await db.commit()
    _invalidate_cache()


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


# ─── Search ────────────────────────────────────────────────────────────────

def search_corpus(query_embedding: np.ndarray, corpus: list[dict], k: int = 5) -> list[dict]:
    """Return top-k corpus chunks by cosine similarity."""
    if not corpus:
        return []
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    scored = []
    for doc in corpus:
        e = doc["embedding"]
        score = float(np.dot(q_norm, e) / (np.linalg.norm(e) + 1e-9))
        scored.append({**doc, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# ─── Embedding ─────────────────────────────────────────────────────────────

async def embed_text(text: str) -> np.ndarray | None:
    """Embed text via Ollama nomic-embed-text. Returns None if unavailable."""
    headers: dict = {"Content-Type": "application/json"}
    if HUGIN_CF_ID and HUGIN_CF_SECRET:
        headers["CF-Access-Client-Id"] = HUGIN_CF_ID
        headers["CF-Access-Client-Secret"] = HUGIN_CF_SECRET
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HUGIN_BASE_URL}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return np.array(data["embedding"], dtype=np.float32)
    except Exception:
        pass
    return None


# ─── Prompt helpers ────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """\
You are a workshop synthesis assistant for a Capgemini Invent facilitator.
Extract structured insights from a live workshop transcript.

OUTPUT FORMAT (strict JSON, no preamble, no markdown fences):
{"themes": [{"name": "<theme>", "alignment": ["<point>"], "disagreement": ["<point>"], "unresolved": ["<point>"]}]}

Rules:
- 3 to 6 themes maximum.
- Each bullet: one direct factual sentence.
- Transcript is the primary source. Corpus is background context only.
- Signal contradictions explicitly in "disagreement".
- "unresolved" = topics debated but not concluded.
- Language: match the transcript language (FR or EN).
- JSON only. No preamble. No commentary."""

QA_SYSTEM_PROMPT = """\
You are a workshop synthesis assistant for a Capgemini Invent facilitator.
Answer the question concisely and directly based ONLY on the transcript and corpus provided.
If the answer is not in the transcript, say so explicitly.
Language: match the transcript language (FR or EN)."""


def build_synthesis_user_prompt(transcript: str, corpus_passages: list[dict]) -> str:
    corpus_block = ""
    if corpus_passages:
        parts = [
            f"[CORPUS: {p['title']}]\n{p['content'][:500]}"
            for p in corpus_passages
        ]
        corpus_block = (
            "\n\nCORPUS CONTEXT (background only — do not override transcript):\n"
            + "\n\n".join(parts)
        )
    return f"WORKSHOP TRANSCRIPT:\n\n{transcript}{corpus_block}\n\nGenerate the synthesis JSON."


def build_qa_user_prompt(transcript: str, question: str, corpus_passages: list[dict]) -> str:
    corpus_block = ""
    if corpus_passages:
        parts = [f"[CORPUS: {p['title']}]\n{p['content'][:300]}" for p in corpus_passages]
        corpus_block = "\n\nCORPUS:\n" + "\n\n".join(parts)
    return f"TRANSCRIPT:\n{transcript}{corpus_block}\n\nQUESTION: {question}"
