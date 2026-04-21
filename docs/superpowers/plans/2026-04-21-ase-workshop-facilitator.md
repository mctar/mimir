# ASE Workshop Facilitator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/facilitator` view to Mímir with live synthesis (themes, alignment, disagreements), Q&A, markdown copy, and Capgemini PPTX export — built additively on top of mimir's existing transcript/WebSocket/LLM infrastructure.

**Architecture:** New files only; minimal changes to `app.py` (mount router, add synthesis background task) and `db.py` (add corpus_docs table). Vector similarity uses in-memory numpy (corpus is small/fixed — no sqlite-vec extension needed). LLM calls follow the same pattern as `generate_recap` in `app.py`, importing provider config from env vars. `routes_facilitator.py` receives state via a `configure()` call from `app.py` at startup.

**Tech Stack:** FastAPI, aiosqlite, numpy, aiohttp, python-pptx, pdfplumber, python-docx. Existing: mimir's LLM chain (Ollama/Gemini/Anthropic), WebSocket broadcast.

**Spec:** `docs/superpowers/specs/2026-04-21-ase-workshop-facilitator-design.md`
**Save this plan to repo:** `docs/superpowers/plans/2026-04-21-ase-workshop-facilitator.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `corpus.py` | Create | Corpus storage (aiosqlite), in-memory embedding search (numpy), embedding via Ollama |
| `ingest_corpus.py` | Create | CLI script: parse PDF/DOCX/PPTX/TXT → chunk → embed → store |
| `routes_facilitator.py` | Create | Routes: /facilitator, /v1/live-synthesis, /v1/qa, /v1/corpus CRUD, /v1/sessions/{id}/export/synthesis-pptx |
| `facilitator.html` | Create | Facilitator view: transcript + synthesis panel + Q&A (Capgemini brand) |
| `corpus.html` | Create | Corpus admin page: list docs, toggle active, delete |
| `assets/template_cap_pptx.pptx` | Create (copy) | Capgemini template used for PPTX export |
| `db.py` | Modify | Add `corpus_docs` table to `init_db()` schema |
| `app.py` | Modify | Call `routes_facilitator.configure()`, include router, add synthesis background task |
| `requirements.txt` | Modify | Add: numpy, pdfplumber, python-docx, python-pptx |
| `tests/conftest.py` | Create | Pytest fixtures: temp aiosqlite DB |
| `tests/test_corpus.py` | Create | Unit tests for corpus store/load/search/embed |
| `tests/test_synthesis.py` | Create | Unit tests for prompt builder and JSON parsing |
| `tests/test_pptx.py` | Create | Unit tests for PPTX generation |

---

## Task 1: Requirements + DB schema

**Files:**
- Modify: `requirements.txt`
- Modify: `db.py` (add corpus_docs table in `init_db`)
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add dependencies to requirements.txt**

Open `requirements.txt` and add these lines:
```
numpy
pdfplumber
python-docx
python-pptx
pytest
pytest-asyncio
```

- [ ] **Step 2: Add corpus_docs table to db.py init_db()**

In `db.py`, inside the `executescript("""...""")` call in `init_db()`, append before the closing `"""`:

```sql
        CREATE TABLE IF NOT EXISTS corpus_docs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            title        TEXT NOT NULL,
            source       TEXT,
            content      TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            embedding    BLOB NOT NULL,
            created_at   REAL NOT NULL,
            active       INTEGER NOT NULL DEFAULT 1
        );
```

- [ ] **Step 3: Create tests/\_\_init\_\_.py**

```python
```
(empty file)

- [ ] **Step 4: Create tests/conftest.py**

```python
import asyncio
import pytest
import aiosqlite


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


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
                embedding    BLOB NOT NULL,
                created_at   REAL NOT NULL,
                active       INTEGER NOT NULL DEFAULT 1
            )
        """)
        await db.commit()
        yield db
```

- [ ] **Step 5: Create pytest.ini**

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 6: Install dependencies**

```bash
cd /Users/ngirard/Documents/ASE/mimir
source .venv/bin/activate
pip install numpy pdfplumber python-docx python-pptx pytest pytest-asyncio
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt db.py tests/ pytest.ini
git commit -m "feat: add corpus_docs schema, test fixtures, new dependencies"
```

---

## Task 2: corpus.py — storage and search

**Files:**
- Create: `corpus.py`
- Create: `tests/test_corpus.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_corpus.py`:

```python
import numpy as np
import pytest
import corpus


@pytest.mark.asyncio
async def test_store_and_list(tmp_db):
    emb = np.ones(768, dtype=np.float32)
    doc_id = await corpus.store_doc(tmp_db, "Doc A", "a.txt", "hello world", emb)
    assert doc_id == 1
    docs = await corpus.list_docs(tmp_db)
    assert len(docs) == 1
    assert docs[0]["title"] == "Doc A"
    assert docs[0]["active"] is True


@pytest.mark.asyncio
async def test_duplicate_content_raises(tmp_db):
    emb = np.ones(768, dtype=np.float32)
    await corpus.store_doc(tmp_db, "Doc A", "a.txt", "hello world", emb)
    with pytest.raises(Exception):
        await corpus.store_doc(tmp_db, "Doc A2", "a2.txt", "hello world", emb)


@pytest.mark.asyncio
async def test_load_and_search(tmp_db):
    emb_a = np.array([1.0] + [0.0] * 767, dtype=np.float32)
    emb_b = np.array([0.0, 1.0] + [0.0] * 766, dtype=np.float32)
    await corpus.store_doc(tmp_db, "A", "a.txt", "content a", emb_a)
    await corpus.store_doc(tmp_db, "B", "b.txt", "content b", emb_b)

    docs = await corpus.load_corpus(tmp_db)
    assert len(docs) == 2

    query = np.array([1.0] + [0.0] * 767, dtype=np.float32)
    results = corpus.search_corpus(query, docs, k=1)
    assert results[0]["title"] == "A"
    assert results[0]["score"] > 0.99


@pytest.mark.asyncio
async def test_set_inactive_excluded_from_load(tmp_db):
    emb = np.ones(768, dtype=np.float32)
    doc_id = await corpus.store_doc(tmp_db, "Doc", "d.txt", "text", emb)
    await corpus.set_doc_active(tmp_db, doc_id, False)
    docs = await corpus.load_corpus(tmp_db)
    assert len(docs) == 0


@pytest.mark.asyncio
async def test_delete_doc(tmp_db):
    emb = np.ones(768, dtype=np.float32)
    doc_id = await corpus.store_doc(tmp_db, "Doc", "d.txt", "text", emb)
    await corpus.delete_doc(tmp_db, doc_id)
    docs = await corpus.list_docs(tmp_db)
    assert len(docs) == 0


def test_search_corpus_empty():
    results = corpus.search_corpus(np.ones(768, dtype=np.float32), [], k=5)
    assert results == []


def test_synthesis_prompt_no_corpus():
    prompt = corpus.build_synthesis_user_prompt("transcript text", [])
    assert "transcript text" in prompt
    assert "CORPUS" not in prompt


def test_synthesis_prompt_with_corpus():
    passages = [{"title": "Study A", "content": "key finding here"}]
    prompt = corpus.build_synthesis_user_prompt("transcript text", passages)
    assert "Study A" in prompt
    assert "CORPUS" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/ngirard/Documents/ASE/mimir
source .venv/bin/activate
pytest tests/test_corpus.py -v
```

Expected: `ModuleNotFoundError: No module named 'corpus'`

- [ ] **Step 3: Create corpus.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_corpus.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add corpus.py tests/test_corpus.py
git commit -m "feat: corpus storage, search, prompt helpers"
```

---

## Task 3: ingest_corpus.py — CLI ingestion script

**Files:**
- Create: `ingest_corpus.py`

- [ ] **Step 1: Create ingest_corpus.py**

```python
#!/usr/bin/env python3
"""
One-shot corpus ingestion for ASE workshop facilitator.
Usage: python ingest_corpus.py --dir /path/to/docs [--db livemind.db] [--chunk 2000] [--overlap 200]

Supported: .txt .md .pdf .docx .pptx
Idempotent: skips already-ingested content (SHA-256 hash check).
"""

import argparse, asyncio, sys, os, hashlib
from pathlib import Path

import aiosqlite
import corpus


def _parse_args():
    p = argparse.ArgumentParser(description="Ingest documents into ASE corpus.")
    p.add_argument("--dir", required=True, help="Directory of documents to ingest")
    p.add_argument("--db", default="livemind.db", help="Path to livemind.db")
    p.add_argument("--chunk", type=int, default=2000, help="Chunk size in chars")
    p.add_argument("--overlap", type=int, default=200, help="Overlap in chars")
    return p.parse_args()


def extract_text(path: Path) -> str:
    """Extract plain text from supported file types."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix == ".pptx":
        from pptx import Presentation
        prs = Presentation(path)
        parts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    parts.append(shape.text_frame.text)
        return "\n".join(parts)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]  # drop tiny tail chunks


async def ingest_file(db: aiosqlite.Connection, path: Path, chunk_size: int, overlap: int):
    print(f"  Processing {path.name}...", end="", flush=True)
    try:
        text = extract_text(path)
    except Exception as e:
        print(f" ERROR: {e}")
        return

    chunks = chunk_text(text, chunk_size, overlap)
    stored, skipped = 0, 0
    for i, chunk in enumerate(chunks):
        emb = await corpus.embed_text(chunk)
        if emb is None:
            print(f"\n  WARNING: embedding unavailable, storing chunk without embedding")
            import numpy as np
            emb = np.zeros(corpus.EMBED_DIM, dtype=np.float32)
        title = f"{path.stem} [{i+1}/{len(chunks)}]"
        try:
            await corpus.store_doc(db, title, path.name, chunk, emb)
            stored += 1
        except Exception:
            skipped += 1  # duplicate
    print(f" {stored} chunks stored, {skipped} skipped (duplicates)")


async def main():
    args = _parse_args()
    doc_dir = Path(args.dir)
    if not doc_dir.is_dir():
        print(f"ERROR: {doc_dir} is not a directory")
        sys.exit(1)

    supported = {".txt", ".md", ".pdf", ".docx", ".pptx"}
    files = [f for f in doc_dir.iterdir() if f.suffix.lower() in supported]
    if not files:
        print(f"No supported files found in {doc_dir}")
        sys.exit(0)

    print(f"Found {len(files)} files in {doc_dir}")
    print(f"DB: {args.db}")

    async with aiosqlite.connect(args.db) as db:
        # Ensure table exists
        await db.execute("""
            CREATE TABLE IF NOT EXISTS corpus_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
                source TEXT, content TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE, embedding BLOB NOT NULL,
                created_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
            )
        """)
        await db.commit()
        for f in sorted(files):
            await ingest_file(db, f, args.chunk, args.overlap)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Test with a sample file**

```bash
cd /Users/ngirard/Documents/ASE/mimir
source .venv/bin/activate
echo "Test document content about positioning and value proposition." > /tmp/test_doc.txt
python ingest_corpus.py --dir /tmp --db /tmp/test_corpus.db
```

Expected output:
```
Found 1 files in /tmp
DB: /tmp/test_corpus.db
  Processing test_doc.txt... 1 chunks stored, 0 skipped (duplicates)

Done.
```

- [ ] **Step 3: Verify idempotency**

```bash
python ingest_corpus.py --dir /tmp --db /tmp/test_corpus.db
```

Expected: `0 chunks stored, 1 skipped (duplicates)`

- [ ] **Step 4: Copy Capgemini template to assets/**

```bash
mkdir -p assets
cp /Users/ngirard/Downloads/template_cap_pptx.pptx assets/template_cap_pptx.pptx
```

- [ ] **Step 5: Commit**

```bash
git add ingest_corpus.py assets/
git commit -m "feat: one-shot corpus ingestion script + Capgemini PPTX template"
```

---

## Task 4: routes_facilitator.py — core LLM logic

**Files:**
- Create: `routes_facilitator.py`
- Create: `tests/test_synthesis.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_synthesis.py`:

```python
import json, re
import pytest
import corpus


def test_synthesis_prompt_contains_transcript():
    prompt = corpus.build_synthesis_user_prompt("people discussed positioning", [])
    assert "people discussed positioning" in prompt


def test_synthesis_prompt_with_passages():
    passages = [{"title": "McKinsey Study", "content": "Market is large"}]
    prompt = corpus.build_synthesis_user_prompt("live content", passages)
    assert "McKinsey Study" in prompt
    assert "CORPUS CONTEXT" in prompt
    assert "do not override" in prompt


def test_qa_prompt():
    prompt = corpus.build_qa_user_prompt("transcript here", "Where is the disagreement?", [])
    assert "transcript here" in prompt
    assert "Where is the disagreement?" in prompt


def test_synthesis_system_prompt_has_json_format():
    assert '"themes"' in corpus.SYNTHESIS_SYSTEM_PROMPT
    assert "JSON only" in corpus.SYNTHESIS_SYSTEM_PROMPT


def test_parse_synthesis_json_valid():
    """Test that valid LLM output can be parsed."""
    raw = '{"themes": [{"name": "Positioning", "alignment": ["a"], "disagreement": ["b"], "unresolved": ["c"]}]}'
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    assert match is not None
    data = json.loads(match.group())
    assert data["themes"][0]["name"] == "Positioning"


def test_parse_synthesis_json_with_preamble():
    """LLM sometimes adds preamble before JSON — regex should still find it."""
    raw = 'Here is the synthesis:\n{"themes": [{"name": "X", "alignment": [], "disagreement": [], "unresolved": []}]}'
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    assert match is not None
    data = json.loads(match.group())
    assert len(data["themes"]) == 1
```

- [ ] **Step 2: Run tests to verify they pass (these test corpus.py, already implemented)**

```bash
pytest tests/test_synthesis.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 3: Create routes_facilitator.py**

```python
"""
FastAPI routes for ASE Workshop Facilitator view.
Mount via: app.include_router(routes_facilitator.router)
Configure via: routes_facilitator.configure(...)
"""

import json, re, time, io, os, datetime
import asyncio
import aiohttp
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import db as db_module
import corpus as corpus_module

router = APIRouter()

# ─── State injected by app.py at startup ────────────────────────────────────
_get_session_id = lambda: None
_broadcast = None
_get_llm_chain = lambda: []
_get_db_conn = lambda: None  # returns the shared aiosqlite.Connection (db._db)

# ─── Provider config (mirrors app.py) ────────────────────────────────────────
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL  = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
HUGIN_BASE_URL      = os.environ.get("HUGIN_BASE_URL", "https://munin.btrbot.com")
HUGIN_CF_ID         = os.environ.get("HUGIN_CF_ID", "")
HUGIN_CF_SECRET     = os.environ.get("HUGIN_CF_SECRET", "")
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL     = "https://generativelanguage.googleapis.com/v1beta/openai"


def configure(get_session_id, broadcast, get_llm_chain, get_db_conn):
    """Call from app.py lifespan to inject shared state."""
    global _get_session_id, _broadcast, _get_llm_chain, _get_db_conn
    _get_session_id = get_session_id
    _broadcast = broadcast
    _get_llm_chain = get_llm_chain
    _get_db_conn = get_db_conn


# ─── Facilitator HTML view ───────────────────────────────────────────────────

@router.get("/facilitator", response_class=FileResponse)
async def facilitator_view():
    return FileResponse("facilitator.html")


@router.get("/corpus-admin", response_class=FileResponse)
async def corpus_admin_view():
    return FileResponse("corpus.html")


# ─── Live synthesis ──────────────────────────────────────────────────────────

@router.post("/v1/live-synthesis")
async def trigger_synthesis():
    result = await _run_synthesis()
    return JSONResponse(result)


async def _run_synthesis() -> dict:
    """Core synthesis. Called by route and background task."""
    session_id = _get_session_id()
    if not session_id:
        return {"error": "no_active_session", "themes": []}

    segments = await db_module.get_session_transcript(session_id)
    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    transcript_excerpt = full_text[-16000:]
    if len(transcript_excerpt) < 100:
        return {"error": "transcript_too_short", "themes": []}

    # Corpus context
    corpus_passages: list[dict] = []
    conn = _get_db_conn()
    if conn is not None:
        loaded = await corpus_module.load_corpus(conn)
        if loaded:
            query_emb = await corpus_module.embed_text(transcript_excerpt[-3000:])
            if query_emb is not None:
                corpus_passages = corpus_module.search_corpus(query_emb, loaded, k=5)

    user_prompt = corpus_module.build_synthesis_user_prompt(transcript_excerpt, corpus_passages)
    chain = _get_llm_chain()
    synthesis = await _call_synthesis_llm(corpus_module.SYNTHESIS_SYSTEM_PROMPT, user_prompt, chain)

    message = {
        "type": "live_synthesis",
        "session_id": session_id,
        "generated_at": time.time(),
        **synthesis,
    }
    if _broadcast:
        await _broadcast(json.dumps(message))
    return synthesis


# ─── Q&A ─────────────────────────────────────────────────────────────────────

@router.post("/v1/qa")
async def qa(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "No question"}, status_code=400)

    session_id = _get_session_id()
    if not session_id:
        return JSONResponse({"error": "No active session"}, status_code=400)

    segments = await db_module.get_session_transcript(session_id)
    transcript = " ".join(s["text"] for s in segments if s.get("text"))[-16000:]

    corpus_passages: list[dict] = []
    conn = _get_db_conn()
    if conn is not None:
        loaded = await corpus_module.load_corpus(conn)
        if loaded:
            emb = await corpus_module.embed_text(question)
            if emb is not None:
                corpus_passages = corpus_module.search_corpus(emb, loaded, k=3)

    user_prompt = corpus_module.build_qa_user_prompt(transcript, question, corpus_passages)
    chain = _get_llm_chain()
    answer = await _call_qa_llm(corpus_module.QA_SYSTEM_PROMPT, user_prompt, chain)
    return JSONResponse({"answer": answer})


# ─── Corpus CRUD ─────────────────────────────────────────────────────────────

@router.get("/v1/corpus")
async def list_corpus():
    conn = _get_db_conn()
    if conn is None:
        return JSONResponse({"docs": []})
    docs = await corpus_module.list_docs(conn)
    return JSONResponse({"docs": docs})


@router.patch("/v1/corpus/{doc_id}")
async def toggle_corpus_doc(doc_id: int, request: Request):
    body = await request.json()
    active = bool(body.get("active", True))
    conn = _get_db_conn()
    if conn is None:
        return JSONResponse({"error": "no db"}, status_code=500)
    await corpus_module.set_doc_active(conn, doc_id, active)
    return JSONResponse({"ok": True})


@router.delete("/v1/corpus/{doc_id}")
async def delete_corpus_doc(doc_id: int):
    conn = _get_db_conn()
    if conn is None:
        return JSONResponse({"error": "no db"}, status_code=500)
    await corpus_module.delete_doc(conn, doc_id)
    return JSONResponse({"ok": True})


# ─── PPTX export ─────────────────────────────────────────────────────────────

@router.post("/v1/sessions/{session_id}/export/synthesis-pptx")
async def export_synthesis_pptx(session_id: str):
    from pptx import Presentation

    segments = await db_module.get_session_transcript(session_id)
    full_text = " ".join(s["text"] for s in segments if s.get("text"))
    if not full_text:
        return JSONResponse({"error": "No transcript for session"}, status_code=404)

    user_prompt = corpus_module.build_synthesis_user_prompt(full_text[-16000:], [])
    chain = _get_llm_chain()
    synthesis = await _call_synthesis_llm(corpus_module.SYNTHESIS_SYSTEM_PROMPT, user_prompt, chain)
    themes = synthesis.get("themes", [])

    template_path = os.path.join(os.path.dirname(__file__), "assets", "template_cap_pptx.pptx")
    prs = Presentation(template_path)

    # Cover slide (layout 21 = "Title Subtitle")
    cover = prs.slides.add_slide(prs.slide_layouts[21])
    cover.placeholders[0].text = "Synthèse Workshop ASE"
    cover.placeholders[22].text = datetime.date.today().strftime("%d/%m/%Y")

    # One slide per theme (layout 35 = "Content 3 Boxes")
    boxes_layout = prs.slide_layouts[35]
    for theme in themes:
        slide = prs.slides.add_slide(boxes_layout)
        slide.placeholders[0].text = theme.get("name", "")

        def fill_box(ph_idx: int, items: list[str], header: str):
            tf = slide.placeholders[ph_idx].text_frame
            tf.clear()
            tf.text = header
            for item in items:
                p = tf.add_paragraph()
                p.text = f"• {item}"

        fill_box(22, theme.get("alignment", []), "✓ Alignement")
        fill_box(35, theme.get("disagreement", []), "✗ Désaccord")
        fill_box(36, theme.get("unresolved", []), "? Non tranché")

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    date_str = datetime.date.today().strftime("%Y%m%d")
    filename = f"synthesis-{date_str}.pptx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─── LLM helpers ─────────────────────────────────────────────────────────────

async def _call_synthesis_llm(system: str, user: str, chain: list[dict]) -> dict:
    for tier in chain:
        try:
            raw = await _llm_call(tier, system, user)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            continue
    return {"themes": [], "error": "llm_unavailable"}


async def _call_qa_llm(system: str, user: str, chain: list[dict]) -> str:
    for tier in chain:
        try:
            return await _llm_call(tier, system, user)
        except Exception:
            continue
    return "LLM indisponible."


async def _llm_call(tier: dict, system: str, user: str) -> str:
    provider = tier["provider"]
    model = tier["model"]
    timeout = aiohttp.ClientTimeout(total=60)

    if provider == "anthropic":
        url = f"{ANTHROPIC_BASE_URL}/v1/messages"
        headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
        if ANTHROPIC_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {ANTHROPIC_AUTH_TOKEN}"
        else:
            headers["x-api-key"] = ANTHROPIC_API_KEY
        body = {"model": model, "max_tokens": 2048, "system": system,
                "messages": [{"role": "user", "content": user}]}
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=body, timeout=timeout) as r:
                data = await r.json()
                return data["content"][0]["text"]

    # Hugin (Ollama) or Gemini — both use OpenAI-compatible /chat/completions
    if provider == "hugin":
        url = f"{HUGIN_BASE_URL}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if HUGIN_CF_ID and HUGIN_CF_SECRET:
            headers["CF-Access-Client-Id"] = HUGIN_CF_ID
            headers["CF-Access-Client-Secret"] = HUGIN_CF_SECRET
    else:  # gemini
        url = f"{GEMINI_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}

    body = {
        "model": model, "max_tokens": 2048,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers, json=body, timeout=timeout) as r:
            data = await r.json()
            return data["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Commit**

```bash
git add routes_facilitator.py tests/test_synthesis.py
git commit -m "feat: routes_facilitator with synthesis, Q&A, corpus CRUD, PPTX export"
```

---

## Task 5: PPTX export test

**Files:**
- Create: `tests/test_pptx.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pptx.py`:

```python
import io, datetime
import pytest
from pptx import Presentation
from pptx.util import Pt


TEMPLATE_PATH = "assets/template_cap_pptx.pptx"
THEMES = [
    {
        "name": "Positioning",
        "alignment": ["Clear market leader position", "Enterprise focus agreed"],
        "disagreement": ["SMB vs Enterprise priority split"],
        "unresolved": ["Geographic expansion timing"],
    },
    {
        "name": "Value Proposition",
        "alignment": ["Cost reduction is primary benefit"],
        "disagreement": [],
        "unresolved": ["Secondary benefits not ranked"],
    },
]


def _generate_pptx(themes: list[dict]) -> Presentation:
    """Helper: generate PPTX from themes list (mirrors route logic)."""
    prs = Presentation(TEMPLATE_PATH)

    cover = prs.slides.add_slide(prs.slide_layouts[21])
    cover.placeholders[0].text = "Synthèse Workshop ASE"
    cover.placeholders[22].text = datetime.date.today().strftime("%d/%m/%Y")

    boxes_layout = prs.slide_layouts[35]
    for theme in themes:
        slide = prs.slides.add_slide(boxes_layout)
        slide.placeholders[0].text = theme.get("name", "")

        def fill_box(ph_idx, items, header):
            tf = slide.placeholders[ph_idx].text_frame
            tf.clear()
            tf.text = header
            for item in items:
                p = tf.add_paragraph()
                p.text = f"• {item}"

        fill_box(22, theme.get("alignment", []), "✓ Alignement")
        fill_box(35, theme.get("disagreement", []), "✗ Désaccord")
        fill_box(36, theme.get("unresolved", []), "? Non tranché")

    return prs


def test_pptx_slide_count():
    prs = _generate_pptx(THEMES)
    # 1 cover + 1 per theme
    assert len(prs.slides) == 1 + len(THEMES)


def test_pptx_cover_title():
    prs = _generate_pptx(THEMES)
    cover = prs.slides[0]
    assert cover.placeholders[0].text == "Synthèse Workshop ASE"


def test_pptx_theme_title():
    prs = _generate_pptx(THEMES)
    theme_slide = prs.slides[1]
    assert theme_slide.placeholders[0].text == "Positioning"


def test_pptx_alignment_box():
    prs = _generate_pptx(THEMES)
    theme_slide = prs.slides[1]
    box_text = theme_slide.placeholders[22].text_frame.text
    assert "Alignement" in box_text
    assert "Clear market leader" in box_text


def test_pptx_is_valid_pptx():
    """Check the file can be saved and re-read."""
    prs = _generate_pptx(THEMES)
    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    prs2 = Presentation(buf)
    assert len(prs2.slides) == 1 + len(THEMES)


def test_pptx_empty_themes():
    prs = _generate_pptx([])
    assert len(prs.slides) == 1  # cover only
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_pptx.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pptx.py
git commit -m "test: PPTX generation unit tests"
```

---

## Task 6: app.py integration — mount router + background task

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add import and configure call to app.py**

At the top of `app.py`, after existing imports, add:

```python
import routes_facilitator
```

- [ ] **Step 2: Add configure() call and router mount**

In `app.py`, find the lifespan function (decorated with `@asynccontextmanager`). After `await db.init_db()` is called and after `manager` is defined, add:

```python
    routes_facilitator.configure(
        get_session_id=lambda: _current_session_id,
        broadcast=manager.broadcast,
        get_llm_chain=lambda: list(_llm_chain),
        get_db_conn=lambda: db._db,
    )
```

Then find where `app = FastAPI(...)` is defined (or just after the lifespan setup) and add:

```python
app.include_router(routes_facilitator.router)
```

- [ ] **Step 3: Add synthesis background task**

In the lifespan function, after `asyncio.create_task(broadcast_loop())`, add:

```python
    asyncio.create_task(_synthesis_loop())
```

Then add this function at module level in `app.py` (near `broadcast_loop`):

```python
async def _synthesis_loop():
    """Auto-generate synthesis every 5 minutes when a session is active."""
    while True:
        await asyncio.sleep(300)
        if _current_session_id:
            try:
                await routes_facilitator._run_synthesis()
            except Exception as e:
                print(f"[synthesis_loop] error: {e}", flush=True)
```

- [ ] **Step 4: Start server and verify routes exist**

```bash
cd /Users/ngirard/Documents/ASE/mimir
source .venv/bin/activate
python app.py --host 0.0.0.0 --port 8765 &
sleep 3
curl -s http://localhost:8765/facilitator | head -5
curl -s -X POST http://localhost:8765/v1/live-synthesis | python3 -m json.tool
curl -s http://localhost:8765/v1/corpus | python3 -m json.tool
kill %1
```

Expected:
- `/facilitator` returns HTML (404 is ok if facilitator.html not yet created — route exists)
- `/v1/live-synthesis` returns `{"error": "no_active_session", "themes": []}`
- `/v1/corpus` returns `{"docs": []}`

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: mount facilitator router + synthesis background task in app.py"
```

---

## Task 7: facilitator.html — facilitator view

**Files:**
- Create: `facilitator.html`

- [ ] **Step 1: Create facilitator.html**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASE Facilitator — Mímir</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Ubuntu:ital,wght@0,300;0,400;0,500;0,700;1,400&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --brand-blue: #0058AB;
  --brand-dark: #121A38;
  --brand-light: #1DB8F2;
  --white: #FFFFFF;
  --turquoise: #00D5D0;
  --bg: #0d1124;
  --panel: #111827;
  --card: rgba(255,255,255,0.04);
  --bdr: rgba(255,255,255,0.08);
  --txt: #d4d6e0;
  --dim: #6b7280;
}

body {
  background: var(--bg);
  color: var(--txt);
  font-family: 'Ubuntu', Verdana, sans-serif;
  font-weight: 300;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Top bar ── */
#topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 20px;
  border-bottom: 1px solid var(--bdr);
  background: var(--panel);
  flex-shrink: 0;
}
#topbar .session-label {
  font-weight: 500;
  font-size: 13px;
  color: var(--white);
  flex: 1;
}
.live-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #34d399;
}
.live-dot { width: 8px; height: 8px; background: #34d399; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

.btn {
  padding: 6px 14px;
  border: none;
  border-radius: 4px;
  font-family: 'Ubuntu', sans-serif;
  font-weight: 500;
  font-size: 12px;
  cursor: pointer;
  transition: opacity .15s;
}
.btn:hover { opacity: .85; }
.btn-primary { background: var(--brand-blue); color: var(--white); }
.btn-secondary { background: var(--card); color: var(--txt); border: 1px solid var(--bdr); }
.btn-sm { padding: 4px 10px; font-size: 11px; }

/* brand gradient separator */
.gradient-bar {
  height: 2px;
  background: linear-gradient(90deg, #121A38 0%, #0058AB 30%, #1DB8F2 70%, #00D5D0 100%);
  flex-shrink: 0;
}

/* ── Main layout ── */
#main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  flex: 1;
  overflow: hidden;
}

/* ── Transcript panel ── */
#transcript-panel {
  border-right: 1px solid var(--bdr);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
#transcript-header {
  padding: 10px 16px;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--dim);
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--bdr);
}
#transcript-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  font-size: 13px;
  line-height: 1.7;
}
.seg { margin-bottom: 8px; }
.seg .ts { color: var(--brand-light); font-family: monospace; font-size: 11px; margin-right: 8px; }

/* ── Right panel ── */
#right-panel {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Synthesis */
#synthesis-panel {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}
#synthesis-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
#synthesis-header h2 {
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--dim);
}
#synthesis-age { font-size: 11px; color: var(--dim); }
#synthesis-spinner { display: none; font-size: 11px; color: var(--brand-light); }

.theme-card {
  background: var(--card);
  border: 1px solid var(--bdr);
  border-radius: 6px;
  margin-bottom: 12px;
  overflow: hidden;
}
.theme-title {
  padding: 10px 14px;
  font-weight: 500;
  font-size: 13px;
  color: var(--white);
  background: rgba(0,88,171,0.15);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.theme-body { padding: 10px 14px; }
.theme-section { margin-bottom: 8px; }
.theme-section-label {
  font-size: 11px;
  font-weight: 500;
  margin-bottom: 4px;
}
.label-align { color: #34d399; }
.label-disagree { color: #fb7185; }
.label-unresolved { color: #fbbf24; }
.theme-section ul { list-style: none; padding: 0; }
.theme-section li { font-size: 12px; line-height: 1.6; padding-left: 12px; position: relative; }
.theme-section li::before { content: "·"; position: absolute; left: 0; color: var(--dim); }

#export-bar {
  padding: 10px 16px;
  display: flex;
  gap: 8px;
  border-top: 1px solid var(--bdr);
  flex-shrink: 0;
}

/* Q&A */
#qa-panel {
  border-top: 1px solid var(--bdr);
  flex-shrink: 0;
  max-height: 280px;
  display: flex;
  flex-direction: column;
}
#qa-header {
  padding: 8px 16px;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--dim);
  border-bottom: 1px solid var(--bdr);
}
#qa-history {
  flex: 1;
  overflow-y: auto;
  padding: 10px 16px;
  font-size: 12px;
  line-height: 1.6;
}
.qa-q { color: var(--brand-light); margin-bottom: 4px; font-weight: 500; }
.qa-a { color: var(--txt); margin-bottom: 12px; white-space: pre-wrap; }
#qa-input-row {
  display: flex;
  gap: 8px;
  padding: 10px 16px;
  border-top: 1px solid var(--bdr);
}
#qa-input {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--bdr);
  border-radius: 4px;
  color: var(--white);
  font-family: 'Ubuntu', sans-serif;
  font-size: 13px;
  padding: 6px 10px;
  outline: none;
}
#qa-input:focus { border-color: var(--brand-blue); }

/* LLM provider badge */
#provider-badge {
  font-size: 10px;
  color: var(--dim);
  margin-left: auto;
  font-family: monospace;
}
</style>
</head>
<body>

<div id="topbar">
  <div class="session-label" id="session-label">Connexion...</div>
  <div class="live-badge" id="live-badge" style="display:none">
    <div class="live-dot"></div> LIVE
  </div>
  <button class="btn btn-secondary btn-sm" onclick="synthNow()">Synthétiser</button>
  <span id="provider-badge"></span>
</div>
<div class="gradient-bar"></div>

<div id="main">

  <!-- Transcript -->
  <div id="transcript-panel">
    <div id="transcript-header">
      <span>Transcript</span>
      <button class="btn btn-secondary btn-sm" onclick="copyTranscript()">Copier tout</button>
    </div>
    <div id="transcript-scroll"></div>
  </div>

  <!-- Right: synthesis + Q&A -->
  <div id="right-panel">
    <div id="synthesis-panel">
      <div id="synthesis-header">
        <h2>Synthèse thématique</h2>
        <span id="synthesis-age"></span>
        <span id="synthesis-spinner">⟳ génération...</span>
      </div>
      <div id="themes-container">
        <div style="color:var(--dim);font-size:12px;">En attente du transcript...</div>
      </div>
    </div>

    <div id="export-bar">
      <button class="btn btn-secondary btn-sm" onclick="copyAllMarkdown()">Copier en markdown</button>
      <button class="btn btn-primary btn-sm" onclick="exportPptx()">Exporter PPTX</button>
    </div>

    <div id="qa-panel">
      <div id="qa-header">Q&A Facilitateur</div>
      <div id="qa-history"></div>
      <div id="qa-input-row">
        <input id="qa-input" type="text" placeholder="Où est le désaccord ? Qu'est-ce qui n'a pas été tranché ?"
               onkeydown="if(event.key==='Enter') askQuestion()">
        <button class="btn btn-primary btn-sm" onclick="askQuestion()">↵</button>
      </div>
    </div>
  </div>
</div>

<script>
// ── State ────────────────────────────────────────────────────────────────────
let ws = null;
let currentSessionId = null;
let currentThemes = [];
let transcriptSegments = [];
let synthLastAt = null;

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    document.getElementById('live-badge').style.display = 'flex';
  };
  ws.onclose = () => {
    document.getElementById('live-badge').style.display = 'none';
    setTimeout(connect, 3000);
  };
  ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
  setInterval(() => ws && ws.readyState === 1 && ws.send(JSON.stringify({type:'ping'})), 20000);
}

function handleMessage(msg) {
  if (msg.type === 'transcript' || msg.type === 'partial_transcript') {
    appendTranscript(msg);
  } else if (msg.type === 'live_synthesis') {
    renderSynthesis(msg);
  } else if (msg.type === 'restore') {
    if (msg.segments) msg.segments.forEach(appendTranscript);
  } else if (msg.type === 'status') {
    document.getElementById('session-label').textContent = msg.message || 'Connecté';
    currentSessionId = msg.session_id || currentSessionId;
  } else if (msg.type === 'session_reset') {
    currentSessionId = msg.session_id;
    transcriptSegments = [];
    document.getElementById('transcript-scroll').innerHTML = '';
  } else if (msg.type === 'metrics') {
    const p = msg.llm_provider || '';
    document.getElementById('provider-badge').textContent = p ? `LLM: ${p}` : '';
  }
}

// ── Transcript ────────────────────────────────────────────────────────────────
function appendTranscript(seg) {
  if (!seg.text || !seg.text.trim()) return;
  transcriptSegments.push(seg);
  const container = document.getElementById('transcript-scroll');
  const ts = seg.timestamp ? new Date(seg.timestamp * 1000).toLocaleTimeString('fr', {hour:'2-digit',minute:'2-digit'}) : '';
  const div = document.createElement('div');
  div.className = 'seg';
  div.innerHTML = `<span class="ts">${ts}</span>${escHtml(seg.text)}`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function copyTranscript() {
  const lines = transcriptSegments.map(s => {
    const ts = s.timestamp ? new Date(s.timestamp*1000).toLocaleTimeString('fr',{hour:'2-digit',minute:'2-digit'}) : '';
    return `[${ts}] ${s.text}`;
  });
  navigator.clipboard.writeText(lines.join('\n'));
}

// ── Synthesis ─────────────────────────────────────────────────────────────────
function renderSynthesis(msg) {
  currentThemes = msg.themes || [];
  synthLastAt = msg.generated_at;
  document.getElementById('synthesis-spinner').style.display = 'none';
  document.getElementById('synthesis-age').textContent = 'À l\'instant';
  updateAgeDisplay();

  const container = document.getElementById('themes-container');
  if (!currentThemes.length) {
    container.innerHTML = '<div style="color:var(--dim);font-size:12px;">Aucun thème extrait.</div>';
    return;
  }
  container.innerHTML = '';
  currentThemes.forEach((theme, i) => {
    const card = document.createElement('div');
    card.className = 'theme-card';
    card.innerHTML = `
      <div class="theme-title">
        <span>${escHtml(theme.name)}</span>
        <button class="btn btn-secondary btn-sm" onclick="copyTheme(${i})">Copier</button>
      </div>
      <div class="theme-body">
        ${renderSection(theme.alignment, '✓ Alignement', 'label-align')}
        ${renderSection(theme.disagreement, '✗ Désaccord', 'label-disagree')}
        ${renderSection(theme.unresolved, '? Non tranché', 'label-unresolved')}
      </div>`;
    container.appendChild(card);
  });
}

function renderSection(items, label, cls) {
  if (!items || !items.length) return '';
  const lis = items.map(i => `<li>${escHtml(i)}</li>`).join('');
  return `<div class="theme-section">
    <div class="theme-section-label ${cls}">${label}</div>
    <ul>${lis}</ul>
  </div>`;
}

function updateAgeDisplay() {
  if (!synthLastAt) return;
  const mins = Math.round((Date.now()/1000 - synthLastAt) / 60);
  document.getElementById('synthesis-age').textContent = mins < 1 ? 'À l\'instant' : `il y a ${mins} min`;
}
setInterval(updateAgeDisplay, 30000);

async function synthNow() {
  document.getElementById('synthesis-spinner').style.display = 'inline';
  try {
    const r = await fetch('/v1/live-synthesis', {method:'POST'});
    const data = await r.json();
    if (data.themes) renderSynthesis({themes: data.themes, generated_at: Date.now()/1000});
  } finally {
    document.getElementById('synthesis-spinner').style.display = 'none';
  }
}

// ── Export ────────────────────────────────────────────────────────────────────
function copyTheme(i) {
  const t = currentThemes[i];
  if (!t) return;
  const lines = [`## ${t.name}`, '',
    '**✓ Alignement**', ...(t.alignment||[]).map(l=>`- ${l}`), '',
    '**✗ Désaccord**',  ...(t.disagreement||[]).map(l=>`- ${l}`), '',
    '**? Non tranché**',...(t.unresolved||[]).map(l=>`- ${l}`),
  ];
  navigator.clipboard.writeText(lines.join('\n'));
}

function copyAllMarkdown() {
  const date = new Date().toLocaleDateString('fr');
  const parts = [`# Synthèse Workshop ASE — ${date}`, ''];
  currentThemes.forEach(t => {
    parts.push(`## ${t.name}`, '',
      '**✓ Alignement**', ...(t.alignment||[]).map(l=>`- ${l}`), '',
      '**✗ Désaccord**',  ...(t.disagreement||[]).map(l=>`- ${l}`), '',
      '**? Non tranché**',...(t.unresolved||[]).map(l=>`- ${l}`),
      '', '---', '');
  });
  navigator.clipboard.writeText(parts.join('\n'));
}

async function exportPptx() {
  if (!currentSessionId) { alert('Aucune session active'); return; }
  const r = await fetch(`/v1/sessions/${currentSessionId}/export/synthesis-pptx`, {method:'POST'});
  if (!r.ok) { alert('Erreur export'); return; }
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `synthesis-${new Date().toISOString().slice(0,10)}.pptx`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Q&A ───────────────────────────────────────────────────────────────────────
async function askQuestion() {
  const input = document.getElementById('qa-input');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  const history = document.getElementById('qa-history');
  history.innerHTML += `<div class="qa-q">> ${escHtml(q)}</div><div class="qa-a" id="qa-pending">...</div>`;
  history.scrollTop = history.scrollHeight;

  try {
    const r = await fetch('/v1/qa', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q}),
    });
    const data = await r.json();
    document.getElementById('qa-pending').textContent = data.answer || data.error || '(pas de réponse)';
    document.getElementById('qa-pending').removeAttribute('id');
    history.scrollTop = history.scrollHeight;
  } catch {
    document.getElementById('qa-pending').textContent = 'Erreur réseau.';
    document.getElementById('qa-pending').removeAttribute('id');
  }
}

// ── Utils ─────────────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

connect();
</script>
</body>
</html>
```

- [ ] **Step 2: Open in browser and verify**

With server running:
```bash
open http://localhost:8765/facilitator
```

Check:
- Page loads with Capgemini dark/blue theme
- WebSocket connects (LIVE badge appears)
- Transcript panel on left, synthesis + Q&A on right

- [ ] **Step 3: Commit**

```bash
git add facilitator.html
git commit -m "feat: facilitator.html — two-column workshop view with synthesis and Q&A"
```

---

## Task 8: corpus.html — corpus admin page

**Files:**
- Create: `corpus.html`

- [ ] **Step 1: Create corpus.html**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Corpus — ASE Facilitator</title>
<link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--brand-blue:#0058AB;--brand-dark:#121A38;--brand-light:#1DB8F2;--white:#fff;--bg:#0d1124;--panel:#111827;--card:rgba(255,255,255,.04);--bdr:rgba(255,255,255,.08);--txt:#d4d6e0;--dim:#6b7280}
body{background:var(--bg);color:var(--txt);font-family:'Ubuntu',Verdana,sans-serif;font-weight:300;padding:32px;min-height:100vh}
h1{font-size:20px;font-weight:500;color:var(--white);margin-bottom:4px}
.subtitle{font-size:13px;color:var(--dim);margin-bottom:24px}
a{color:var(--brand-light);text-decoration:none;font-size:13px}
.gradient-bar{height:2px;background:linear-gradient(90deg,#121A38 0%,#0058AB 30%,#1DB8F2 70%,#00D5D0 100%);margin-bottom:24px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-size:11px;font-weight:500;letter-spacing:2px;text-transform:uppercase;color:var(--dim);padding:8px 12px;border-bottom:1px solid var(--bdr)}
td{padding:10px 12px;border-bottom:1px solid var(--bdr);vertical-align:middle}
tr:hover td{background:var(--card)}
.btn{padding:4px 12px;border:none;border-radius:4px;font-family:'Ubuntu',sans-serif;font-size:11px;font-weight:500;cursor:pointer}
.btn-del{background:rgba(251,113,133,.15);color:#fb7185}
.btn-del:hover{background:rgba(251,113,133,.3)}
.toggle{cursor:pointer;accent-color:var(--brand-blue);width:16px;height:16px}
.empty{color:var(--dim);font-size:13px;padding:24px 0}
</style>
</head>
<body>
<a href="/facilitator">← Facilitateur</a>
<h1 style="margin-top:16px">Corpus documentaire</h1>
<p class="subtitle">Documents indexés pour enrichir la synthèse. Ingestion via <code>python ingest_corpus.py --dir /path/</code></p>
<div class="gradient-bar"></div>

<div id="content"><p class="empty">Chargement...</p></div>

<script>
async function load() {
  const r = await fetch('/v1/corpus');
  const {docs} = await r.json();
  const el = document.getElementById('content');
  if (!docs.length) { el.innerHTML = '<p class="empty">Aucun document ingéré.</p>'; return; }
  el.innerHTML = `<table>
    <thead><tr><th>Titre</th><th>Source</th><th>Ingéré le</th><th>Actif</th><th></th></tr></thead>
    <tbody>${docs.map(d => `
      <tr>
        <td>${esc(d.title)}</td>
        <td style="color:var(--dim)">${esc(d.source||'')}</td>
        <td style="color:var(--dim)">${new Date(d.created_at*1000).toLocaleDateString('fr')}</td>
        <td><input type="checkbox" class="toggle" ${d.active?'checked':''} onchange="toggle(${d.id},this.checked)"></td>
        <td><button class="btn btn-del" onclick="del(${d.id})">Supprimer</button></td>
      </tr>`).join('')}
    </tbody>
  </table>`;
}

async function toggle(id, active) {
  await fetch(`/v1/corpus/${id}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({active})});
}

async function del(id) {
  if (!confirm('Supprimer ce document du corpus ?')) return;
  await fetch(`/v1/corpus/${id}`, {method:'DELETE'});
  load();
}

function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;'); }

load();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify in browser**

```bash
open http://localhost:8765/corpus-admin
```

Expected: page loads, shows empty corpus message, link back to facilitator.

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add corpus.html
git commit -m "feat: corpus.html admin page"
```

- [ ] **Step 5: Copy plan to repo**

```bash
mkdir -p docs/superpowers/plans
cp /path/to/plan-file docs/superpowers/plans/2026-04-21-ase-workshop-facilitator.md
git add docs/superpowers/plans/
git commit -m "docs: add facilitator implementation plan"
```

---

## Verification

End-to-end manual test:

1. Start server: `python app.py --host 0.0.0.0 --port 8765`
2. Open `/monitor` on technician device, start audio capture
3. Open `/facilitator` — verify transcript appears in left panel
4. Click "Synthétiser" — verify themes appear in right panel with ✓/✗/? sections
5. Click "Copier en markdown" — paste in a text editor, verify format
6. Click "Exporter PPTX" — open file in PowerPoint, verify Capgemini template with theme slides
7. Type a question in Q&A box — verify answer appears
8. Wait 5 minutes — verify auto-synthesis triggers (badge updates)
9. Open `/corpus-admin` — verify page loads (empty or with ingested docs)
10. Run `python ingest_corpus.py --dir /path/to/docs` — re-open `/corpus-admin` — verify docs listed

---

## Notes

- `sqlite-vec` was replaced by in-memory numpy cosine similarity. Adequate for a fixed corpus of 50-500 chunks.
- If Ollama (Hugin) is unavailable, `embed_text()` returns `None` and synthesis proceeds without corpus context.
- PPTX layout indices (21 for cover, 35 for 3 boxes) were verified by inspecting `template_cap_pptx.pptx` directly.
- The WebSocket endpoint (`/ws`) is mimir's existing endpoint — no changes needed.
