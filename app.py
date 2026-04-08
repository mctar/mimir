#!/usr/bin/env python3
"""
Mímir — FastAPI Server
Real-time conversation visualization. Receives audio from browser,
dispatches to STT, proxies LLM calls, manages graph reconciliation.
"""

import asyncio, json, time, threading, queue, sys, argparse, os, uuid, base64
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import aiohttp

import db
import stt_worker
from stt_worker import configure_stt, get_stt_config
from reconciler import GraphReconciler

# ─── Load .env ───
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── Hugin (self-hosted Ollama) ───
HUGIN_BASE_URL = os.environ.get("HUGIN_BASE_URL", "https://munin.btrbot.com")
HUGIN_CF_ID = os.environ.get("HUGIN_CF_ID", "")
HUGIN_CF_SECRET = os.environ.get("HUGIN_CF_SECRET", "")

# ─── Gemini ───
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# ─── Remote STT ───
STT_SERVER_URL = os.environ.get("STT_SERVER_URL", "https://stt.btrbot.com")

# Active LLM config — mutable at runtime via admin panel
_active_llm = {
    "provider": "hugin",             # "anthropic" | "hugin" | "gemini"
    "model": "gemma4:26b",
}

# Extra system prompt for non-Claude models to improve graph quality
_SMALL_MODEL_GRAPH_PREFIX = """CRITICAL: Output ONLY the raw JSON object. No thinking, no reasoning, no explanation, no markdown fences.

GRAPH EVOLUTION (follow strictly):
- You MUST add new nodes for every new concept, person, or topic in the NEW SEGMENT
- Always evolve the graph — never return it unchanged. The conversation is progressing, the graph must too.
- Every node MUST connect to at least 2 different nodes — no orphans
- NEVER create a star/hub where all nodes link to one central node
- Create cross-connections between related concepts, not just to the main topic
- Vary relationship labels: "enables", "requires", "part of", "contrasts", "drives", "informs", "blocks"
- Use "group" field (not "type") for node category

"""
_active_llm_lock = threading.Lock()

# ─── Global state ───
transcript_queue: queue.Queue = queue.Queue()
connected_clients: set[WebSocket] = set()
client_sessions: dict[WebSocket, str] = {}  # ws → session_id

metrics = {
    "started_at": time.time(),
    "chunks_processed": 0,
    "chunks_skipped_silent": 0,
    "chunks_skipped_catchup": 0,
    "stt_last_duration": 0.0,
    "stt_avg_duration": 0.0,
    "stt_total_time": 0.0,
    "stt_last_text": "",
    "stt_empty_results": 0,
    "stt_partials_emitted": 0,
    "audio_buffer_seconds": 0.0,
    "audio_rms": 0.0,
    "tokenizer_recreations": 0,
    "tokenizer_last_ms": 0.0,
    "stt_e2e_last": 0.0,
    "stt_e2e_avg": 0.0,
    "stt_e2e_total": 0.0,
    "claude_calls": 0,
    "claude_errors": 0,
    "claude_last_duration": 0.0,
    "claude_avg_duration": 0.0,
    "claude_total_time": 0.0,
    "ws_clients": 0,
    "transcript_queue_size": 0,
    "chunk_seconds": 2,
    "cb_state": "closed",
    "cb_failures": 0,
    "vad_state": "silent",
    "ws_reconnects": 0,
    "last_restore_ms": 0.0,
    "frontend_fps": 0.0,
    "nodes_added_per_min": 0,
    "nodes_removed_per_min": 0,
    "edge_churn_per_min": 0,
    "analysis_queue_depth": 0,
    "claude_last_error": "",
}
metrics_lock = threading.Lock()

# Circuit breaker
cb_state = "closed"
cb_failures = 0
cb_backoff_until = 0.0
cb_backoff_secs = 5.0
CB_MAX_BACKOFF = 60.0
CB_FAILURE_THRESHOLD = 3
cb_lock = threading.Lock()

# Monotonic sequence counter for transcript messages
_seq_counter = 0
_seq_lock = threading.Lock()

# Graph reconciler
reconciler = GraphReconciler()
_current_session_id: str | None = None
_summary: str = ""


def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


# ─── Lifespan ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    # Default STT backend
    configure_stt("remote")
    print(f"  STT: faster-whisper (localhost:8766)")
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(snapshot_loop())
    print(f"  Server ready — audio arrives from browser via WebSocket")
    print(f"  Main:     http://0.0.0.0:{WS_PORT}/")
    print(f"  Monitor:  http://0.0.0.0:{WS_PORT}/monitor")
    print(f"  Sessions: http://0.0.0.0:{WS_PORT}/sessions\n")
    yield
    await db.close_db()


app = FastAPI(lifespan=lifespan)
WS_PORT = 8765


# ─── Static serving ───
@app.get("/mimir-favicon.svg")
async def serve_favicon():
    return FileResponse(os.path.join(os.path.dirname(__file__), "mimir-favicon.svg"), media_type="image/svg+xml")

@app.get("/mimir-logo-clean.svg")
async def serve_logo():
    return FileResponse(os.path.join(os.path.dirname(__file__), "mimir-logo-clean.svg"), media_type="image/svg+xml")

@app.get("/", response_class=HTMLResponse)
async def serve_main():
    return FileResponse(os.path.join(os.path.dirname(__file__), "live-mindmap.html"))


@app.get("/monitor", response_class=HTMLResponse)
async def serve_monitor():
    return FileResponse(os.path.join(os.path.dirname(__file__), "monitor.html"))


@app.get("/doc", response_class=HTMLResponse)
async def serve_doc():
    return FileResponse(os.path.join(os.path.dirname(__file__), "doc.html"))

@app.get("/sessions", response_class=HTMLResponse)
async def serve_sessions():
    return FileResponse(os.path.join(os.path.dirname(__file__), "sessions.html"))

@app.get("/sessions/archive", response_class=HTMLResponse)
async def serve_archive():
    return FileResponse(os.path.join(os.path.dirname(__file__), "sessions.html"))


# Legacy redirects
from starlette.responses import RedirectResponse

@app.get("/admin/sessions")
async def redirect_admin_sessions():
    return RedirectResponse("/sessions")

@app.get("/admin")
async def redirect_admin():
    return RedirectResponse("/sessions")


# ─── REST endpoints ───

@app.get("/v1/sessions")
async def list_sessions(archived: bool = False):
    sessions = await db.list_sessions(archived=archived)
    return JSONResponse(sessions)


@app.post("/v1/sessions/archive")
async def archive_sessions(request: Request):
    """Move sessions to archive."""
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if not session_ids:
        return JSONResponse({"error": "No session IDs provided"}, status_code=400)
    await db.archive_sessions(session_ids)
    return JSONResponse({"ok": True, "archived": len(session_ids)})


@app.post("/v1/sessions/unarchive")
async def unarchive_sessions(request: Request):
    """Move sessions out of archive."""
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if not session_ids:
        return JSONResponse({"error": "No session IDs provided"}, status_code=400)
    await db.unarchive_sessions(session_ids)
    return JSONResponse({"ok": True, "unarchived": len(session_ids)})


@app.post("/v1/sessions")
async def create_session(request: Request):
    body = await request.json()
    session_id = str(uuid.uuid4())[:8]
    topic = body.get("topic", "")
    session = await db.create_session(session_id, topic)
    global _current_session_id
    _current_session_id = session_id
    return JSONResponse(session)


@app.post("/v1/sessions/new")
async def new_session(request: Request):
    """End current session (if any) and start a fresh one. Returns the new session."""
    global _current_session_id, _summary, _seq_counter
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    topic = body.get("topic", "")
    # End current session
    if _current_session_id:
        if reconciler.nodes:
            await db.store_snapshot(_current_session_id, _seq_counter, reconciler.get_full_state(), "end")
        await db.end_session(_current_session_id, _summary)
    # Reset all state
    reconciler.nodes.clear()
    reconciler.edges.clear()
    reconciler._mention_log.clear()
    reconciler._churn_log.clear()
    _summary = ""
    with _seq_lock:
        _seq_counter = 0
    # Drain any leftover transcript messages from STT queue
    while not transcript_queue.empty():
        try:
            transcript_queue.get_nowait()
        except Exception:
            break
    # Reset session metrics
    with metrics_lock:
        keep = {"started_at", "ws_clients", "cb_state", "cb_failures"}
        for k, v in metrics.items():
            if k not in keep:
                if isinstance(v, (int, float)):
                    metrics[k] = 0 if isinstance(v, int) else 0.0
                elif isinstance(v, str):
                    metrics[k] = ""
        metrics["started_at"] = time.time()
    # Create new
    session_id = str(uuid.uuid4())[:8]
    session = await db.create_session(session_id, topic)
    _current_session_id = session_id
    # Notify all connected frontends
    msg = json.dumps({"type": "session_reset", "session_id": session_id, "topic": topic})
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass
    return JSONResponse(session)


@app.post("/v1/sessions/{session_id}/end")
async def end_session(session_id: str, request: Request):
    """End a session: flush final snapshot, store summary, reset reconciler."""
    global _current_session_id, _summary, _seq_counter
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    summary = body.get("summary", _summary)
    # Final snapshot
    if reconciler.nodes:
        await db.store_snapshot(session_id, _seq_counter, reconciler.get_full_state(), "end")
    await db.end_session(session_id, summary)
    # Reset server state
    reconciler.nodes.clear()
    reconciler.edges.clear()
    reconciler._mention_log.clear()
    reconciler._churn_log.clear()
    _summary = ""
    with _seq_lock:
        _seq_counter = 0
    while not transcript_queue.empty():
        try:
            transcript_queue.get_nowait()
        except Exception:
            break
    if _current_session_id == session_id:
        _current_session_id = None
    # Notify all connected frontends
    msg = json.dumps({"type": "session_ended", "session_id": session_id})
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass
    return JSONResponse({"ok": True, "session_id": session_id})


@app.get("/v1/sessions/synthesis")
async def list_synthesis():
    """List all cross-session synthesis recaps."""
    return JSONResponse(await db.list_synthesis_recaps())


@app.get("/v1/sessions/synthesis/{synthesis_id}")
async def get_synthesis(synthesis_id: int):
    """Get a single cross-session synthesis recap."""
    result = await db.get_synthesis_recap(synthesis_id)
    if not result:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(result)


@app.delete("/v1/sessions/synthesis/{synthesis_id}")
async def delete_synthesis(synthesis_id: int):
    """Delete a cross-session synthesis recap."""
    await db.delete_synthesis_recap(synthesis_id)
    return JSONResponse({"ok": True})


@app.get("/v1/sessions/{session_id}/restore")
async def restore_session(session_id: str, from_seq: int = 0):
    t0 = time.time()
    snapshot = await db.get_latest_snapshot(session_id)
    segments = await db.get_segments_since(session_id, from_seq)
    restore_ms = (time.time() - t0) * 1000
    with metrics_lock:
        metrics["last_restore_ms"] = restore_ms
    return JSONResponse({
        "snapshot": snapshot,
        "segments": segments,
        "restore_ms": round(restore_ms, 1),
    })


@app.post("/v1/sessions/{session_id}/actions")
async def session_action(session_id: str, request: Request):
    body = await request.json()
    action_type = body.get("action")
    payload = body.get("payload", {})
    await db.store_action(session_id, action_type, payload)
    graph = reconciler.apply_action(action_type, payload)
    # Broadcast updated graph to all connected clients
    msg = json.dumps({"type": "graph_update", "graph": graph, "session_id": _current_session_id})
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass
    return JSONResponse({"ok": True, "graph": graph})


@app.get("/v1/sessions/{session_id}/snapshots")
async def get_session_snapshots_endpoint(session_id: str):
    """Get all snapshots for playback."""
    snapshots = await db.get_session_snapshots(session_id)
    return JSONResponse({
        "session_id": session_id,
        "count": len(snapshots),
        "snapshots": snapshots,
    })


@app.get("/v1/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Get session detail: transcript, final snapshot, and recap."""
    transcript = await db.get_session_transcript(session_id)
    snapshot = await db.get_latest_snapshot(session_id)
    recap = await db.get_recap(session_id)
    # Also fetch session metadata
    sessions = await db.list_sessions()
    session_meta = next((s for s in sessions if s["id"] == session_id), None)
    return JSONResponse({
        "session": session_meta,
        "transcript": transcript,
        "snapshot": snapshot,
        "recap": recap,
    })


@app.post("/v1/sessions/{session_id}/recap")
async def generate_recap(session_id: str):
    """Generate an AI recap for a session (on-demand, server-side LLM call)."""
    segments = await db.get_session_transcript(session_id)
    if not segments:
        return JSONResponse({"error": "No transcript found"}, status_code=404)

    full_text = " ".join(seg.get("cleaned_text") or seg["text"] for seg in segments)

    # Detect session language from STT metadata (majority vote)
    lang_counts: dict[str, int] = {}
    for seg in segments:
        lang = seg.get("stt_language", "")
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    session_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "en"

    # Get final snapshot — full graph with nodes and edges
    snapshot = await db.get_latest_snapshot(session_id)
    graph_context = ""
    if snapshot and snapshot.get("graph"):
        graph = snapshot["graph"]
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", [])
        # Build structured graph description
        active_nodes = {nid: n for nid, n in nodes.items() if n.get("state") == "active"}
        if active_nodes:
            node_lines = [f"  - {n.get('label', nid)} (category: {n.get('group', 'unknown')})" for nid, n in active_nodes.items()]
            # Map node IDs to labels for edge descriptions
            id_to_label = {nid: n.get("label", nid) for nid, n in nodes.items()}
            edge_lines = []
            for e in edges:
                src = e.get("source", "")
                tgt = e.get("target", "")
                lbl = e.get("label", "relates to")
                src_label = id_to_label.get(src, src)
                tgt_label = id_to_label.get(tgt, tgt)
                if src in active_nodes or tgt in active_nodes:
                    edge_lines.append(f"  - {src_label} --[{lbl}]--> {tgt_label}")
            graph_context = "\n\nFINAL KNOWLEDGE GRAPH:\nNodes:\n" + "\n".join(node_lines)
            if edge_lines:
                graph_context += "\n\nEdges (relationships):\n" + "\n".join(edge_lines)

    # Compute stats
    duration_minutes = 0.0
    if len(segments) >= 2:
        duration_minutes = (segments[-1]["timestamp"] - segments[0]["timestamp"]) / 60
    stats = {
        "total_segments": len(segments),
        "total_chars": len(full_text),
        "duration_minutes": round(duration_minutes, 1),
    }

    # Truncate if very long
    max_chars = 150000
    transcript_for_recap = full_text[:max_chars]
    if len(full_text) > max_chars:
        transcript_for_recap += f"\n\n[Transcript truncated at {max_chars} chars out of {len(full_text)}]"

    lang_name = {"en": "English", "no": "Norwegian", "sv": "Swedish", "da": "Danish",
                 "de": "German", "fr": "French"}.get(session_lang, "English")

    system_prompt = f"""You generate structured session recap documents that surface insight, not meeting minutes.
You have access to both the full transcript AND the final knowledge graph (nodes and their relationships).

Return ONLY valid JSON with this exact structure:
{{
  "elevator_pitch": "2-3 sentences a participant could say out loud after the session about what it means. First person plural is fine. Written in {lang_name}.",
  "non_obvious_connections": [
    {{"topics": ["Topic A", "Topic B"], "insight": "What the link reveals that wasn't stated explicitly."}}
  ],
  "retain": ["First thing worth remembering a week from now.", "Second.", "Third."],
  "contradictions": ["Where the discussion diverged from stated positions or earlier claims."],
  "summary": "One paragraph reference summary.",
  "decisions": ["Decisions made, if any."],
  "open_threads": ["Unresolved tensions or threads worth following up."]
}}

LANGUAGE: Write ALL fields in {lang_name}. Every field — elevator_pitch, retain, non_obvious_connections insights, contradictions, summary, decisions, open_threads — must be written in {lang_name}. Do NOT switch to English for any field.

Rules:
- elevator_pitch: Write in {lang_name}, in the voice of a participant (first person). 2-3 sentences someone could actually say out loud.
- non_obvious_connections: 0 to 3 items ONLY. Draw on the knowledge graph edges to find links participants likely didn't notice in the room. Return an EMPTY ARRAY rather than fabricate connections. One real connection beats three plausible ones. Write the insight in {lang_name}.
- retain: EXACTLY 3 items in {lang_name}. The three ideas that should survive the week. Forces you to prioritize.
- contradictions: Often empty — that's fine. Only include when there's genuine divergence between what was said vs. stated positions, slides, or earlier claims. Return an EMPTY ARRAY if none. Write in {lang_name}.
- summary: One concise paragraph in {lang_name}. This is reference material, not the headline.
- decisions: Empty array if none. Name the people involved where possible. Write in {lang_name}.
- open_threads: Unresolved tensions or questions worth following up. Empty array if none. Write in {lang_name}.
- Prefer empty arrays over speculation. Never invent or pad.
- Use specific names, not "the user" or "the participant"."""

    user_prompt = f"SESSION TRANSCRIPT:\n\n{transcript_for_recap}{graph_context}\n\nGenerate the recap."

    model = "gemma4:31b"
    ollama_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 4096,
        },
    }

    # Retry once on parse failure with stricter reminder
    max_attempts = 2
    last_error = None

    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                # Add stricter reminder on retry
                ollama_body["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + "\n\nIMPORTANT: Return ONLY the raw JSON object. No markdown, no explanation, no code fences."},
                ]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{HUGIN_BASE_URL}/api/chat",
                    headers={"Content-Type": "application/json"},
                    json=ollama_body,
                    timeout=aiohttp.ClientTimeout(total=180),
                    ssl=False,
                ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        err = data.get("error", "") or str(data)
                        return JSONResponse({"error": f"Ollama: {err}"}, status_code=502)

            raw_text = data.get("message", {}).get("content", "")
            # Strip markdown code fences if present
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            recap = json.loads(cleaned.strip())

            # Add metadata
            recap["language"] = session_lang
            recap["schema_version"] = 2
            recap["transcript_stats"] = stats

            await db.store_recap(session_id, recap, model)
            return JSONResponse({"recap": recap, "model": model, "created_at": time.time()})

        except json.JSONDecodeError as e:
            last_error = e
            print(f"  Recap parse attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt < max_attempts - 1:
                continue
            # Final failure — store error state
            error_recap = {
                "schema_version": 2,
                "language": session_lang,
                "error": f"Failed to parse LLM response after {max_attempts} attempts: {str(last_error)}",
                "raw_response": raw_text[:2000],
                "transcript_stats": stats,
            }
            await db.store_recap(session_id, error_recap, model)
            return JSONResponse({"error": f"Failed to parse LLM response: {last_error}"}, status_code=502)
        except Exception as e:
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


# ─── Transcript cleaning ───

@app.post("/v1/sessions/{session_id}/clean-transcript")
async def clean_transcript(session_id: str):
    """Clean transcript segments using LLM to fix obvious STT errors."""
    segments = await db.get_session_transcript(session_id)
    if not segments:
        return JSONResponse({"error": "No transcript found"}, status_code=404)

    # Detect language from segments
    lang_counts: dict[str, int] = {}
    for seg in segments:
        lang = seg.get("stt_language", "")
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    session_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "en"
    lang_name = {"en": "English", "no": "Norwegian", "sv": "Swedish", "da": "Danish",
                 "de": "German", "fr": "French"}.get(session_lang, "English")

    # Pre-filter: catch repetition loops and obvious garbage before LLM
    import re as _re
    def pre_clean(text: str) -> str:
        """Catch STT hallucination loops and garbage that the LLM shouldn't waste tokens on."""
        words = text.split()
        if len(words) >= 10:
            # Find the longest repetition loop and replace it
            best = None  # (start, end, count)
            for plen in (1, 2, 3):
                for start in range(len(words)):
                    if start + plen * 5 > len(words):
                        break
                    pattern = " ".join(words[start:start + plen]).lower().rstrip(".,!?")
                    if not pattern:
                        continue
                    pos = start
                    while pos + plen <= len(words):
                        chunk = " ".join(words[pos:pos + plen]).lower().rstrip(".,!?")
                        if chunk == pattern:
                            pos += plen
                        else:
                            break
                    count = (pos - start) // plen
                    if count >= 5:
                        span = pos - start
                        if not best or span > (best[1] - best[0]):
                            best = (start, pos, count)
            if best:
                start, end, _ = best
                before = " ".join(words[:start]).strip()
                after = " ".join(words[end:]).strip()
                parts = [p for p in [before, "[inaudible]", after] if p]
                return pre_clean(" ".join(parts))
        # Single repeated character sequences
        text = _re.sub(r'(.)\1{20,}', '[inaudible]', text)
        return text

    for seg in segments:
        seg["text"] = pre_clean(seg["text"])

    system_prompt = f"""You are a transcript cleaner. You receive raw speech-to-text segments and fix obvious transcription errors.

Rules:
- Fix misspelled words, garbled text, and wrong language fragments
- Add missing punctuation and capitalization
- Fix obvious name misspellings (be consistent across segments)
- Preserve the speaker's original words — do NOT rephrase, summarize, or paraphrase
- If a segment contains "[inaudible]", keep that marker as-is
- If a segment is mostly noise or completely unintelligible, replace it with "[inaudible]"
- If a segment is fine, return it unchanged
- The transcript is in {lang_name}. Some segments may contain English terms or code-switching — preserve those naturally
- Return EXACTLY the same number of items as the input, in the same order
- Return ONLY a JSON array of strings, one per segment: ["cleaned segment 1", "cleaned segment 2", ...]
- Do NOT add any explanation, just the JSON array"""

    # Process in chunks of 40 segments
    CHUNK_SIZE = 40
    all_cleaned = []
    model = "gemma4:31b"

    for i in range(0, len(segments), CHUNK_SIZE):
        chunk = segments[i:i + CHUNK_SIZE]
        texts = [seg["text"] for seg in chunk]

        user_prompt = "Clean these transcript segments:\n" + json.dumps(texts, ensure_ascii=False)

        ollama_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 4096,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{HUGIN_BASE_URL}/api/chat",
                    headers={"Content-Type": "application/json"},
                    json=ollama_body,
                    timeout=aiohttp.ClientTimeout(total=180),
                    ssl=False,
                ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        err = data.get("error", "") or str(data)
                        return JSONResponse({"error": f"Ollama error on chunk {i // CHUNK_SIZE + 1}: {err}"}, status_code=502)

            raw_text = data.get("message", {}).get("content", "")
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            parsed = json.loads(cleaned.strip())

            # Handle both array format and object-with-array format
            if isinstance(parsed, dict):
                # LLM might wrap in an object — find the array
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break

            if not isinstance(parsed, list) or len(parsed) != len(chunk):
                print(f"  Clean chunk {i // CHUNK_SIZE + 1}: expected {len(chunk)} items, got {len(parsed) if isinstance(parsed, list) else 'non-list'}", file=sys.stderr)
                # Fall back to originals for this chunk
                all_cleaned.extend([{"seq": seg["seq"], "cleaned_text": seg["text"]} for seg in chunk])
                continue

            for j, seg in enumerate(chunk):
                ct = parsed[j] if isinstance(parsed[j], str) else seg["text"]
                all_cleaned.append({"seq": seg["seq"], "cleaned_text": ct})

        except json.JSONDecodeError as e:
            print(f"  Clean chunk {i // CHUNK_SIZE + 1} parse error: {e}", file=sys.stderr)
            all_cleaned.extend([{"seq": seg["seq"], "cleaned_text": seg["text"]} for seg in chunk])
        except Exception as e:
            return JSONResponse({"error": f"Chunk {i // CHUNK_SIZE + 1}: {type(e).__name__}: {e}"}, status_code=500)

    # Store all cleaned segments
    await db.store_cleaned_segments(session_id, all_cleaned)

    # Count how many actually changed
    changed = sum(1 for c, seg in zip(all_cleaned, segments) if c["cleaned_text"] != seg["text"])

    return JSONResponse({
        "ok": True,
        "total_segments": len(segments),
        "changed": changed,
        "model": model,
    })


# ─── Cross-session synthesis ───

@app.post("/v1/sessions/synthesis")
async def generate_synthesis(request: Request):
    """Generate a cross-session synthesis recap from multiple sessions' individual recaps."""
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if len(session_ids) < 2:
        return JSONResponse({"error": "Need at least 2 sessions"}, status_code=400)

    # Load each session's recap + graph
    sessions_data = []
    missing_recaps = []
    for sid in session_ids:
        recap = await db.get_recap(sid)
        if not recap:
            missing_recaps.append(sid)
            continue
        snapshot = await db.get_latest_snapshot(sid)
        # Get session metadata
        all_sessions = await db.list_sessions()
        meta = next((s for s in all_sessions if s["id"] == sid), {})
        sessions_data.append({
            "id": sid,
            "topic": meta.get("topic", ""),
            "created_at": meta.get("created_at", 0),
            "ended_at": meta.get("ended_at"),
            "recap": recap["recap"],
            "snapshot": snapshot,
        })

    if missing_recaps:
        return JSONResponse({
            "error": f"Sessions missing recaps: {', '.join(missing_recaps)}. Generate individual recaps first."
        }, status_code=400)

    # Detect language (majority from session recaps)
    lang_counts: dict[str, int] = {}
    for sd in sessions_data:
        lang = sd["recap"].get("language", "en")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    session_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "en"
    lang_name = {"en": "English", "no": "Norwegian", "sv": "Swedish", "da": "Danish",
                 "de": "German", "fr": "French"}.get(session_lang, "English")

    # Build per-session blocks
    session_blocks = []
    for i, sd in enumerate(sessions_data, 1):
        duration = ""
        if sd["ended_at"] and sd["created_at"]:
            mins = round((sd["ended_at"] - sd["created_at"]) / 60)
            duration = f" ({mins} min)"

        r = sd["recap"]
        block = f'SESSION {i}: "{sd["topic"] or "Untitled"}"{duration}\n'
        block += f'ID: {sd["id"]}\n'

        # Include recap highlights
        if r.get("elevator_pitch"):
            block += f'PITCH: {r["elevator_pitch"]}\n'
        if r.get("retain"):
            block += 'KEY TAKEAWAYS:\n' + '\n'.join(f'  - {item}' for item in r["retain"]) + '\n'
        if r.get("non_obvious_connections"):
            block += 'CONNECTIONS:\n'
            for conn in r["non_obvious_connections"]:
                topics = " ↔ ".join(conn.get("topics", []))
                block += f'  - {topics}: {conn.get("insight", "")}\n'
        if r.get("summary"):
            block += f'SUMMARY: {r["summary"]}\n'
        if r.get("contradictions"):
            block += 'CONTRADICTIONS:\n' + '\n'.join(f'  - {c}' for c in r["contradictions"]) + '\n'

        # Include graph
        if sd["snapshot"] and sd["snapshot"].get("graph"):
            graph = sd["snapshot"]["graph"]
            nodes = graph.get("nodes", {})
            edges = graph.get("edges", [])
            active = {nid: n for nid, n in nodes.items() if n.get("state") == "active"}
            if active:
                id_to_label = {nid: n.get("label", nid) for nid, n in nodes.items()}
                block += 'GRAPH NODES: ' + ', '.join(n.get("label", nid) for nid, n in active.items()) + '\n'
                edge_strs = []
                for e in edges:
                    src = id_to_label.get(e.get("source", ""), e.get("source", ""))
                    tgt = id_to_label.get(e.get("target", ""), e.get("target", ""))
                    edge_strs.append(f'{src} --[{e.get("label", "")}]--> {tgt}')
                if edge_strs:
                    block += 'GRAPH EDGES: ' + '; '.join(edge_strs) + '\n'

        session_blocks.append(block)

    all_sessions_text = '\n---\n\n'.join(session_blocks)

    system_prompt = f"""You synthesize insights across multiple session recaps from the same event or day.
You have access to each session's recap (elevator pitch, key takeaways, connections, summary) AND its knowledge graph.

Your job is to find the threads that run BETWEEN sessions — ideas that evolved, echoed, or contradicted each other across different conversations.

Return ONLY valid JSON with this exact structure:
{{
  "elevator_pitch": "The day/event in 2-3 sentences. What would a participant tell a colleague? Written in {lang_name}, first person plural.",
  "cross_connections": [
    {{"sessions": ["id1", "id2"], "topics": ["Topic A", "Topic B"], "insight": "What the link across these sessions reveals."}}
  ],
  "evolution": ["How an idea or theme evolved from one session to the next."],
  "tensions": ["Where one session contradicted or complicated another's conclusions."],
  "synthesis": "2-3 paragraph narrative of the day's arc — what emerged across all sessions taken together.",
  "language": "{session_lang}"
}}

Rules:
- elevator_pitch: Written in {lang_name}, first person. Something a participant would actually say.
- cross_connections: 0 to 5 items. Reference the specific session IDs. Draw on graph edges across sessions to find themes that link different conversations. Return an EMPTY ARRAY rather than fabricate.
- evolution: How ideas developed across the timeline of sessions. Empty array if nothing evolved.
- tensions: Where sessions disagreed or complicated each other. Often empty — that's fine.
- synthesis: A narrative, not a list. This is the "big picture" view of the day.
- Prefer empty arrays over speculation. Never invent connections."""

    user_prompt = f"{all_sessions_text}\n\nGenerate the cross-session synthesis."

    model = "gemma4:31b"
    ollama_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 4096,
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HUGIN_BASE_URL}/api/chat",
                headers={"Content-Type": "application/json"},
                json=ollama_body,
                timeout=aiohttp.ClientTimeout(total=300),
                ssl=False,
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    err = data.get("error", "") or str(data)
                    return JSONResponse({"error": f"Ollama: {err}"}, status_code=502)

        raw_text = data.get("message", {}).get("content", "")
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        synthesis = json.loads(cleaned.strip())
        synthesis["schema_version"] = 1
        synthesis["session_count"] = len(session_ids)

        row_id = await db.store_synthesis(session_ids, synthesis, model)
        return JSONResponse({
            "id": row_id,
            "session_ids": session_ids,
            "recap": synthesis,
            "model": model,
            "created_at": time.time(),
        })

    except json.JSONDecodeError as e:
        return JSONResponse({"error": f"Failed to parse LLM response: {e}"}, status_code=502)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)




@app.get("/v1/metrics")
async def get_metrics_rest():
    with metrics_lock:
        m = {**metrics, "uptime": time.time() - metrics["started_at"]}
    churn = reconciler.get_churn_metrics()
    m.update(churn)
    m["current_session_id"] = _current_session_id
    m["active_nodes"] = len([ns for ns in reconciler.nodes.values() if ns.state == "active"])
    with _active_llm_lock:
        m["llm_provider"] = _active_llm["provider"]
        m["llm_model"] = _active_llm["model"]
    stt = get_stt_config()
    m["stt_backend"] = stt["backend"]
    m["stt_remote_url"] = stt.get("remote_url", "")
    return JSONResponse(m)


# ─── LLM provider endpoints ───
@app.get("/v1/llm/providers")
async def list_llm_providers():
    """Return available LLM providers and models."""
    providers = {
        "anthropic": {
            "label": "Anthropic (Claude)",
            "models": [
                {"id": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4", "note": "Default"},
                {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5", "note": "Fast"},
            ],
            "available": bool(ANTHROPIC_API_KEY),
        },
    }
    if HUGIN_BASE_URL:
        providers["hugin"] = {
            "label": "Hugin (Self-hosted)",
            "models": [],
            "available": True,
        }
        # Fetch live model list from Ollama
        try:
            headers = {}
            if HUGIN_CF_ID and HUGIN_CF_SECRET:
                headers["CF-Access-Client-Id"] = HUGIN_CF_ID
                headers["CF-Access-Client-Secret"] = HUGIN_CF_SECRET
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{HUGIN_BASE_URL}/api/tags",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=False,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for m in data.get("models", []):
                            size = m.get("details", {}).get("parameter_size", "")
                            providers["hugin"]["models"].append({
                                "id": m["name"],
                                "label": m["name"],
                                "note": size,
                            })
        except Exception as e:
            print(f"  Hugin model list failed: {e}", file=sys.stderr)
            providers["hugin"]["available"] = False
            providers["hugin"]["error"] = str(e)

    if GEMINI_API_KEY:
        providers["gemini"] = {
            "label": "Google (Gemini)",
            "models": [
                {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash", "note": "$0.075/$0.30 per M tok"},
                {"id": "gemini-2.0-flash", "label": "Gemini 2.0 Flash", "note": "$0.10/$0.40 per M tok"},
            ],
            "available": True,
        }

    with _active_llm_lock:
        active = {**_active_llm}
    return JSONResponse({"providers": providers, "active": active})


@app.post("/v1/llm/active")
async def set_active_llm(request: Request):
    """Switch the active LLM provider and model."""
    body = await request.json()
    provider = body.get("provider", "").strip()
    model = body.get("model", "").strip()

    valid_providers = {"anthropic", "hugin", "gemini"}
    if provider not in valid_providers:
        return JSONResponse({"error": f"Unknown provider: {provider}"}, status_code=400)
    if not model:
        return JSONResponse({"error": "Model is required"}, status_code=400)
    if provider == "hugin" and not HUGIN_BASE_URL:
        return JSONResponse({"error": "HUGIN_BASE_URL not configured"}, status_code=400)
    if provider == "gemini" and not GEMINI_API_KEY:
        return JSONResponse({"error": "Gemini API key not configured"}, status_code=400)

    with _active_llm_lock:
        old = {**_active_llm}
        _active_llm["provider"] = provider
        _active_llm["model"] = model

    print(f"  LLM switched: {old['provider']}/{old['model']} → {provider}/{model}")
    return JSONResponse({"active": {"provider": provider, "model": model}})


# ─── STT backend endpoints ───
@app.get("/v1/stt/backends")
async def list_stt_backends():
    """Return available STT backends with health checks."""
    stt_cfg = get_stt_config()
    backends = {
        "remote": {
            "label": "faster-whisper",
            "note": "99 languages, proven",
            "url": stt_cfg["remote_url"],
            "available": False,
        },
        "parakeet": {
            "label": "Parakeet TDT 0.6b v3",
            "note": "25 EU languages, highest throughput",
            "url": stt_cfg["parakeet_url"],
            "available": False,
        },
        "canary": {
            "label": "Canary 1b v2",
            "note": "25 EU languages, best accuracy",
            "url": stt_cfg["canary_url"],
            "available": False,
        },
    }

    # Health check each backend in parallel
    async def check_health(key, url):
        try:
            headers = {}
            if key == "remote":
                cf_id = os.environ.get("HUGIN_CF_ID", "")
                cf_secret = os.environ.get("HUGIN_CF_SECRET", "")
                if cf_id and cf_secret:
                    headers["CF-Access-Client-Id"] = cf_id
                    headers["CF-Access-Client-Secret"] = cf_secret
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{url}/health",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=3),
                    ssl=False,
                ) as resp:
                    if resp.status == 200:
                        backends[key]["available"] = True
                        data = await resp.json()
                        if "model" in data:
                            backends[key]["model"] = data["model"]
        except Exception:
            pass

    await asyncio.gather(
        check_health("remote", stt_cfg["remote_url"]),
        check_health("parakeet", stt_cfg["parakeet_url"]),
        check_health("canary", stt_cfg["canary_url"]),
    )

    active = stt_cfg
    return JSONResponse({"backends": backends, "active": active})


@app.post("/v1/stt/active")
async def set_active_stt(request: Request):
    """Switch the active STT backend and/or language."""
    body = await request.json()
    backend = body.get("backend")
    language = body.get("language")

    old = get_stt_config()

    # If backend is being changed, validate it
    if backend is not None:
        backend = backend.strip()
        if backend not in ("remote", "parakeet", "canary"):
            return JSONResponse({"error": f"Unknown backend: {backend}"}, status_code=400)
    else:
        backend = old["backend"]

    lang = language.strip() if language is not None else old.get("language", "")
    configure_stt(backend, "", lang)
    new = get_stt_config()

    print(f"  STT config: backend={new['backend']}, language={new['language'] or 'auto'}")
    return JSONResponse({"active": new})


# ─── LLM cost tracking ───
# Pricing per million tokens (input, output)
_LLM_PRICING = {
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
}
_DEFAULT_PRICING = (0.0, 0.0)  # self-hosted = free


def _extract_usage(data: dict, provider: str) -> dict:
    """Extract token usage from API response, normalised."""
    if provider == "anthropic":
        u = data.get("usage", {})
        return {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)}
    else:
        # OpenAI-compatible (Gemini, Hugin/Ollama) — check _usage (stashed) or usage
        u = data.get("_usage", data.get("usage", {}))
        return {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)}


# ─── LLM proxy ───
async def _call_anthropic(body: dict) -> tuple[int, dict]:
    """POST to Anthropic Messages API. Returns (status, response_dict)."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json=body,
        ) as resp:
            data = await resp.json()
            return resp.status, data


async def _call_hugin(body: dict) -> tuple[int, dict]:
    """Translate Anthropic-format request to OpenAI-compatible, call Hugin,
    translate response back to Anthropic format."""
    with _active_llm_lock:
        model = _active_llm["model"]

    # Build OpenAI-compatible messages from Anthropic format
    oai_messages = []
    system_text = body.get("system", "")
    # Always prepend the structured-output prefix for local models
    system_text = _SMALL_MODEL_GRAPH_PREFIX + system_text
    oai_messages.append({"role": "system", "content": system_text})
    for msg in body.get("messages", []):
        oai_messages.append({"role": msg["role"], "content": msg["content"]})

    # Use Ollama native /api/chat — supports think:false and format:json
    ollama_body = {
        "model": model,
        "messages": oai_messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 2048,
        },
    }

    print(f"  Hugin request: model={model}, think={ollama_body['think']}, num_predict=2048")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{HUGIN_BASE_URL}/api/chat",
            headers={"Content-Type": "application/json"},
            json=ollama_body,
            timeout=aiohttp.ClientTimeout(total=120),
            ssl=False,
        ) as resp:
            data = await resp.json()

            if resp.status != 200:
                err_msg = data.get("error", "") or str(data)
                return resp.status, {"error": {"message": f"Hugin: {err_msg}"}}

            # Ollama native response: data.message.content
            msg = data.get("message", {})
            text = msg.get("content") or ""
            if not text:
                print(f"  Hugin: empty content. Message keys: {list(msg.keys())}", file=sys.stderr)
                print(f"  Hugin: raw response keys: {list(data.keys())}", file=sys.stderr)
                print(f"  Hugin: raw message: {json.dumps(msg)[:500]}", file=sys.stderr)

            # Translate to Anthropic format; extract usage from Ollama metrics
            result = {"content": [{"type": "text", "text": text}]}
            if "prompt_eval_count" in data:
                result["_usage"] = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                }
            return 200, result


async def _call_gemini(body: dict) -> tuple[int, dict]:
    """Translate Anthropic-format request to OpenAI-compatible, call Gemini,
    translate response back to Anthropic format."""
    with _active_llm_lock:
        model = _active_llm["model"]

    # Build OpenAI-compatible messages
    oai_messages = []
    system_text = body.get("system", "")
    if system_text:
        # Gemini is good at JSON but benefits from the same structural hints
        system_text = _SMALL_MODEL_GRAPH_PREFIX + system_text
        oai_messages.append({"role": "system", "content": system_text})
    for msg in body.get("messages", []):
        oai_messages.append({"role": msg["role"], "content": msg["content"]})

    oai_body = {
        "model": model,
        "messages": oai_messages,
        "temperature": 0,
    }
    # Gemini 2.5 Flash uses thinking tokens from the max_tokens budget.
    # The graph JSON needs ~1-2k tokens, but thinking can consume 2-4k.
    # Set a generous budget so thinking doesn't starve the actual output.
    oai_body["max_tokens"] = 16384

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GEMINI_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GEMINI_API_KEY}",
            },
            json=oai_body,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            data = await resp.json()

            if resp.status != 200:
                err_msg = data.get("error", {}).get("message", "") or str(data)
                return resp.status, {"error": {"message": f"Gemini: {err_msg}"}}

            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            # Gemini 2.5 Flash thinking mode: content can be null,
            # actual text may be in reasoning_content or parts
            text = msg.get("content") or ""
            if not text:
                text = msg.get("reasoning_content") or ""
            if not text:
                print(f"  Gemini: empty content. Keys: {list(msg.keys())}", file=sys.stderr)
                print(f"  Gemini: choice: {json.dumps(choice)[:500]}", file=sys.stderr)
            result = {"content": [{"type": "text", "text": text}]}
            if "usage" in data:
                result["_usage"] = data["usage"]
            return 200, result


def _extract_graph_json(raw_text: str) -> dict:
    """Extract a JSON object containing 'nodes' from LLM output.
    Handles markdown fences, thinking preamble, and extra text."""
    import re
    cleaned = raw_text.replace("```json", "").replace("```", "")

    # Strategy 1: find {"nodes" and parse from there (handles thinking preamble)
    for pattern in [r'\{\s*"nodes"\s*:', r"\{\s*'nodes'\s*:"]:
        match = re.search(pattern, cleaned)
        if match:
            start = match.start()
            # Find matching closing brace by counting depth
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start:i + 1])
                        except json.JSONDecodeError:
                            break  # try next strategy

    # Strategy 2: try the whole thing stripped
    stripped = cleaned.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Strategy 3: first { to last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        f"No valid graph JSON found in {len(raw_text)} chars",
        raw_text[:200], 0
    )


async def _broadcast_llm_response(status_code, data, req_id):
    """Send LLM response to ALL connected clients (guards against disconnected sockets)."""
    payload = json.dumps({
        "type": "claude_response",
        "status": status_code,
        "data": data,
        "req_id": req_id,
    })
    for ws in list(connected_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            pass


async def _proxy_claude(websocket: WebSocket, req: dict):
    """Proxy an LLM request server-side with circuit breaker."""
    global cb_state, cb_failures, cb_backoff_until, cb_backoff_secs, _summary
    # Capture session at request time — discard results if session changed
    _req_session_id = _current_session_id

    with cb_lock:
        now = time.time()
        if cb_state == "open":
            if now < cb_backoff_until:
                wait = cb_backoff_until - now
                print(f"  LLM: circuit breaker OPEN, retry in {wait:.0f}s", file=sys.stderr)
                with metrics_lock:
                    metrics["cb_state"] = "open"
                await _broadcast_llm_response(503,
                    {"error": "Circuit breaker open", "retry_after": wait},
                    req.get("req_id"))
                return
            else:
                cb_state = "half_open"
                print("  LLM: circuit breaker half_open, testing...")
                with metrics_lock:
                    metrics["cb_state"] = "half_open"

    # Determine provider and inject server-side model
    with _active_llm_lock:
        provider = _active_llm["provider"]
        model = _active_llm["model"]

    t0 = time.time()
    try:
        body = req.get("body", {})
        body["model"] = model  # server overrides client model

        if provider == "hugin":
            status_code, data = await _call_hugin(body)
        elif provider == "gemini":
            status_code, data = await _call_gemini(body)
        else:
            status_code, data = await _call_anthropic(body)

        dt = time.time() - t0

        # Extract token usage and calculate cost
        usage = _extract_usage(data, provider)
        pricing = _LLM_PRICING.get(model, _DEFAULT_PRICING)
        call_cost = (usage["input"] * pricing[0] + usage["output"] * pricing[1]) / 1_000_000

        with metrics_lock:
            metrics["claude_calls"] += 1
            metrics["claude_last_duration"] = dt
            metrics["claude_total_time"] += dt
            metrics["claude_avg_duration"] = metrics["claude_total_time"] / metrics["claude_calls"]
            metrics["llm_input_tokens"] = metrics.get("llm_input_tokens", 0) + usage["input"]
            metrics["llm_output_tokens"] = metrics.get("llm_output_tokens", 0) + usage["output"]
            metrics["llm_session_cost"] = metrics.get("llm_session_cost", 0.0) + call_cost
            metrics["llm_last_cost"] = call_cost
            metrics["llm_last_input_tokens"] = usage["input"]
            metrics["llm_last_output_tokens"] = usage["output"]

        if status_code == 200:
            cost_str = f"${call_cost:.4f}" if call_cost > 0 else "free"
            print(f"  LLM [{provider}/{model}]: 200 OK ({dt:.1f}s, {usage['input']}+{usage['output']} tok, {cost_str})")
            with cb_lock:
                cb_state = "closed"
                cb_failures = 0
                cb_backoff_secs = 5.0
            with metrics_lock:
                metrics["cb_state"] = "closed"
                metrics["cb_failures"] = 0
                metrics["claude_last_error"] = ""

            # Run reconciler on LLM response
            try:
                raw_text = "".join(c.get("text", "") for c in data.get("content", []))
                parsed = _extract_graph_json(raw_text)
                if parsed.get("nodes") and parsed.get("edges") is not None:
                    # Discard if session changed while LLM was processing
                    if _current_session_id != _req_session_id:
                        print(f"  LLM: discarding stale response (session changed {_req_session_id} → {_current_session_id})")
                        return
                    n_before = len(reconciler.nodes)
                    graph = reconciler.reconcile(parsed)
                    n_after = len(reconciler.nodes)
                    if parsed.get("summary"):
                        _summary = parsed["summary"]
                    if _current_session_id:
                        await db.store_snapshot(
                            _current_session_id, _seq_counter,
                            reconciler.get_full_state(), "analysis"
                        )
                    data = {
                        "content": [{"type": "text", "text": json.dumps({
                            **graph, "summary": _summary,
                        })}],
                    }
                    churn = reconciler.get_churn_metrics()
                    with metrics_lock:
                        metrics["nodes_added_per_min"] = churn["nodes_added_per_min"]
                        metrics["nodes_removed_per_min"] = churn["nodes_removed_per_min"]
                        metrics["edge_churn_per_min"] = churn["edge_churn_per_min"]
                        metrics["llm_parse_ok"] = metrics.get("llm_parse_ok", 0) + 1
                        metrics["llm_last_node_count"] = len(parsed.get("nodes", []))
                    print(f"  LLM: parsed {len(parsed['nodes'])} nodes, {len(parsed.get('edges',[]))} edges (reconciler: {n_before}→{n_after})")

                    # Broadcast reconciled graph to all clients as graph_update
                    graph_msg = json.dumps({"type": "graph_update", "graph": graph, "session_id": _current_session_id})
                    for ws in list(connected_clients):
                        try:
                            await ws.send_text(graph_msg)
                        except Exception:
                            pass
                else:
                    print(f"  LLM: parsed JSON but missing nodes/edges keys", file=sys.stderr)
                    with metrics_lock:
                        metrics["llm_parse_no_graph"] = metrics.get("llm_parse_no_graph", 0) + 1
            except (json.JSONDecodeError, KeyError) as parse_err:
                print(f"  LLM: response parse error: {parse_err}", file=sys.stderr)
                print(f"  LLM: raw text: {raw_text[:500]}", file=sys.stderr)
                with metrics_lock:
                    metrics["llm_parse_fail"] = metrics.get("llm_parse_fail", 0) + 1
                    metrics["llm_last_raw_fail"] = raw_text[:500]

        elif status_code == 429:
            err_msg = data.get("error", {}).get("message", "Rate limited")
            print(f"  LLM: 429 RATE LIMITED ({dt:.1f}s) — {err_msg}", file=sys.stderr)
            with cb_lock:
                cb_state = "open"
                cb_backoff_secs = min(cb_backoff_secs * 2, CB_MAX_BACKOFF)
                cb_backoff_until = time.time() + cb_backoff_secs
                cb_failures += 1
            with metrics_lock:
                metrics["claude_errors"] += 1
                metrics["cb_state"] = "open"
                metrics["cb_failures"] = cb_failures
                metrics["claude_last_error"] = f"429: {err_msg}"

        elif status_code >= 500:
            err_msg = data.get("error", {}).get("message", f"Server error {status_code}")
            print(f"  LLM: {status_code} SERVER ERROR ({dt:.1f}s) — {err_msg}", file=sys.stderr)
            with cb_lock:
                cb_failures += 1
                if cb_failures >= CB_FAILURE_THRESHOLD:
                    cb_state = "open"
                    cb_backoff_secs = min(cb_backoff_secs * 2, CB_MAX_BACKOFF)
                    cb_backoff_until = time.time() + cb_backoff_secs
            with metrics_lock:
                metrics["claude_errors"] += 1
                metrics["cb_state"] = cb_state
                metrics["cb_failures"] = cb_failures
                metrics["claude_last_error"] = f"{status_code}: {err_msg}"

        else:
            err_msg = data.get("error", {}).get("message", f"HTTP {status_code}")
            print(f"  LLM: {status_code} ERROR ({dt:.1f}s) — {err_msg}", file=sys.stderr)
            with metrics_lock:
                metrics["claude_errors"] += 1
                metrics["claude_last_error"] = f"{status_code}: {err_msg}"

        await _broadcast_llm_response(status_code, data, req.get("req_id"))
    except Exception as e:
        dt = time.time() - t0
        print(f"  LLM: EXCEPTION ({dt:.1f}s) — {type(e).__name__}: {e}", file=sys.stderr)
        with cb_lock:
            cb_failures += 1
            if cb_failures >= CB_FAILURE_THRESHOLD:
                cb_state = "open"
                cb_backoff_secs = min(cb_backoff_secs * 2, CB_MAX_BACKOFF)
                cb_backoff_until = time.time() + cb_backoff_secs
        with metrics_lock:
            metrics["claude_calls"] += 1
            metrics["claude_errors"] += 1
            metrics["cb_state"] = cb_state
            metrics["cb_failures"] = cb_failures
            metrics["claude_last_error"] = f"{type(e).__name__}: {e}"
        await _broadcast_llm_response(500, {"error": str(e)}, req.get("req_id"))


# ─── Audio chunk handler ───
async def _handle_audio_chunk(audio_arr: np.ndarray, source_rate: int):
    """Process an audio chunk from the browser: STT → store segment → broadcast."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            stt_worker.transcribe_audio_chunk,
            audio_arr, source_rate, metrics, metrics_lock,
        )
    except Exception as e:
        print(f"  STT error: {e}", file=sys.stderr)
        return

    if result:
        seq = _next_seq()
        msg = {
            "type": "transcript",
            "text": result["text"],
            "seq": seq,
            "timestamp": time.time(),
        }

        # Persist with STT metadata
        if _current_session_id:
            await db.store_segment(
                _current_session_id, seq, result["text"],
                is_partial=False,
                timestamp=msg["timestamp"],
                stt_language=result.get("language", ""),
                stt_backend=result.get("backend", ""),
                stt_latency_ms=result.get("latency_ms"),
                stt_raw_text=result.get("raw_text"),
            )

        # Broadcast to all clients
        payload = json.dumps(msg)
        for ws in list(connected_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                pass


# ─── WebSocket ───
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    with metrics_lock:
        metrics["ws_clients"] = len(connected_clients)
    print(f"Browser connected ({len(connected_clients)} clients)")
    await websocket.send_json({
        "type": "status", "status": "connected",
        "message": "STT server ready",
    })
    # If a session is active, send restore so late-joining displays catch up
    if _current_session_id:
        client_sessions[websocket] = _current_session_id
        snapshot = await db.get_latest_snapshot(_current_session_id)
        segments = await db.get_segments_since(_current_session_id, 0)
        await websocket.send_json({
            "type": "restore",
            "session_id": _current_session_id,
            "snapshot": snapshot,
            "segments": segments,
            "restore_ms": 0,
        })
    try:
        while True:
            raw = await websocket.receive_text()
            d = json.loads(raw)
            msg_type = d.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "get_metrics":
                with metrics_lock:
                    m = {**metrics, "uptime": time.time() - metrics["started_at"]}
                churn = reconciler.get_churn_metrics()
                m.update(churn)
                m["current_session_id"] = _current_session_id
                m["active_nodes"] = len([ns for ns in reconciler.nodes.values() if ns.state == "active"])
                with _active_llm_lock:
                    m["llm_provider"] = _active_llm["provider"]
                    m["llm_model"] = _active_llm["model"]
                stt = get_stt_config()
                m["stt_backend"] = stt["backend"]
                m["stt_remote_url"] = stt.get("remote_url", "")
                await websocket.send_json({"type": "metrics", **m})

            elif msg_type == "claude_request":
                asyncio.create_task(_proxy_claude(websocket, d))

            elif msg_type == "connect_session":
                session_id = d.get("session_id")
                last_seq = d.get("last_seq", 0)
                if session_id:
                    client_sessions[websocket] = session_id
                    with metrics_lock:
                        metrics["ws_reconnects"] += 1
                    # Send restore data
                    t0 = time.time()
                    snapshot = await db.get_latest_snapshot(session_id)
                    segments = await db.get_segments_since(session_id, last_seq)
                    restore_ms = (time.time() - t0) * 1000
                    with metrics_lock:
                        metrics["last_restore_ms"] = restore_ms
                    await websocket.send_json({
                        "type": "restore",
                        "snapshot": snapshot,
                        "segments": segments,
                        "restore_ms": round(restore_ms, 1),
                    })

            elif msg_type == "frontend_metrics":
                fps = d.get("fps", 0)
                with metrics_lock:
                    metrics["frontend_fps"] = fps

            elif msg_type == "audio_chunk":
                audio_bytes = base64.b64decode(d["data"])
                audio_arr = np.frombuffer(audio_bytes, dtype=np.float32)
                source_rate = d.get("sample_rate", 48000)
                asyncio.create_task(_handle_audio_chunk(audio_arr, source_rate))

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        connected_clients.discard(websocket)
        client_sessions.pop(websocket, None)
        with metrics_lock:
            metrics["ws_clients"] = len(connected_clients)
        print(f"Browser disconnected ({len(connected_clients)} clients)")


# ─── Background loops ───
async def broadcast_loop():
    """Poll transcript_queue and broadcast to all WS clients."""
    while True:
        try:
            msg = transcript_queue.get_nowait()
            seq = _next_seq()
            msg["seq"] = seq

            # Persist segment
            if _current_session_id and msg.get("type") in ("transcript", "partial_transcript"):
                await db.store_segment(
                    _current_session_id, seq, msg["text"],
                    is_partial=(msg["type"] == "partial_transcript"),
                    timestamp=msg.get("timestamp", time.time()),
                )

            if connected_clients:
                p = json.dumps(msg)
                for ws in list(connected_clients):
                    try:
                        await ws.send_text(p)
                    except Exception:
                        pass
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)


async def snapshot_loop():
    """Periodic graph snapshot every 60s."""
    while True:
        await asyncio.sleep(60)
        if _current_session_id and reconciler.nodes:
            try:
                await db.store_snapshot(
                    _current_session_id, _seq_counter,
                    reconciler.get_full_state(), "periodic"
                )
            except Exception as e:
                print(f"  Snapshot error: {e}", file=sys.stderr)


# ─── Entry point ───
if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser(description="Mímir Server")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8765, help="Bind port")
    args = p.parse_args()

    print("=" * 50)
    print("  Mímir : Server")
    print("=" * 50 + "\n")

    WS_PORT = args.port

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
