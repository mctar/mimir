#!/usr/bin/env python3
"""
Mimir — Session Graph Export
Generate PDF snapshots and video replays of session knowledge graphs.
Uses Playwright to render the D3 visualization headlessly.

Usage:
    python export.py pdf <session_id>
    python export.py pdf <session_id> -o graph.pdf
    python export.py video <session_id>
    python export.py video <session_id> -o output.mp4 --speed 2.0

Prerequisites:
    pip install playwright && playwright install chromium
"""

import argparse, asyncio, json, logging, os, re, subprocess, sys, tempfile

import aiohttp
import db


# ─── LLM config for slides (mirrors routes_facilitator.py pattern) ────────────

ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL   = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
HUGIN_BASE_URL       = os.environ.get("HUGIN_BASE_URL", "https://munin.btrbot.com")
HUGIN_CF_ID          = os.environ.get("HUGIN_CF_ID", "")
HUGIN_CF_SECRET      = os.environ.get("HUGIN_CF_SECRET", "")
GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL      = "https://generativelanguage.googleapis.com/v1beta/openai"

_SLIDES_SYSTEM = """\
You are an expert presentation designer. Create complete, beautiful HTML slide decks.

RULES (non-negotiable):
- Output ONLY raw HTML. No markdown fences, no explanation, nothing else.
- Every slide must fit the viewport exactly: height:100vh; overflow:hidden; margin:0; padding:0.
- The deck must be fully self-contained: no external URLs, no CDN links.
- Navigation: left/right arrow keys and space bar advance slides.
- Use CSS transitions between slides (fade or horizontal slide).
- Style: dark, modern, bold. Background: #0a0a0f. Accent: #7c3aed.
- Typography: system fonts only (-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif).
- Large text, high contrast, minimal clutter — legible from the back of a room.
- Slide structure: title, overview/pitch, key points (1 per slide), concepts map,
  non-obvious connections, tensions, conclusion.
"""


EXPORT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export-graph.html")


def find_peak_snapshot(snapshots: list[dict]) -> dict:
    """Find the snapshot with the most active nodes (best visual)."""
    def score(s):
        nodes = s["graph"].get("nodes", {})
        active = sum(1 for n in nodes.values() if n.get("state") == "active")
        edges = len(s["graph"].get("edges", []))
        return (active, edges)
    return max(snapshots, key=score)


def format_date(ts: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


# ─── PDF Export ───

async def export_pdf(session_id: str, output: str, db_path: str = "livemind.db"):
    """Export the peak graph snapshot as a two-page PDF."""
    from playwright.async_api import async_playwright

    if not db._db:
        await db.init_db(db_path)

    # Get session info
    sessions = await db.list_sessions()
    session = next((s for s in sessions if s["id"] == session_id), None)
    if not session:
        sessions = await db.list_sessions(archived=True)
        session = next((s for s in sessions if s["id"] == session_id), None)
    if not session:
        print(f"Session '{session_id}' not found.", file=sys.stderr)
        return

    # Get snapshots
    snapshots = await db.get_session_snapshots(session_id)
    if not snapshots:
        print(f"No snapshots for session '{session_id}'.", file=sys.stderr)
        return

    peak = find_peak_snapshot(snapshots)
    nodes = peak["graph"].get("nodes", {})
    active_count = sum(1 for n in nodes.values() if n.get("state") == "active")
    edge_count = len(peak["graph"].get("edges", []))
    print(f"Peak snapshot: {active_count} nodes, {edge_count} edges")

    # Get summary
    recap_data = await db.get_recap(session_id)
    summary = ""
    if recap_data and recap_data.get("recap"):
        r = recap_data["recap"]
        summary = r.get("elevator_pitch", "") or r.get("summary", "")
    if not summary:
        summary = session.get("summary", "") or ""

    # Compute duration
    duration = ""
    if session.get("ended_at") and session.get("created_at"):
        duration = format_duration(session["ended_at"] - session["created_at"])

    meta = {
        "topic": session.get("topic", "Untitled"),
        "date": format_date(session["created_at"]),
        "summary": summary,
        "stats": {
            "nodes": active_count,
            "edges": edge_count,
            "segments": session.get("segment_count", "—"),
            "duration": duration or "—",
        },
    }

    # Render with Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        await page.goto("file://" + EXPORT_HTML)

        # Inject data and render
        await page.evaluate(f"renderStatic({json.dumps(peak['graph'])}, {json.dumps(meta)})")
        await page.wait_for_function("window.__READY__ === true", timeout=10000)
        await page.wait_for_timeout(800)  # fonts + render settle

        # Two-page PDF: graph page + metadata page
        await page.pdf(
            path=output,
            width="1920px",
            height="1080px",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )

        await browser.close()

    size_kb = os.path.getsize(output) / 1024
    print(f"PDF saved: {output} ({size_kb:.0f} KB)")


# ─── Video Export ───

async def export_video(
    session_id: str,
    output: str,
    db_path: str = "livemind.db",
    speed: float = 1.0,
    max_hold: float = 3.0,
    settle_time: float = 2.5,
):
    """Export session graph evolution as an mp4 video.

    Uses Playwright's built-in screen recording to capture smooth D3 transitions
    instead of jerky frame-by-frame screenshots.
    """
    from playwright.async_api import async_playwright

    if not db._db:
        await db.init_db(db_path)

    snapshots = await db.get_session_snapshots(session_id)
    if not snapshots or len(snapshots) < 2:
        print(f"Not enough snapshots for video (need >= 2, got {len(snapshots or [])}).", file=sys.stderr)
        return

    print(f"Recording {len(snapshots)} snapshots → {output}")
    print(f"  Speed: {speed}x, max hold: {max_hold}s, settle: {settle_time}s")

    tmpdir = tempfile.mkdtemp(prefix="mimir-export-")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()

            # Create context with video recording
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                record_video_dir=tmpdir,
                record_video_size={"width": 1920, "height": 1080},
            )
            page = await context.new_page()

            await page.goto("file://" + EXPORT_HTML)
            await page.evaluate("window.__initVideo__()")
            await page.wait_for_function("window.__READY__ === true", timeout=10000)

            # Brief pause before first snapshot (clean start)
            await page.wait_for_timeout(500)

            for i, snap in enumerate(snapshots):
                # Apply snapshot — D3 animates the transition live
                await page.evaluate(f"window.__applySnapshot__({json.dumps(snap['graph'])})")

                # Let D3 animate and settle naturally
                await page.wait_for_timeout(int(settle_time * 1000))

                # Hold: compressed real-time gap between snapshots
                if i < len(snapshots) - 1:
                    real_gap = snapshots[i + 1]["created_at"] - snap["created_at"]
                    hold_ms = min(max_hold, real_gap / speed) * 1000
                    # Subtract settle time already waited
                    extra_hold = max(0, hold_ms - settle_time * 1000)
                    if extra_hold > 0:
                        await page.wait_for_timeout(int(extra_hold))

                pct = (i + 1) / len(snapshots) * 100
                print(f"\r  Recording: [{pct:5.1f}%] snapshot {i+1}/{len(snapshots)}", end="", flush=True)

            # Brief pause at end
            await page.wait_for_timeout(1500)

            # Close page + context to finalize the recording
            await page.close()
            video_path = await page.video.path()
            await context.close()
            await browser.close()

        print(f"\n  Converting to mp4...")

        # Convert WebM to MP4 with ffmpeg
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "20",
            "-loglevel", "error",
            output,
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr.decode()}", file=sys.stderr)
            return

        size = os.path.getsize(output)
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.0f} KB"
        print(f"  Video saved: {output} ({size_str})")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── HTML Slides Export ───

async def _llm_call_slides(tier: dict, system: str, user: str) -> str:
    """Call a single LLM tier for slides generation. No JSON prefix applied."""
    provider = tier["provider"]
    model = tier["model"]
    timeout = aiohttp.ClientTimeout(total=120)

    if provider == "anthropic":
        url = f"{ANTHROPIC_BASE_URL}/v1/messages"
        headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
        if ANTHROPIC_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {ANTHROPIC_AUTH_TOKEN}"
        else:
            headers["x-api-key"] = ANTHROPIC_API_KEY
        body = {"model": model, "max_tokens": 8192, "system": system,
                "messages": [{"role": "user", "content": user}]}
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=body, timeout=timeout, ssl=False) as r:
                if r.status != 200:
                    raise RuntimeError(f"Anthropic error {r.status}: {await r.text()}")
                data = await r.json()
                return data["content"][0]["text"]

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
        "model": model, "max_tokens": 8192,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers, json=body, timeout=timeout) as r:
            if r.status != 200:
                raise RuntimeError(f"{provider} error {r.status}: {await r.text()}")
            data = await r.json()
            return data["choices"][0]["message"]["content"]


async def export_slides(session_id: str, output: str, db_path: str = "livemind.db",
                        chain: list[dict] | None = None):
    """Generate a self-contained HTML slide deck from a session recap via LLM."""
    if not db._db:
        await db.init_db(db_path)

    sessions = await db.list_sessions()
    session = next((s for s in sessions if s["id"] == session_id), None)
    if not session:
        sessions = await db.list_sessions(archived=True)
        session = next((s for s in sessions if s["id"] == session_id), None)
    if not session:
        raise ValueError(f"Session '{session_id}' not found.")

    recap_data = await db.get_recap(session_id)
    if not recap_data or not recap_data.get("recap"):
        raise ValueError(f"Session '{session_id}' has no recap. Generate one first.")
    recap = recap_data["recap"]

    # Build concept summary from peak graph snapshot
    snapshots = await db.get_session_snapshots(session_id)
    nodes_text = edges_text = ""
    if snapshots:
        peak = find_peak_snapshot(snapshots)
        nodes = peak["graph"].get("nodes", {})
        active = [n for n in nodes.values() if n.get("state") == "active"]
        nodes_text = ", ".join(n.get("label", n.get("id", "?")) for n in active[:24])
        edges_text = "; ".join(
            f"{e.get('source', '?')} → {e.get('target', '?')}"
            for e in peak["graph"].get("edges", [])[:20]
        )

    duration = ""
    if session.get("ended_at") and session.get("created_at"):
        duration = format_duration(session["ended_at"] - session["created_at"])

    def _fmt_list(value) -> str:
        if isinstance(value, list):
            return "\n".join(
                f"- {item.get('topics', item) if isinstance(item, dict) else item}"
                + (f": {item['insight']}" if isinstance(item, dict) and item.get("insight") else "")
                for item in value
            )
        return str(value) if value else ""

    user_prompt = f"""Create a complete HTML presentation for this session.

Topic: {session.get('topic', 'Untitled')}
Date: {format_date(session.get('created_at', 0))} | Duration: {duration or 'N/A'}

RECAP:
Pitch: {recap.get('elevator_pitch', '')}
Summary: {recap.get('summary', '')}

Key takeaways:
{_fmt_list(recap.get('key_takeaways', []))}

Non-obvious connections:
{_fmt_list(recap.get('non_obvious_connections', []))}

Tensions & contradictions:
{_fmt_list(recap.get('contradictions', []))}

Active concepts:
{nodes_text}

Relationships:
{edges_text}

Output ONLY the complete HTML. Nothing else."""

    if not chain:
        raise RuntimeError("No LLM chain provided. Cannot generate slides.")

    topic = session.get('topic', 'Untitled')
    n_nodes = len(nodes_text.split(",")) if nodes_text else 0
    print(f"Slides export: session={session_id[:8]}… topic='{topic}' nodes={n_nodes}")
    print(f"  Prompt: {len(user_prompt)} chars | Chain: {[t['provider']+'/'+t['model'] for t in chain]}")

    html_text = None
    last_error = None
    for tier in chain:
        provider, model = tier["provider"], tier["model"]
        print(f"  Calling {provider}/{model}…", flush=True)
        t0 = __import__("time").time()
        try:
            raw = await _llm_call_slides(tier, _SLIDES_SYSTEM, user_prompt)
            dt = __import__("time").time() - t0
            print(f"  {provider}/{model}: {len(raw)} chars in {dt:.1f}s")
            # Strip markdown fencing if present
            m = re.search(r'```(?:html)?\s*(<!DOCTYPE|<html).*?```', raw, re.DOTALL | re.IGNORECASE)
            if m:
                raw = m.group(0).replace("```html", "").replace("```", "").strip()
            # Find HTML start
            for marker in ("<!DOCTYPE html>", "<!doctype html>", "<html"):
                idx = raw.find(marker)
                if idx >= 0:
                    html_text = raw[idx:]
                    break
            if html_text:
                if not html_text.rstrip().lower().endswith("</html>"):
                    html_text = html_text.rstrip() + "\n</html>"
                print(f"  HTML extracted: {len(html_text)} chars")
                break
            else:
                print(f"  WARNING: no HTML found in response, trying next tier", file=sys.stderr)
        except Exception as e:
            dt = __import__("time").time() - t0
            print(f"  {provider}/{model}: FAILED after {dt:.1f}s — {e}", file=sys.stderr)
            logging.warning("export_slides: tier %s/%s failed: %s", provider, model, e)
            last_error = e

    if not html_text:
        raise RuntimeError(f"All LLM tiers failed. Last error: {last_error}")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html_text)

    size_kb = os.path.getsize(output) / 1024
    print(f"Slides saved: {output} ({size_kb:.0f} KB)")


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description="Export Mimir session graphs as PDF or video"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pdf_parser = sub.add_parser("pdf", help="Export peak graph snapshot as PDF")
    pdf_parser.add_argument("session_id", help="Session ID to export")
    pdf_parser.add_argument("-o", "--output", help="Output file path (default: <session_id>.pdf)")
    pdf_parser.add_argument("--db", default="livemind.db", help="Database path")

    vid_parser = sub.add_parser("video", help="Export graph evolution as mp4 video")
    vid_parser.add_argument("session_id", help="Session ID to export")
    vid_parser.add_argument("-o", "--output", help="Output file path (default: <session_id>.mp4)")
    vid_parser.add_argument("--speed", type=float, default=2.0, help="Playback speed multiplier (default: 2.0)")
    vid_parser.add_argument("--max-hold", type=float, default=3.0, help="Max seconds per snapshot (default: 3.0)")
    vid_parser.add_argument("--settle", type=float, default=2.5, help="Seconds for D3 animation to settle per snapshot (default: 2.5)")
    vid_parser.add_argument("--db", default="livemind.db", help="Database path")

    args = parser.parse_args()

    if args.command == "pdf":
        output = args.output or f"{args.session_id}.pdf"
        asyncio.run(export_pdf(args.session_id, output, args.db))

    elif args.command == "video":
        output = args.output or f"{args.session_id}.mp4"
        asyncio.run(export_video(
            args.session_id, output, args.db,
            speed=args.speed, max_hold=args.max_hold, settle_time=args.settle,
        ))


if __name__ == "__main__":
    main()
