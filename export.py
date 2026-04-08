#!/usr/bin/env python3
"""
Mimir — Session Graph Export
Generate PDF snapshots and video replays of session knowledge graphs.
Uses Playwright to render the D3 visualization headlessly.

Usage:
    python export.py pdf <session_id>
    python export.py pdf <session_id> -o graph.pdf
    python export.py video <session_id>
    python export.py video <session_id> -o output.mp4 --fps 30 --speed 2.0

Prerequisites:
    pip install playwright && playwright install chromium
"""

import argparse, asyncio, json, os, shutil, sys, tempfile, time

import db


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


# ─── PDF Export ───

async def export_pdf(session_id: str, output: str, db_path: str = "livemind.db"):
    """Export the peak graph snapshot as a PDF."""
    from playwright.async_api import async_playwright

    await db.init_db(db_path)

    # Get session info
    sessions = await db.list_sessions()
    session = next((s for s in sessions if s["id"] == session_id), None)
    if not session:
        # Check archived too
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
    print(f"Peak snapshot: {active_count} active nodes, {len(peak['graph'].get('edges', []))} edges")

    # Get summary from recap if available
    recap_data = await db.get_recap(session_id)
    summary = ""
    if recap_data and recap_data.get("recap"):
        r = recap_data["recap"]
        summary = r.get("elevator_pitch", "") or r.get("summary", "")

    # If no recap summary, try the session summary
    if not summary:
        summary = session.get("summary", "") or ""

    meta = {
        "topic": session.get("topic", "Untitled"),
        "date": format_date(session["created_at"]),
        "summary": summary,
    }

    await db.close_db()

    # Render with Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        file_url = "file://" + EXPORT_HTML
        await page.goto(file_url)

        # Inject data and render
        await page.evaluate(f"renderStatic({json.dumps(peak['graph'])}, {json.dumps(meta)})")

        # Wait for render
        await page.wait_for_function("window.__READY__ === true", timeout=10000)

        # Small delay for fonts to load
        await page.wait_for_timeout(500)

        # Export PDF
        await page.pdf(
            path=output,
            width="1920px",
            height="1080px",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )

        await browser.close()

    print(f"PDF saved: {output}")


# ─── Video Export ───

async def export_video(
    session_id: str,
    output: str,
    db_path: str = "livemind.db",
    fps: int = 10,
    speed: float = 1.0,
    max_hold: float = 3.0,
):
    """Export session graph evolution as an mp4 video."""
    from playwright.async_api import async_playwright

    await db.init_db(db_path)

    snapshots = await db.get_session_snapshots(session_id)
    if not snapshots or len(snapshots) < 2:
        print(f"Not enough snapshots for video (need >= 2, got {len(snapshots)}).", file=sys.stderr)
        return

    await db.close_db()

    print(f"Rendering {len(snapshots)} snapshots → {output}")
    print(f"  FPS: {fps}, speed: {speed}x, max hold: {max_hold}s")

    tmpdir = tempfile.mkdtemp(prefix="mimir-export-")
    frame_num = 0

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": 1920, "height": 1080})

            file_url = "file://" + EXPORT_HTML
            await page.goto(file_url)

            # Init video mode
            await page.evaluate("window.__initVideo__()")
            await page.wait_for_function("window.__READY__ === true", timeout=10000)

            for i, snap in enumerate(snapshots):
                # Apply snapshot
                await page.evaluate(f"window.__applySnapshot__({json.dumps(snap['graph'])})")
                await page.wait_for_function("window.__SETTLED__ === true", timeout=10000)

                # Small delay for SVG render
                await page.wait_for_timeout(50)

                # Take screenshot
                frame_path = os.path.join(tmpdir, f"frame_{frame_num:05d}.png")
                await page.screenshot(path=frame_path)
                frame_num += 1

                # Calculate hold frames (compressed real-time)
                if i < len(snapshots) - 1:
                    real_gap = snapshots[i + 1]["created_at"] - snap["created_at"]
                    hold_time = min(max_hold, real_gap / speed)
                    hold_frames = max(1, int(hold_time * fps)) - 1  # -1 because we already saved one frame

                    # Duplicate frames for hold duration
                    for _ in range(hold_frames):
                        dup_path = os.path.join(tmpdir, f"frame_{frame_num:05d}.png")
                        shutil.copy2(frame_path, dup_path)
                        frame_num += 1

                pct = (i + 1) / len(snapshots) * 100
                print(f"\r  Rendering: [{pct:5.1f}%] snapshot {i+1}/{len(snapshots)} | {frame_num} frames", end="", flush=True)

            await browser.close()

        print(f"\n  Encoding {frame_num} frames...")

        # Encode with ffmpeg
        import subprocess
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", os.path.join(tmpdir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            "-loglevel", "error",
            output,
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True)
        if result.returncode != 0:
            print(f"\n  ffmpeg error: {result.stderr.decode()}", file=sys.stderr)
            return

        # Report file size
        size = os.path.getsize(output)
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        else:
            size_str = f"{size / 1024:.0f} KB"

        duration = frame_num / fps
        print(f"  Video saved: {output} ({size_str}, {duration:.1f}s)")

    finally:
        # Clean up temp frames
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description="Export Mimir session graphs as PDF or video"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # PDF subcommand
    pdf_parser = sub.add_parser("pdf", help="Export peak graph snapshot as PDF")
    pdf_parser.add_argument("session_id", help="Session ID to export")
    pdf_parser.add_argument("-o", "--output", help="Output file path (default: <session_id>.pdf)")
    pdf_parser.add_argument("--db", default="livemind.db", help="Database path")

    # Video subcommand
    vid_parser = sub.add_parser("video", help="Export graph evolution as mp4 video")
    vid_parser.add_argument("session_id", help="Session ID to export")
    vid_parser.add_argument("-o", "--output", help="Output file path (default: <session_id>.mp4)")
    vid_parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    vid_parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    vid_parser.add_argument("--max-hold", type=float, default=3.0, help="Max seconds per snapshot (default: 3.0)")
    vid_parser.add_argument("--db", default="livemind.db", help="Database path")

    args = parser.parse_args()

    if args.command == "pdf":
        output = args.output or f"{args.session_id}.pdf"
        asyncio.run(export_pdf(args.session_id, output, args.db))

    elif args.command == "video":
        output = args.output or f"{args.session_id}.mp4"
        asyncio.run(export_video(
            args.session_id, output, args.db,
            fps=args.fps, speed=args.speed, max_hold=args.max_hold,
        ))


if __name__ == "__main__":
    main()
