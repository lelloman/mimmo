#!/usr/bin/env python3
"""
Evaluate mimmo classification + GLiNER extraction on random samples.
Outputs a markdown table for review.
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

# Add parent dir for mimmo imports if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from gliner import GLiNER

# Use the raw magnetico dump
MAGNETICO_DB = Path.home() / "Downloads" / "dht-magnetico-dump" / "32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info" / "database.sqlite3"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "extraction_evaluation.md"


def ensure_str(val) -> str:
    """Convert bytes to str if needed."""
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return val

# GLiNER labels by medium type
AUDIO_LABELS = ["artist", "album", "year", "audio format", "bitrate"]
VIDEO_LABELS = ["title", "year", "season", "episode", "resolution", "codec"]


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def get_files_for_torrent(conn: sqlite3.Connection, torrent_id: int, n: int = 3) -> list[tuple[str, int]]:
    """Get top n largest files for a torrent."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT path, size FROM files
        WHERE torrent_id = ?
        ORDER BY size DESC
        LIMIT ?
    """, (torrent_id, n))
    return [(ensure_str(row[0]), row[1]) for row in cursor.fetchall()]


def run_mimmo(name: str, files: list[tuple[str, int]]) -> dict:
    """Run mimmo classification on a sample."""
    try:
        lines = [name]
        for path, size in files[:10]:
            lines.append(f"{path} ({format_size(size)})")

        input_text = "\n".join(lines)

        # Run mimmo CLI
        result = subprocess.run(
            ["cargo", "run", "--release", "--quiet", "--"],
            input=input_text,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=30
        )

        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception as e:
        pass

    return {"medium": "error", "confidence": 0}


def run_gliner(model: GLiNER, text: str, medium: str) -> dict:
    """Run GLiNER extraction based on medium type."""
    if medium == "audio":
        labels = AUDIO_LABELS
    elif medium == "video":
        labels = VIDEO_LABELS
    else:
        return {}

    try:
        entities = model.predict_entities(text, labels, threshold=0.3)
        result = {}
        for ent in entities:
            label = ent["label"]
            if label not in result or ent["score"] > result[label]["score"]:
                result[label] = {"text": ent["text"], "score": ent["score"]}
        return {k: v["text"] for k, v in result.items()}
    except:
        return {}


def main():
    print("Loading GLiNER model...")
    gliner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

    print(f"Connecting to magnetico DB: {MAGNETICO_DB}")
    conn = sqlite3.connect(MAGNETICO_DB)
    conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
    cursor = conn.cursor()

    # Get 100 random torrents
    print("Fetching random samples...")
    cursor.execute("""
        SELECT id, name, total_size
        FROM torrents
        ORDER BY RANDOM()
        LIMIT 100
    """)
    samples = cursor.fetchall()

    print(f"Processing {len(samples)} samples...")

    results = []
    for i, (torrent_id, name, total_size) in enumerate(samples):
        name = ensure_str(name)
        print(f"  [{i+1}/{len(samples)}] {name[:50]}...")

        # Get top files for this torrent
        top_files = get_files_for_torrent(conn, torrent_id, n=5)

        # Run mimmo
        mimmo_result = run_mimmo(name, top_files)

        # Run GLiNER on the torrent name
        gliner_result = run_gliner(gliner, name, mimmo_result.get("medium", ""))

        results.append({
            "name": name,
            "total_size": total_size,
            "top_files": [(f[0], format_size(f[1])) for f in top_files],
            "mimmo": mimmo_result,
            "gliner": gliner_result
        })

    conn.close()

    # Write markdown output
    print(f"Writing results to {OUTPUT_PATH}...")

    with open(OUTPUT_PATH, "w") as f:
        f.write("# Extraction Evaluation Results\n\n")
        f.write(f"Evaluated {len(results)} random samples from magnetico dump\n\n")

        # Summary stats by medium
        medium_counts = {}
        for r in results:
            m = r["mimmo"].get("medium", "error")
            medium_counts[m] = medium_counts.get(m, 0) + 1

        f.write("## Summary\n\n")
        f.write("### Medium Distribution\n\n")
        for m, count in sorted(medium_counts.items(), key=lambda x: -x[1]):
            f.write(f"- {m}: {count}\n")
        f.write("\n")

        # Detailed table
        f.write("## Detailed Results\n\n")

        for i, r in enumerate(results):
            f.write(f"### Sample {i+1}\n\n")

            # Input
            f.write(f"**Torrent Name:** `{r['name']}`\n\n")
            f.write(f"**Total Size:** {format_size(r['total_size'])}\n\n")

            if r["top_files"]:
                f.write("**Top Files:**\n")
                for fname, fsize in r["top_files"]:
                    # Truncate long filenames
                    display_name = fname if len(fname) <= 80 else fname[:77] + "..."
                    f.write(f"- `{display_name}` ({fsize})\n")
                f.write("\n")

            # Classification
            mimmo = r["mimmo"]
            f.write(f"**Mimmo Classification:** `{mimmo.get('medium', 'N/A')}`")
            if mimmo.get("subcategory"):
                f.write(f" / `{mimmo['subcategory']}`")
            f.write(f" (conf: {mimmo.get('confidence', 0):.2f})\n\n")

            # Extraction
            if r["gliner"]:
                f.write("**GLiNER Extraction:**\n")
                for k, v in r["gliner"].items():
                    f.write(f"- {k}: `{v}`\n")
            else:
                f.write("**GLiNER Extraction:** (none - not audio/video)\n")
            f.write("\n")

            f.write("---\n\n")

    print(f"Done! Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
