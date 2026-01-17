#!/usr/bin/env python3
"""Extract music torrent directory structures from Magnetico database."""

import json
import random
import sqlite3
from pathlib import Path

DB_PATH = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "directory_samples.jsonl"

TARGET_SAMPLES = 10000

# Audio file extensions
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.ogg', '.m4a', '.wav', '.ape', '.wma', '.aac', '.opus', '.wv'}


def build_tree(files: list[tuple[int, str]]) -> str:
    """Build a simple tree string from file paths."""
    lines = []
    for size, path in sorted(files, key=lambda x: x[1]):
        lines.append(path)
    return '\n'.join(lines)


def get_music_torrent_ids(conn: sqlite3.Connection, limit: int = 100000) -> list[int]:
    """Get IDs of torrents containing audio files."""
    print("Finding music torrents...")

    # Build extension pattern
    ext_conditions = " OR ".join([f"f.path LIKE '%{ext}'" for ext in AUDIO_EXTENSIONS])

    cursor = conn.execute(f"""
        SELECT DISTINCT t.id
        FROM torrents t
        JOIN files f ON f.torrent_id = t.id
        WHERE {ext_conditions}
        LIMIT ?
    """, (limit,))

    ids = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(ids):,} music torrents")
    return ids


def get_torrent_info(conn: sqlite3.Connection, torrent_id: int) -> dict | None:
    """Get torrent name and files."""
    cursor = conn.execute("""
        SELECT name, total_size FROM torrents WHERE id = ?
    """, (torrent_id,))
    row = cursor.fetchone()
    if not row:
        return None

    name, total_size = row

    cursor = conn.execute("""
        SELECT size, path FROM files WHERE torrent_id = ? ORDER BY path
    """, (torrent_id,))
    files = cursor.fetchall()

    return {
        'id': torrent_id,
        'name': name,
        'total_size': total_size,
        'files': files
    }


def extract_samples():
    """Extract music torrent samples."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode('utf-8', errors='replace')

    # Get music torrent IDs
    all_ids = get_music_torrent_ids(conn, limit=200000)

    # Sample randomly
    if len(all_ids) > TARGET_SAMPLES * 2:
        sampled_ids = random.sample(all_ids, TARGET_SAMPLES * 2)
    else:
        sampled_ids = all_ids

    print(f"Processing {len(sampled_ids):,} torrents...")

    samples = []
    for i, tid in enumerate(sampled_ids):
        if i % 1000 == 0:
            print(f"  Processed {i:,}, collected {len(samples):,} samples")

        if len(samples) >= TARGET_SAMPLES:
            break

        info = get_torrent_info(conn, tid)
        if not info or not info['files']:
            continue

        # Filter: at least 3 audio files
        audio_files = [f for f in info['files']
                       if any(f[1].lower().endswith(ext) for ext in AUDIO_EXTENSIONS)]
        if len(audio_files) < 3:
            continue

        sample = {
            'id': info['id'],
            'name': info['name'],
            'tree': build_tree(info['files'])
        }
        samples.append(sample)

    conn.close()

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Wrote {len(samples):,} samples to {OUTPUT_FILE}")

    # Print samples
    print("\nSample entries:")
    for s in random.sample(samples, min(3, len(samples))):
        print(f"\n{'='*60}")
        print(f"Name: {s['name']}")
        print(f"Files:\n{s['tree'][:500]}...")


if __name__ == '__main__':
    random.seed(42)
    extract_samples()
