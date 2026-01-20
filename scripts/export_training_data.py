#!/usr/bin/env python3
"""
Export matched torrents as training samples.

Reads the matches from the database and exports them in a format
suitable for training a metadata extraction model.

Usage:
    python scripts/export_training_data.py --output data/training_samples.jsonl
"""

import argparse
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "training_ground_truth.db"


def export_samples(conn: sqlite3.Connection, output_path: Path, min_score: float = 10.0):
    """Export high-confidence matches as training samples."""

    cursor = conn.execute("""
        SELECT
            gt.type,
            gt.title,
            gt.original_title,
            gt.artist,
            gt.year,
            m.torrent_name,
            m.torrent_size,
            m.match_score
        FROM matches m
        JOIN ground_truth gt ON gt.id = m.ground_truth_id
        WHERE m.match_score >= ?
        ORDER BY gt.type, gt.title, m.match_score DESC
    """, (min_score,))

    samples = []
    seen = set()  # Deduplicate by torrent name

    for row in cursor:
        gt_type, title, original_title, artist, year, torrent_name, torrent_size, score = row

        # Skip duplicates
        if torrent_name in seen:
            continue
        seen.add(torrent_name)

        # Build label based on content type
        if gt_type == "movie":
            label = {
                "type": "video/movie",
                "title": title,
            }
            if year:
                label["year"] = year
        elif gt_type == "tv":
            label = {
                "type": "video/series",
                "series_title": title,
            }
            if year:
                label["year"] = year
        elif gt_type == "album":
            label = {
                "type": "audio/album",
                "album_name": title,
            }
            if artist:
                label["artist"] = artist
            if year:
                label["year"] = year
        else:
            continue

        sample = {
            "input": torrent_name,
            "label": label,
            "metadata": {
                "match_score": score,
                "torrent_size": torrent_size,
                "source": "magnetico_match"
            }
        }
        samples.append(sample)

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    return samples


def print_stats(samples: list[dict]):
    """Print statistics about exported samples."""
    by_type = {}
    for s in samples:
        t = s["label"]["type"]
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\nExported {len(samples)} samples:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")


def print_examples(samples: list[dict], n: int = 5):
    """Print example samples."""
    print(f"\nExample samples:")
    for s in samples[:n]:
        print(f"\n  Input: {s['input'][:70]}...")
        print(f"  Label: {s['label']}")


def main():
    parser = argparse.ArgumentParser(description="Export training samples")
    parser.add_argument("--output", "-o", type=str, default="data/training_samples.jsonl",
                        help="Output file path (JSONL format)")
    parser.add_argument("--min-score", type=float, default=10.0,
                        help="Minimum match score to include")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    output_path = Path(args.output)

    print(f"Database: {args.db}")
    print(f"Min score: {args.min_score}")
    print(f"Output: {output_path}")
    print("-" * 60)

    samples = export_samples(conn, output_path, args.min_score)

    print_stats(samples)
    print_examples(samples)

    print(f"\n✓ Wrote {len(samples)} samples to {output_path}")

    conn.close()


if __name__ == "__main__":
    main()
