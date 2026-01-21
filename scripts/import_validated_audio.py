#!/usr/bin/env python3
"""
Import validated Spotify audio matches into the training ground truth database.

Usage:
    python scripts/import_validated_audio.py /tmp/validated_audio_big.jsonl
"""

import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "training_ground_truth.db"


def main():
    if len(sys.argv) < 2:
        print("Usage: python import_validated_audio.py <validated.jsonl>")
        sys.exit(1)

    input_file = sys.argv[1]
    conn = sqlite3.connect(DB_PATH)

    # Get initial counts
    initial_gt = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    initial_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"Initial: {initial_gt} ground truth, {initial_matches} matches")

    inserted_gt = 0
    inserted_matches = 0
    skipped = 0

    with open(input_file) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            album = data.get("spotify_album", "")
            artist = data.get("spotify_artist", "")
            year = data.get("spotify_year")
            torrent_name = data.get("torrent_name", "")
            torrent_size = data.get("torrent_size", 0)
            match_score = data.get("match_score", 10.0)

            if not album or not artist or not torrent_name:
                skipped += 1
                continue

            # Create a unique external_id from album+artist
            external_id = f"spotify:{artist[:50]}:{album[:50]}"

            # Insert ground truth (or get existing)
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO ground_truth
                    (type, title, artist, year, external_id, source, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("album", album, artist, year, external_id, "spotify", "{}"))

                if conn.total_changes > 0:
                    inserted_gt += 1
            except sqlite3.IntegrityError:
                pass

            # Get the ground truth ID
            gt_row = conn.execute(
                "SELECT id FROM ground_truth WHERE source='spotify' AND external_id=?",
                (external_id,)
            ).fetchone()

            if not gt_row:
                skipped += 1
                continue

            gt_id = gt_row[0]

            # Insert match
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO matches
                    (ground_truth_id, torrent_name, torrent_size, match_score)
                    VALUES (?, ?, ?, ?)
                """, (gt_id, torrent_name, torrent_size, match_score))

                if conn.total_changes > 0:
                    inserted_matches += 1
            except sqlite3.IntegrityError:
                pass

            if (inserted_gt + inserted_matches) % 10000 == 0:
                conn.commit()
                print(f"Progress: {inserted_gt} ground truth, {inserted_matches} matches")

    conn.commit()

    # Final counts
    final_gt = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    final_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    print(f"\nDone!")
    print(f"  New ground truth: {inserted_gt}")
    print(f"  New matches: {inserted_matches}")
    print(f"  Skipped: {skipped}")
    print(f"  Total ground truth: {final_gt}")
    print(f"  Total matches: {final_matches}")

    # Show by type
    cursor = conn.execute("""
        SELECT gt.type, gt.source, COUNT(DISTINCT gt.id), COUNT(m.id)
        FROM ground_truth gt
        LEFT JOIN matches m ON gt.id = m.ground_truth_id
        GROUP BY gt.type, gt.source
    """)
    print("\nBy type/source:")
    for row in cursor:
        print(f"  {row[0]}/{row[1]}: {row[2]} entries, {row[3]} matches")

    conn.close()


if __name__ == "__main__":
    main()
