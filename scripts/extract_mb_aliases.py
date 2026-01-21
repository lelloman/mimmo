#!/usr/bin/env python3
"""Extract artist aliases from MusicBrainz dump into SQLite."""

import json
import sqlite3
import subprocess
import sys

DB_PATH = "/home/lelloman/lelloprojects/mimmo/data/training_ground_truth.db"
MB_ARTIST_ARCHIVE = "/home/lelloman/Downloads/musicbrainz/artist.tar.xz"


def main():
    conn = sqlite3.connect(DB_PATH)

    # Get all artist names we care about from ground truth
    cursor = conn.execute('SELECT DISTINCT artist FROM ground_truth WHERE artist IS NOT NULL')
    target_artists = set(row[0].lower().strip() for row in cursor)
    print(f'Target artists from ground truth: {len(target_artists)}', flush=True)

    # Sample some targets
    for t in list(target_artists)[:5]:
        print(f'  Sample target: "{t}"', flush=True)

    # Extract and process
    proc = subprocess.Popen(
        ['tar', '-xf', MB_ARTIST_ARCHIVE, '-O', 'mbdump/artist'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    inserted = 0
    matched_artists = 0
    total = 0

    for line in proc.stdout:
        total += 1
        if total % 500000 == 0:
            print(f'Processed {total} artists, matched {matched_artists}...', flush=True)

        try:
            data = json.loads(line.decode('utf-8'))
            name = data.get('name', '')
            mb_id = data.get('id', '')
            aliases = data.get('aliases', [])

            # Check if this artist matches our targets (by canonical name or any alias)
            all_names = [name] + [a.get('name', '') for a in aliases if a.get('name')]

            matched = False
            for n in all_names:
                if n.lower().strip() in target_artists:
                    matched = True
                    break

            if matched:
                matched_artists += 1
                # Insert canonical name as self-alias
                conn.execute(
                    'INSERT OR IGNORE INTO artist_aliases (canonical_name, alias, mb_id) VALUES (?, ?, ?)',
                    (name, name, mb_id)
                )
                inserted += 1
                # Insert all aliases
                for alias in aliases:
                    alias_name = alias.get('name', '')
                    if alias_name:
                        conn.execute(
                            'INSERT OR IGNORE INTO artist_aliases (canonical_name, alias, mb_id) VALUES (?, ?, ?)',
                            (name, alias_name, mb_id)
                        )
                        inserted += 1

        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    proc.wait()
    conn.commit()
    conn.close()
    print(f'Done: processed {total}, matched {matched_artists} artists, inserted {inserted} alias records', flush=True)


if __name__ == "__main__":
    main()
