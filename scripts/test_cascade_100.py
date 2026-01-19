#!/usr/bin/env python3
"""Test the cascade classifier on 100 random samples from magnetico database."""

import sqlite3
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import os

MAGNETICO_DB = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"
MIMMO_BIN = Path(__file__).parent.parent / "target/release/mimmo"
OUTPUT_FILE = Path(__file__).parent.parent / "cascade_test_results.txt"

def ensure_str(val):
    """Convert bytes to string if needed."""
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return val

def get_random_torrents(n=100):
    """Get random torrents with their files from the database."""
    conn = sqlite3.connect(MAGNETICO_DB)
    conn.text_factory = bytes
    cursor = conn.cursor()

    # Get random torrent IDs
    cursor.execute("SELECT id, name FROM torrents ORDER BY RANDOM() LIMIT ?", (n,))
    torrents_raw = cursor.fetchall()

    torrents = []
    for tid, name in torrents_raw:
        name = ensure_str(name)

        # Get files for this torrent
        cursor.execute("SELECT path, size FROM files WHERE torrent_id = ?", (tid,))
        files = []
        for f in cursor.fetchall():
            files.append({
                'path': ensure_str(f[0]),
                'size': f[1] or 0
            })

        torrents.append({
            'id': tid,
            'name': name,
            'files': files
        })

    conn.close()
    return torrents

def classify_torrents_batch(torrents):
    """Classify torrents using mimmo cascade in interactive mode with file info."""
    # Create temp directory structure for each torrent to simulate real files
    results = []

    for t in torrents:
        # Create a temp dir with the torrent structure
        with tempfile.TemporaryDirectory() as tmpdir:
            torrent_dir = Path(tmpdir) / t['name'][:100]  # Truncate long names
            torrent_dir.mkdir(parents=True, exist_ok=True)

            # Create files with correct names and sizes
            for f in t['files'][:50]:  # Limit to 50 files
                file_path = torrent_dir / f['path']
                file_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # Write actual bytes to set file size (capped at 1MB to not fill disk)
                    size = min(f['size'], 1024 * 1024)
                    with open(file_path, 'wb') as fp:
                        fp.write(b'\x00' * size)
                except:
                    pass

            # Run mimmo on the directory
            try:
                result = subprocess.run(
                    [str(MIMMO_BIN), "--cascade", str(torrent_dir)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    results.append(json.loads(result.stdout.strip()))
                else:
                    results.append({"error": result.stderr.strip()})
            except subprocess.TimeoutExpired:
                results.append({"error": "timeout"})
            except Exception as e:
                results.append({"error": str(e)})

    return results

def format_size(size):
    """Format size in human readable form."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"

def main():
    print(f"Fetching 100 random torrents from magnetico database...")
    torrents = get_random_torrents(100)
    print(f"Got {len(torrents)} torrents")
    print(f"Classifying with cascade (creating temp directories with file structure)...")

    results = []
    for i, t in enumerate(torrents):
        # Classify one at a time to show progress
        classification = classify_torrents_batch([t])[0]
        results.append({
            'torrent': t,
            'classification': classification
        })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/100...")

    # Write results to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"Cascade Classifier Test Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Samples: {len(results)}\n")
        f.write("=" * 80 + "\n\n")

        # Summary stats
        by_medium = {}
        by_source = {}
        for r in results:
            c = r['classification']
            if 'error' not in c:
                medium = c.get('medium', 'unknown')
                source = c.get('source', 'unknown')
                by_medium[medium] = by_medium.get(medium, 0) + 1
                by_source[source] = by_source.get(source, 0) + 1
            else:
                by_medium['error'] = by_medium.get('error', 0) + 1

        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write("By Medium:\n")
        for k, v in sorted(by_medium.items(), key=lambda x: -x[1]):
            f.write(f"  {k}: {v}\n")
        f.write("\nBy Source (stage):\n")
        for k, v in sorted(by_source.items(), key=lambda x: -x[1]):
            f.write(f"  {k}: {v}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # Individual results
        f.write("DETAILED RESULTS\n")
        f.write("-" * 40 + "\n\n")

        for i, r in enumerate(results):
            t = r['torrent']
            c = r['classification']

            f.write(f"[{i+1}] {t['name'][:100]}\n")

            if t['files']:
                total_size = sum(file['size'] for file in t['files'])
                f.write(f"    Files: {len(t['files'])} ({format_size(total_size)} total)\n")
                # Show top 3 files by size
                sorted_files = sorted(t['files'], key=lambda x: -x['size'])[:3]
                for file in sorted_files:
                    f.write(f"      - {file['path']} ({format_size(file['size'])})\n")
            else:
                f.write(f"    Files: (none in database)\n")

            if 'error' in c:
                f.write(f"    => ERROR: {c['error']}\n")
            else:
                f.write(f"    => {c['medium'].upper()} (conf={c['confidence']}, source={c.get('source', '?')})\n")

            f.write("\n")

    print(f"\nResults written to: {OUTPUT_FILE}")
    print(f"\nSummary:")
    print(f"  By Medium: {by_medium}")
    print(f"  By Source: {by_source}")

if __name__ == "__main__":
    main()
