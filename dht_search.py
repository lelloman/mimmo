#!/usr/bin/env python3
"""Quick CLI to search the Magnetico DHT dump and view torrent structures."""

import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

DB_PATH = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"

def human_size(size: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"

def build_tree(files: list[tuple[int, str]]) -> str:
    """Build a tree structure from file paths."""
    # Parse paths into a nested dict structure
    tree = {}
    for size, path in files:
        parts = path.split('/')
        current = tree
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        # Leaf node with size
        current[parts[-1]] = size

    # Render the tree
    lines = []
    def render(node, prefix="", is_last=True):
        items = sorted(node.items(), key=lambda x: (isinstance(x[1], dict), x[0].lower()))
        for i, (name, value) in enumerate(items):
            is_last_item = i == len(items) - 1
            connector = "└── " if is_last_item else "├── "

            if isinstance(value, dict):
                lines.append(f"{prefix}{connector}{name}/")
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                render(value, new_prefix, is_last_item)
            else:
                lines.append(f"{prefix}{connector}{name}  [{human_size(value)}]")

    render(tree)
    return "\n".join(lines)

def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[tuple]:
    """Search torrents using FTS5."""
    # Use FTS5 MATCH for fast full-text search
    cursor = conn.execute("""
        SELECT t.id, t.name, t.total_size,
               datetime(t.discovered_on, 'unixepoch') as discovered,
               (SELECT COUNT(*) FROM files WHERE torrent_id = t.id) as file_count
        FROM torrents t
        JOIN torrents_idx idx ON t.id = idx.rowid
        WHERE torrents_idx MATCH ?
        ORDER BY t.total_size DESC
        LIMIT ?
    """, (query, limit))
    return cursor.fetchall()

def get_files(conn: sqlite3.Connection, torrent_id: int) -> list[tuple]:
    """Get all files for a torrent."""
    cursor = conn.execute("""
        SELECT size, path FROM files
        WHERE torrent_id = ?
        ORDER BY path
    """, (torrent_id,))
    return cursor.fetchall()

def get_torrent(conn: sqlite3.Connection, torrent_id: int) -> tuple:
    """Get torrent details."""
    cursor = conn.execute("""
        SELECT id, name, total_size, datetime(discovered_on, 'unixepoch')
        FROM torrents WHERE id = ?
    """, (torrent_id,))
    return cursor.fetchone()

def interactive_search(conn: sqlite3.Connection):
    """Interactive search mode."""
    print("\n🔍 DHT Torrent Search (type 'q' to quit)\n")

    while True:
        try:
            query = input("Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() == 'q':
            break

        # Check if it's a number (selecting a result)
        if query.isdigit():
            torrent_id = int(query)
            torrent = get_torrent(conn, torrent_id)
            if torrent:
                print(f"\n{'='*60}")
                print(f"📦 {torrent[1]}")
                print(f"   Size: {human_size(torrent[2])} | Discovered: {torrent[3]}")
                print(f"{'='*60}\n")

                files = get_files(conn, torrent_id)
                if files:
                    print(build_tree(files))
                else:
                    print("(no files)")
                print()
            else:
                print(f"Torrent ID {torrent_id} not found\n")
            continue

        # Search
        results = search(conn, query)

        if not results:
            print("No results found.\n")
            continue

        print(f"\n{'ID':<10} {'Size':<10} {'Files':<8} Name")
        print("-" * 80)
        for row in results:
            tid, name, size, discovered, file_count = row
            name = name.decode('utf-8', errors='replace') if isinstance(name, bytes) else name
            # Truncate name if too long
            display_name = name[:55] + "..." if len(name) > 58 else name
            print(f"{tid:<10} {human_size(size):<10} {file_count:<8} {display_name}")
        print(f"\nEnter ID to view tree, or search again:\n")

def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.text_factory = lambda x: x.decode('utf-8', errors='replace')

    # If args provided, do single search
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

        # Check if it's "tree <id>"
        if sys.argv[1] == "tree" and len(sys.argv) > 2:
            torrent_id = int(sys.argv[2])
            torrent = get_torrent(conn, torrent_id)
            if torrent:
                print(f"📦 {torrent[1]} [{human_size(torrent[2])}]\n")
                files = get_files(conn, torrent_id)
                print(build_tree(files))
            else:
                print(f"Torrent {torrent_id} not found")
        else:
            results = search(conn, query)
            for row in results:
                tid, name, size, discovered, file_count = row
                print(f"{tid:<10} {human_size(size):<10} {file_count:<8} {name[:60]}")
    else:
        interactive_search(conn)

    conn.close()

if __name__ == "__main__":
    main()
