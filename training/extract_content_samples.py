#!/usr/bin/env python3
"""Extract content type samples from Magnetico database for BERT classifier training."""

import json
import random
import re
import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "content_classifier.jsonl"

TARGET_PER_CLASS = 5000  # Samples per content type

# File extension patterns for each content type
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.ogg', '.m4a', '.wav', '.ape', '.wma', '.aac', '.opus', '.wv'}
VIDEO_EXTENSIONS = {'.mkv', '.avi', '.mp4', '.mov', '.wmv', '.m4v', '.webm', '.ts', '.vob'}
SOFTWARE_EXTENSIONS = {'.exe', '.msi', '.iso', '.dmg', '.deb', '.rpm', '.apk', '.app', '.pkg'}
BOOK_EXTENSIONS = {'.epub', '.mobi', '.pdf', '.azw', '.azw3', '.djvu', '.cbr', '.cbz'}
ARCHIVE_EXTENSIONS = {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'}

# Name patterns for adult content detection (case-insensitive)
ADULT_PATTERNS = [
    r'\bxxx\b', r'\bporn', r'\badult\b', r'\bsex\b', r'\bnude', r'\berotic',
    r'\bhentai\b', r'\bjav\b', r'\bbrazzers\b', r'\bbangbros\b', r'\bnaughty',
    r'\bmilf\b', r'\blezb', r'\bgay\s*porn', r'\bfetish', r'\bplayboy\b',
]
ADULT_RE = re.compile('|'.join(ADULT_PATTERNS), re.IGNORECASE)

# Movie/TV patterns
MOVIE_PATTERNS = [
    r'\b(720p|1080p|2160p|4k)\b', r'\b(bluray|bdrip|brrip|dvdrip|webrip|hdtv)\b',
    r'\b(x264|x265|hevc|h\.?264|h\.?265)\b', r'\bS\d{2}E\d{2}\b',  # S01E01
    r'\b(complete\s*series|season\s*\d+)\b',
]
MOVIE_RE = re.compile('|'.join(MOVIE_PATTERNS), re.IGNORECASE)

# Software patterns
SOFTWARE_PATTERNS = [
    r'\b(crack|keygen|patch|serial|activat|portable)\b',
    r'\b(setup|install|windows|linux|macos|android)\b',
    r'\bv?\d+\.\d+\.\d+\b',  # Version numbers like v1.2.3
]
SOFTWARE_RE = re.compile('|'.join(SOFTWARE_PATTERNS), re.IGNORECASE)


def build_tree(files: list[tuple[int, str | bytes]]) -> str:
    """Build a simple file listing from paths."""
    lines = []
    for size, path in sorted(files, key=lambda x: x[1] if isinstance(x[1], str) else x[1].decode('utf-8', errors='replace')):
        if isinstance(path, bytes):
            path = path.decode('utf-8', errors='replace')
        lines.append(path)
    return '\n'.join(lines)


def count_extensions(files: list[tuple[int, str | bytes]]) -> dict[str, int]:
    """Count file extensions in a torrent."""
    counts = defaultdict(int)
    for _, path in files:
        if isinstance(path, bytes):
            path = path.decode('utf-8', errors='replace')
        ext = Path(path).suffix.lower()
        if ext:
            counts[ext] += 1
    return dict(counts)


def get_total_size_by_ext(files: list[tuple[int, str | bytes]], extensions: set) -> int:
    """Get total size of files matching given extensions."""
    total = 0
    for size, path in files:
        if isinstance(path, bytes):
            path = path.decode('utf-8', errors='replace')
        ext = Path(path).suffix.lower()
        if ext in extensions:
            total += size
    return total


def classify_torrent(name: str | bytes, files: list[tuple[int, str]], total_size: int) -> str | None:
    """
    Classify a torrent based on name patterns and file extensions.
    Returns content type or None if uncertain.
    """
    # Ensure name is string
    if isinstance(name, bytes):
        name = name.decode('utf-8', errors='replace')
    name_lower = name.lower()
    ext_counts = count_extensions(files)
    total_files = len(files)

    if total_files == 0:
        return None

    # Count files by category
    audio_count = sum(ext_counts.get(ext, 0) for ext in AUDIO_EXTENSIONS)
    video_count = sum(ext_counts.get(ext, 0) for ext in VIDEO_EXTENSIONS)
    software_count = sum(ext_counts.get(ext, 0) for ext in SOFTWARE_EXTENSIONS)
    book_count = sum(ext_counts.get(ext, 0) for ext in BOOK_EXTENSIONS)

    audio_pct = audio_count / total_files if total_files > 0 else 0
    video_pct = video_count / total_files if total_files > 0 else 0

    # Adult content: check name patterns first
    if ADULT_RE.search(name):
        # Confirm with video files
        if video_count > 0 or '.jpg' in ext_counts or '.jpeg' in ext_counts:
            return 'porn'

    # Music: mostly audio files, multiple tracks
    if audio_pct > 0.5 and audio_count >= 3:
        # Not a movie with soundtrack
        if video_count == 0:
            return 'music'

    # Movie/TV: video files with typical naming patterns
    if video_count > 0:
        video_size = get_total_size_by_ext(files, VIDEO_EXTENSIONS)
        # Video should be majority of content
        if video_size > total_size * 0.7:
            if MOVIE_RE.search(name):
                return 'movie'
            # Large video files without specific patterns
            if video_count <= 3 and video_size > 500 * 1024 * 1024:  # >500MB
                return 'movie'

    # Software: executables, ISOs, or software naming patterns
    if software_count > 0:
        if SOFTWARE_RE.search(name):
            return 'software'
        # ISOs are usually software
        if ext_counts.get('.iso', 0) > 0:
            return 'software'

    # Books: ebook formats
    if book_count > 0:
        book_pct = book_count / total_files
        # Mostly books, or clear ebook collection
        if book_pct > 0.5 or (book_count >= 3 and 'ebook' in name_lower):
            return 'book'
        # Single PDF could be book or document
        if book_count == 1 and ext_counts.get('.pdf', 0) == 1:
            if total_size < 100 * 1024 * 1024:  # <100MB typical for books
                return 'book'

    # Fallback checks
    if audio_pct > 0.3 and audio_count >= 3:
        return 'music'

    if video_count > 0 and video_pct > 0.3:
        return 'movie'

    return 'other'


def get_sample_torrents(conn: sqlite3.Connection, limit_per_scan: int = 50000) -> dict[str, list]:
    """Sample torrents for each content type."""
    print("Scanning torrents for content classification...")

    samples = defaultdict(list)
    target_total = TARGET_PER_CLASS * 6  # 6 content types

    # Use random sampling via ROWID
    cursor = conn.execute("SELECT MAX(id) FROM torrents")
    max_id = cursor.fetchone()[0]

    scanned = 0
    batch_size = 10000
    attempts = 0
    max_attempts = 50  # Prevent infinite loops

    while sum(len(v) for v in samples.values()) < target_total and attempts < max_attempts:
        attempts += 1

        # Get random batch of IDs
        random_ids = random.sample(range(1, max_id + 1), min(batch_size, max_id))
        placeholders = ','.join('?' * len(random_ids))

        cursor = conn.execute(f"""
            SELECT t.id, t.name, t.total_size
            FROM torrents t
            WHERE t.id IN ({placeholders})
        """, random_ids)

        torrents = cursor.fetchall()

        for tid, name, total_size in torrents:
            scanned += 1

            if scanned % 5000 == 0:
                counts = {k: len(v) for k, v in samples.items()}
                print(f"  Scanned {scanned:,}, samples: {counts}")

            # Check if we have enough of each type
            all_full = all(len(samples[t]) >= TARGET_PER_CLASS
                          for t in ['music', 'movie', 'software', 'book', 'porn', 'other'])
            if all_full:
                break

            # Get files for this torrent
            file_cursor = conn.execute("""
                SELECT size, path FROM files WHERE torrent_id = ? ORDER BY path
            """, (tid,))
            files = file_cursor.fetchall()

            if not files:
                continue

            # Classify
            label = classify_torrent(name, files, total_size)
            if label is None:
                continue

            # Skip if we have enough of this type
            if len(samples[label]) >= TARGET_PER_CLASS:
                continue

            # Build sample
            # Ensure name is string
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            sample = {
                'id': tid,
                'name': name,
                'tree': build_tree(files),
                'label': label
            }
            samples[label].append(sample)

        # Check if we've reached targets
        all_full = all(len(samples[t]) >= TARGET_PER_CLASS
                      for t in ['music', 'movie', 'software', 'book', 'porn', 'other'])
        if all_full:
            break

    return dict(samples)


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    random.seed(42)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode('utf-8', errors='replace')

    samples = get_sample_torrents(conn)
    conn.close()

    # Report counts
    print("\nSamples collected:")
    total = 0
    for label in ['music', 'movie', 'software', 'book', 'porn', 'other']:
        count = len(samples.get(label, []))
        print(f"  {label}: {count:,}")
        total += count
    print(f"  Total: {total:,}")

    # Combine and shuffle
    all_samples = []
    for label, items in samples.items():
        all_samples.extend(items)
    random.shuffle(all_samples)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(all_samples):,} samples to {OUTPUT_FILE}")

    # Show examples
    print("\nExample samples:")
    for label in ['music', 'movie', 'software', 'book', 'other']:
        items = samples.get(label, [])
        if items:
            s = random.choice(items)
            print(f"\n{'='*60}")
            print(f"Label: {label}")
            print(f"Name: {s['name']}")
            tree_preview = s['tree'][:300]
            if len(s['tree']) > 300:
                tree_preview += '...'
            print(f"Files:\n{tree_preview}")


if __name__ == '__main__':
    main()
