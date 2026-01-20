#!/usr/bin/env python3
"""
Match ground truth metadata against magnetico torrent database.

Searches the magnetico DHT dump for torrents matching movies/TV/albums
from our ground truth database, scores them, and stores high-confidence matches.

Usage:
    python scripts/match_torrents.py --limit 100  # Test with 100 entries
    python scripts/match_torrents.py              # Process all
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher

GROUND_TRUTH_DB = Path(__file__).parent.parent / "data" / "training_ground_truth.db"
MAGNETICO_DB = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"

# Size ranges for sanity checks (in bytes)
SIZE_RANGES = {
    "movie": (400_000_000, 80_000_000_000),   # 400MB - 80GB
    "tv": (100_000_000, 200_000_000_000),      # 100MB - 200GB (could be full series)
    "album": (30_000_000, 3_000_000_000),      # 30MB - 3GB
}

# Negative signals - torrents with these are likely not what we want
NEGATIVE_PATTERNS = [
    r'\bsample\b',
    r'\btrailer\b',
    r'\bteaser\b',
    r'\bscreener\b',
    r'\bcam\b',
    r'\bts\b',
    r'\bhdts\b',
    r'\bbonus\b',
    r'\bextras?\b',
    r'\bfeaturette\b',
    r'\binterview\b',
    r'\bmaking.?of\b',
    r'\bdocumentary\b',
    r'\bkaraoke\b',
    r'\binstrumental\b',
    r'\bremix\b',
    r'\bcover\b',
    r'\btribute\b',
]
NEGATIVE_RE = re.compile('|'.join(NEGATIVE_PATTERNS), re.IGNORECASE)

# Content type indicators
AUDIO_INDICATORS = [
    r'\bflac\b', r'\bmp3\b', r'\b320\s*kbps\b', r'\b320kbps\b', r'\baac\b',
    r'\balac\b', r'\bwav\b', r'\bogg\b', r'\bape\b', r'\bm4a\b',
    r'\b(16|24)\s*bit\b', r'\b(44|48|96)\.?1?\s*khz\b',
    r'\blossless\b', r'\bhi-?res\b', r'\bcd\s*rip\b', r'\bcdrip\b',
]
AUDIO_RE = re.compile('|'.join(AUDIO_INDICATORS), re.IGNORECASE)

VIDEO_INDICATORS = [
    r'\b1080p\b', r'\b720p\b', r'\b2160p\b', r'\b4k\b', r'\buhd\b',
    r'\bbluray\b', r'\bblu-ray\b', r'\bbdrip\b', r'\bbrrip\b',
    r'\bwebrip\b', r'\bweb-dl\b', r'\bhdtv\b', r'\bdvdrip\b',
    r'\bx264\b', r'\bx265\b', r'\bhevc\b', r'\bavc\b', r'\bh\.?264\b', r'\bh\.?265\b',
    r'\bmkv\b', r'\bavi\b', r'\bmp4\b',
    r'\byts\b', r'\byify\b', r'\brarbg\b', r'\bsparks\b',
]
VIDEO_RE = re.compile('|'.join(VIDEO_INDICATORS), re.IGNORECASE)


@dataclass
class GroundTruth:
    id: int
    type: str
    title: str
    artist: str | None
    year: int | None
    source: str


@dataclass
class TorrentMatch:
    torrent_name: str
    torrent_size: int
    score: float
    year_in_name: bool
    title_similarity: float


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    # Remove common prefixes/suffixes
    title = re.sub(r'^(the|a|an)\s+', '', title.lower())
    # Remove punctuation
    title = re.sub(r'[^\w\s]', ' ', title)
    # Collapse whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def extract_year_from_name(name: str) -> int | None:
    """Extract a year (1950-2030) from torrent name."""
    match = re.search(r'\b(19[5-9]\d|20[0-3]\d)\b', name)
    return int(match.group(1)) if match else None


def title_in_torrent(title: str, torrent_name: str) -> tuple[bool, float]:
    """Check if title appears in torrent name and calculate similarity."""
    norm_title = normalize_title(title)
    norm_torrent = normalize_title(torrent_name)

    # Check for exact substring match
    if norm_title in norm_torrent:
        return True, 1.0

    # Calculate similarity for partial matches
    # Split torrent name by common separators and check each part
    parts = re.split(r'[\.\-_\s\[\]\(\)]+', torrent_name.lower())
    title_words = norm_title.split()

    # Count how many title words appear in torrent parts
    matched_words = sum(1 for word in title_words if any(word in part for part in parts))
    word_ratio = matched_words / len(title_words) if title_words else 0

    # Also calculate sequence match ratio
    seq_ratio = SequenceMatcher(None, norm_title, norm_torrent[:len(norm_title)*2]).ratio()

    # Combine both scores
    similarity = max(word_ratio, seq_ratio)

    return similarity > 0.7, similarity


def score_match(gt: GroundTruth, torrent_name: str, torrent_size: int) -> TorrentMatch | None:
    """Score how well a torrent matches ground truth."""
    score = 0.0

    # Check for negative patterns
    if NEGATIVE_RE.search(torrent_name):
        return None

    # Size sanity check
    min_size, max_size = SIZE_RANGES.get(gt.type, (0, float('inf')))
    if not (min_size <= torrent_size <= max_size):
        return None

    # Content type verification
    has_audio_indicators = bool(AUDIO_RE.search(torrent_name))
    has_video_indicators = bool(VIDEO_RE.search(torrent_name))

    if gt.type == "album":
        # For albums: prefer audio indicators, reject if only video indicators
        if has_video_indicators and not has_audio_indicators:
            return None  # Likely a movie/TV, not an album
    elif gt.type in ("movie", "tv"):
        # For movies/TV: prefer video indicators, reject if only audio indicators
        if has_audio_indicators and not has_video_indicators:
            return None  # Likely a music release, not a video

    # Title match
    title_found, title_sim = title_in_torrent(gt.title, torrent_name)
    if not title_found:
        return None

    score += title_sim * 10  # Up to 10 points for title

    # Year match
    torrent_year = extract_year_from_name(torrent_name)
    year_in_name = False
    if gt.year and torrent_year:
        if torrent_year == gt.year:
            score += 5  # Exact year match
            year_in_name = True
        elif abs(torrent_year - gt.year) <= 1:
            score += 2  # Off by one year (sometimes happens)
            year_in_name = True
        else:
            # Year mismatch - likely wrong content
            return None

    # Artist match (for albums)
    if gt.artist and gt.type == "album":
        artist_found, artist_sim = title_in_torrent(gt.artist, torrent_name)
        if artist_found:
            score += artist_sim * 5  # Up to 5 points for artist

    # Size reasonableness bonus
    if gt.type == "movie":
        if 700_000_000 <= torrent_size <= 4_000_000_000:
            score += 2  # Typical movie size
        elif 4_000_000_000 <= torrent_size <= 15_000_000_000:
            score += 1  # High quality
    elif gt.type == "album":
        if 50_000_000 <= torrent_size <= 500_000_000:
            score += 2  # Typical album size

    # Content type indicator bonus
    if gt.type == "album" and has_audio_indicators:
        score += 3  # Confirmed audio content
    elif gt.type in ("movie", "tv") and has_video_indicators:
        score += 3  # Confirmed video content

    return TorrentMatch(
        torrent_name=torrent_name,
        torrent_size=torrent_size,
        score=score,
        year_in_name=year_in_name,
        title_similarity=title_sim,
    )


def build_fts_query(gt: GroundTruth) -> str | None:
    """Build FTS5 query for searching magnetico. Returns None if title is too generic."""
    # Skip very short or generic titles
    if len(gt.title) < 3:
        return None
    if gt.title.isdigit():  # Just a number like "1", "2"
        return None

    # Quote each word and join with AND
    words = gt.title.split()

    # For multi-word titles, require at least the important words
    if len(words) <= 3:
        query_parts = [f'"{w}"' for w in words if len(w) > 2]
    else:
        # For longer titles, just use first 3-4 significant words
        significant = [w for w in words if len(w) > 3][:4]
        query_parts = [f'"{w}"' for w in significant]

    # If no good words found, skip
    if not query_parts:
        return None

    # Add year if available
    if gt.year:
        query_parts.append(str(gt.year))

    # For albums, also require artist name if available
    if gt.type == "album" and gt.artist:
        artist_words = [w for w in gt.artist.split() if len(w) > 2][:2]
        query_parts.extend([f'"{w}"' for w in artist_words])

    return ' '.join(query_parts)


def search_magnetico(mag_conn: sqlite3.Connection, gt: GroundTruth, limit: int = 50) -> list[TorrentMatch]:
    """Search magnetico for matching torrents."""
    query = build_fts_query(gt)

    if query is None:
        return []  # Title too generic to search

    try:
        cursor = mag_conn.execute("""
            SELECT t.name, t.total_size
            FROM torrents t
            JOIN torrents_idx ON torrents_idx.rowid = t.id
            WHERE torrents_idx MATCH ?
            LIMIT ?
        """, (query, limit))

        matches = []
        for name, size in cursor:
            # Handle bytes encoding from some torrent names
            if isinstance(name, bytes):
                try:
                    name = name.decode('utf-8')
                except UnicodeDecodeError:
                    continue
            match = score_match(gt, name, size)
            if match and match.score >= 8:  # Minimum score threshold
                matches.append(match)

        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:10]  # Keep top 10 per entry

    except sqlite3.OperationalError as e:
        # FTS query syntax error
        return []


def human_size(size: int) -> str:
    """Convert bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def main():
    parser = argparse.ArgumentParser(description="Match ground truth to magnetico torrents")
    parser.add_argument("--limit", type=int, help="Limit number of ground truth entries to process")
    parser.add_argument("--type", choices=["movie", "tv", "album"], help="Only process this type")
    parser.add_argument("--min-score", type=float, default=8.0, help="Minimum match score")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show match details")
    args = parser.parse_args()

    # Connect to databases
    gt_conn = sqlite3.connect(GROUND_TRUTH_DB)
    mag_conn = sqlite3.connect(MAGNETICO_DB)

    print(f"Ground truth DB: {GROUND_TRUTH_DB}")
    print(f"Magnetico DB: {MAGNETICO_DB}")
    print("-" * 60)

    # Fetch ground truth entries
    query = "SELECT id, type, title, artist, year, source FROM ground_truth"
    params = []
    if args.type:
        query += " WHERE type = ?"
        params.append(args.type)
    if args.limit:
        query += " LIMIT ?"
        params.append(args.limit)

    cursor = gt_conn.execute(query, params)
    entries = [GroundTruth(*row) for row in cursor]
    print(f"Processing {len(entries)} ground truth entries...")
    print("-" * 60)

    # Process each entry
    total_matches = 0
    entries_with_matches = 0

    for i, gt in enumerate(entries):
        matches = search_magnetico(mag_conn, gt)

        if matches:
            entries_with_matches += 1
            total_matches += len(matches)

            # Store matches in database
            for match in matches:
                gt_conn.execute("""
                    INSERT OR IGNORE INTO matches
                    (ground_truth_id, torrent_name, torrent_size, match_score)
                    VALUES (?, ?, ?, ?)
                """, (gt.id, match.torrent_name, match.torrent_size, match.score))

            if args.verbose:
                print(f"\n[{gt.type}] {gt.title} ({gt.year}) - {len(matches)} matches")
                for m in matches[:3]:
                    print(f"  {m.score:.1f}: {m.torrent_name[:70]}... ({human_size(m.torrent_size)})")

        if (i + 1) % 50 == 0:
            gt_conn.commit()
            pct = (i + 1) / len(entries) * 100
            print(f"Progress: {i+1}/{len(entries)} ({pct:.0f}%) - {entries_with_matches} with matches, {total_matches} total")

    gt_conn.commit()

    print("-" * 60)
    print(f"Done!")
    print(f"  Entries processed: {len(entries)}")
    print(f"  Entries with matches: {entries_with_matches} ({entries_with_matches/len(entries)*100:.1f}%)")
    print(f"  Total matches: {total_matches}")
    print(f"  Avg matches per entry: {total_matches/entries_with_matches:.1f}" if entries_with_matches else "")

    # Show summary by type
    cursor = gt_conn.execute("""
        SELECT gt.type, COUNT(DISTINCT m.ground_truth_id), COUNT(*)
        FROM matches m
        JOIN ground_truth gt ON gt.id = m.ground_truth_id
        GROUP BY gt.type
    """)
    print("\nBy type:")
    for row in cursor:
        print(f"  {row[0]}: {row[1]} entries, {row[2]} matches")

    gt_conn.close()
    mag_conn.close()


if __name__ == "__main__":
    main()
