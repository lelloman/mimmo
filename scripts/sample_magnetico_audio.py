#!/usr/bin/env python3
"""
Sample audio-looking torrents from magnetico for offline validation.
Outputs JSON lines that can be validated against Spotify catalog.

Usage:
    python scripts/sample_magnetico_audio.py --limit 50000 > audio_samples.jsonl
"""

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path

MAGNETICO_DB = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"

# Size range for albums
AUDIO_SIZE_RANGE = (30_000_000, 3_000_000_000)  # 30MB - 3GB

# Audio indicators
AUDIO_INDICATORS = re.compile(
    r'\b(flac|mp3|320\s*kbps|aac|alac|wav|ogg|ape|m4a|'
    r'(16|24)\s*bit|(44|48|96)\.?1?\s*khz|lossless|hi-?res|cd\s*rip)\b',
    re.IGNORECASE
)

VIDEO_INDICATORS = re.compile(
    r'\b(1080p|720p|2160p|4k|uhd|bluray|blu-ray|bdrip|brrip|'
    r'webrip|web-dl|hdtv|dvdrip|x264|x265|hevc|avc|h\.?264|h\.?265|'
    r'mkv|yts|yify|rarbg|sparks)\b',
    re.IGNORECASE
)

SKIP_PATTERNS = re.compile(
    r'\b(sample|trailer|karaoke|instrumental|xxx|porn|adult)\b',
    re.IGNORECASE
)


def extract_audio_metadata(name: str) -> dict | None:
    """Extract potential album metadata from torrent name."""
    if SKIP_PATTERNS.search(name):
        return None

    # Must have audio indicators
    if not AUDIO_INDICATORS.search(name):
        return None

    # Skip video content
    if VIDEO_INDICATORS.search(name) and not AUDIO_INDICATORS.search(name):
        return None

    # Extract year
    year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', name)
    year = int(year_match.group(1)) if year_match else None

    # Clean the name
    clean = name
    clean = re.sub(r'\[.*?\]', ' ', clean)
    clean = re.sub(r'\(.*?\)', ' ', clean)
    clean = re.sub(r'[\.\s](flac|mp3|320|aac|lossless|cd\s*rip|web).*$', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[\._]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Try to split by " - "
    if " - " in clean:
        parts = clean.split(" - ", 1)
        artist = parts[0].strip()
        album = parts[1].strip() if len(parts) > 1 else ""

        if year:
            album = re.sub(rf'\b{year}\b', '', album).strip()

        if artist and album and len(artist) > 1 and len(album) > 1:
            return {
                "artist": artist,
                "album": album,
                "year": year,
            }

    return None


def sample_audio_fts(mag_conn: sqlite3.Connection, limit: int) -> list[dict]:
    """Sample audio torrents using FTS."""
    audio_keywords = [
        "flac", "320kbps", "mp3", "lossless", "24bit", "album",
        "discography", "vinyl", "remaster", "deluxe", "aac",
        "wav", "dsd", "sacd", "hdtracks", "qobuz", "tidal",
        "192khz", "96khz", "44khz", "16bit", "studio",
    ]

    samples = []
    seen = set()

    for keyword in audio_keywords:
        if len(samples) >= limit:
            break

        # Multiple random offsets for variety
        for _ in range(50):  # More iterations per keyword
            if len(samples) >= limit:
                break

            offset = random.randint(0, 200000)  # Wider range

            try:
                cursor = mag_conn.execute("""
                    SELECT t.name, t.total_size
                    FROM torrents t
                    JOIN torrents_idx ON torrents_idx.rowid = t.id
                    WHERE torrents_idx MATCH ?
                    LIMIT 10000 OFFSET ?
                """, (keyword, offset))

                for name_bytes, size in cursor:
                    if len(samples) >= limit:
                        break

                    try:
                        name = name_bytes.decode('utf-8')
                    except (UnicodeDecodeError, AttributeError):
                        continue

                    if name in seen:
                        continue
                    seen.add(name)

                    if not (AUDIO_SIZE_RANGE[0] <= size <= AUDIO_SIZE_RANGE[1]):
                        continue

                    meta = extract_audio_metadata(name)
                    if meta:
                        meta["torrent_name"] = name
                        meta["torrent_size"] = size
                        samples.append(meta)

            except sqlite3.OperationalError:
                continue

    random.shuffle(samples)
    return samples[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50000)
    args = parser.parse_args()

    mag_conn = sqlite3.connect(MAGNETICO_DB)
    mag_conn.text_factory = bytes

    print(f"Sampling {args.limit} audio torrents...", file=__import__('sys').stderr)
    samples = sample_audio_fts(mag_conn, args.limit)
    print(f"Got {len(samples)} samples", file=__import__('sys').stderr)

    for sample in samples:
        print(json.dumps(sample, ensure_ascii=False))

    mag_conn.close()


if __name__ == "__main__":
    main()
