#!/usr/bin/env python3
"""
Reverse-match: Sample torrents from magnetico and validate against metadata APIs.

Instead of fetching popular metadata and matching to magnetico, this script:
1. Samples torrents that look like video/audio content
2. Extracts potential title/artist/year via heuristics
3. Validates against TMDB/MusicBrainz
4. Stores confirmed matches

This gives us more diverse training data beyond just "popular" content.

Usage:
    python scripts/reverse_match_magnetico.py --video 10000 --audio 10000
"""

import argparse
import json
import random
import re
import sqlite3
import time
from pathlib import Path
from difflib import SequenceMatcher

import requests

# Paths
MAGNETICO_DB = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"
GROUND_TRUTH_DB = Path(__file__).parent.parent / "data" / "training_ground_truth.db"

# APIs
TMDB_API_KEY = "543ba71a128563c1afb45daaa84df1fb"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
MB_BASE_URL = "https://musicbrainz.org/ws/2"
MB_USER_AGENT = "MimmoTrainingDataFetcher/1.0 (https://github.com/user/mimmo)"

# Content type indicators
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

# Size ranges (bytes)
VIDEO_SIZE_RANGE = (400_000_000, 80_000_000_000)  # 400MB - 80GB
AUDIO_SIZE_RANGE = (30_000_000, 3_000_000_000)    # 30MB - 3GB

# Skip patterns
SKIP_PATTERNS = re.compile(
    r'\b(sample|trailer|teaser|screener|cam\b|ts\b|hdts|bonus|extras?|'
    r'featurette|interview|making.?of|karaoke|instrumental|xxx|porn|adult)\b',
    re.IGNORECASE
)


def extract_video_metadata(name: str) -> dict | None:
    """Extract potential movie/TV metadata from torrent name."""
    # Skip unwanted content
    if SKIP_PATTERNS.search(name):
        return None

    # Must have video indicators
    if not VIDEO_INDICATORS.search(name):
        return None

    # Extract year
    year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', name)
    year = int(year_match.group(1)) if year_match else None

    # Clean the name to extract title
    clean = name

    # Remove release group tags in brackets
    clean = re.sub(r'\[.*?\]', ' ', clean)
    clean = re.sub(r'\(.*?\)', ' ', clean)

    # Remove common suffixes starting with quality/codec
    clean = re.sub(r'[\.\s](1080p|720p|2160p|4k|bluray|bdrip|webrip|web-dl|hdtv|dvdrip|x264|x265|hevc).*$', '', clean, flags=re.IGNORECASE)

    # Remove year from title if present
    if year:
        clean = re.sub(rf'[\.\s]{year}[\.\s]?.*$', '', clean)

    # Replace dots/underscores with spaces
    clean = re.sub(r'[\._]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    if len(clean) < 2:
        return None

    # Check if it's TV (has season/episode markers)
    is_tv = bool(re.search(r's\d{1,2}e\d{1,2}|season|\d{1,2}x\d{1,2}', name, re.IGNORECASE))

    return {
        "type": "tv" if is_tv else "movie",
        "title": clean,
        "year": year,
    }


def extract_audio_metadata(name: str) -> dict | None:
    """Extract potential album metadata from torrent name."""
    if SKIP_PATTERNS.search(name):
        return None

    # Must have audio indicators (or no video indicators)
    has_audio = bool(AUDIO_INDICATORS.search(name))
    has_video = bool(VIDEO_INDICATORS.search(name))

    if has_video and not has_audio:
        return None
    if not has_audio:
        return None  # Require explicit audio markers

    # Extract year
    year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', name)
    year = int(year_match.group(1)) if year_match else None

    # Try to extract artist - album patterns
    # Common formats:
    #   Artist - Album (Year)
    #   Artist_-_Album-Year
    #   (Year) Artist - Album

    clean = name

    # Remove brackets content
    clean = re.sub(r'\[.*?\]', ' ', clean)
    clean = re.sub(r'\(.*?\)', ' ', clean)

    # Remove audio quality tags
    clean = re.sub(r'[\.\s](flac|mp3|320|aac|lossless|cd\s*rip|web).*$', '', clean, flags=re.IGNORECASE)

    # Replace separators
    clean = re.sub(r'[\._]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Try to split by " - "
    if " - " in clean:
        parts = clean.split(" - ", 1)
        artist = parts[0].strip()
        album = parts[1].strip() if len(parts) > 1 else ""

        # Remove year from album if present
        if year:
            album = re.sub(rf'\b{year}\b', '', album).strip()

        if artist and album and len(artist) > 1 and len(album) > 1:
            return {
                "type": "album",
                "title": album,
                "artist": artist,
                "year": year,
            }

    return None


def validate_movie_tmdb(title: str, year: int | None, session: requests.Session) -> dict | None:
    """Search TMDB to validate a movie exists."""
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
    }
    if year:
        params["year"] = year

    resp = session.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=15)
    if resp.status_code != 200:
        return None

    results = resp.json().get("results", [])
    if not results:
        return None

    # Check if any result matches well
    for result in results[:5]:
        result_title = result.get("title", "")
        result_year = None
        release_date = result.get("release_date", "")
        if release_date and len(release_date) >= 4:
            result_year = int(release_date[:4])

        # Calculate similarity
        sim = SequenceMatcher(None, title.lower(), result_title.lower()).ratio()

        # Accept if title is similar and year matches (if we have both)
        if sim > 0.8:
            if year and result_year and abs(year - result_year) <= 1:
                return {
                    "title": result_title,
                    "year": result_year,
                    "tmdb_id": str(result["id"]),
                    "type": "movie",
                }
            elif not year or not result_year:
                return {
                    "title": result_title,
                    "year": result_year,
                    "tmdb_id": str(result["id"]),
                    "type": "movie",
                }

    return None


def validate_tv_tmdb(title: str, year: int | None, session: requests.Session) -> dict | None:
    """Search TMDB to validate a TV series exists."""
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
    }
    if year:
        params["first_air_date_year"] = year

    resp = session.get(f"{TMDB_BASE_URL}/search/tv", params=params, timeout=15)
    if resp.status_code != 200:
        return None

    results = resp.json().get("results", [])
    if not results:
        return None

    for result in results[:5]:
        result_title = result.get("name", "")
        result_year = None
        first_air = result.get("first_air_date", "")
        if first_air and len(first_air) >= 4:
            result_year = int(first_air[:4])

        sim = SequenceMatcher(None, title.lower(), result_title.lower()).ratio()

        if sim > 0.8:
            return {
                "title": result_title,
                "year": result_year,
                "tmdb_id": str(result["id"]),
                "type": "tv",
            }

    return None


def validate_album_musicbrainz(title: str, artist: str, year: int | None, session: requests.Session) -> dict | None:
    """Search MusicBrainz to validate an album exists."""
    query = f'release:"{title}" AND artist:"{artist}"'
    params = {
        "query": query,
        "fmt": "json",
        "limit": 10,
    }

    resp = session.get(f"{MB_BASE_URL}/release-group", params=params, timeout=15)
    if resp.status_code == 503:
        time.sleep(2)
        resp = session.get(f"{MB_BASE_URL}/release-group", params=params, timeout=15)

    if resp.status_code != 200:
        return None

    results = resp.json().get("release-groups", [])
    if not results:
        return None

    for result in results[:5]:
        result_title = result.get("title", "")

        # Get artist name
        artist_credit = result.get("artist-credit", [])
        result_artist = artist_credit[0].get("name", "") if artist_credit else ""

        # Get year
        first_release = result.get("first-release-date", "")
        result_year = int(first_release[:4]) if first_release and len(first_release) >= 4 else None

        # Calculate similarities
        title_sim = SequenceMatcher(None, title.lower(), result_title.lower()).ratio()
        artist_sim = SequenceMatcher(None, artist.lower(), result_artist.lower()).ratio()

        if title_sim > 0.7 and artist_sim > 0.7:
            return {
                "title": result_title,
                "artist": result_artist,
                "year": result_year,
                "mb_id": result.get("id"),
                "type": "album",
            }

    return None


def sample_video_torrents_fts(mag_conn: sqlite3.Connection, limit: int) -> list[tuple]:
    """Sample video torrents using FTS for fast filtering."""
    # Video quality keywords to search for
    video_keywords = [
        "1080p", "720p", "2160p", "bluray", "bdrip", "webrip", "web-dl",
        "hdtv", "x264", "x265", "hevc", "yts", "rarbg"
    ]

    samples = []
    seen = set()

    for keyword in video_keywords:
        if len(samples) >= limit:
            break

        # Random offset within results for this keyword
        offset = random.randint(0, 10000)

        try:
            cursor = mag_conn.execute("""
                SELECT t.name, t.total_size
                FROM torrents t
                JOIN torrents_idx ON torrents_idx.rowid = t.id
                WHERE torrents_idx MATCH ?
                LIMIT 5000 OFFSET ?
            """, (keyword, offset))

            for name_bytes, size in cursor:
                if len(samples) >= limit:
                    break

                try:
                    name = name_bytes.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    continue

                # Skip duplicates
                if name in seen:
                    continue
                seen.add(name)

                # Check size range
                if not (VIDEO_SIZE_RANGE[0] <= size <= VIDEO_SIZE_RANGE[1]):
                    continue

                # Verify it has video indicators
                if VIDEO_INDICATORS.search(name):
                    samples.append((name, size))

        except sqlite3.OperationalError:
            continue

    # Shuffle to mix different keywords
    random.shuffle(samples)
    return samples[:limit]


def sample_audio_torrents_fts(mag_conn: sqlite3.Connection, limit: int) -> list[tuple]:
    """Sample audio torrents using FTS for fast filtering."""
    audio_keywords = [
        "flac", "320kbps", "mp3", "lossless", "24bit", "album"
    ]

    samples = []
    seen = set()

    for keyword in audio_keywords:
        if len(samples) >= limit:
            break

        offset = random.randint(0, 10000)

        try:
            cursor = mag_conn.execute("""
                SELECT t.name, t.total_size
                FROM torrents t
                JOIN torrents_idx ON torrents_idx.rowid = t.id
                WHERE torrents_idx MATCH ?
                LIMIT 5000 OFFSET ?
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

                # Skip video content
                if VIDEO_INDICATORS.search(name) and not AUDIO_INDICATORS.search(name):
                    continue

                if AUDIO_INDICATORS.search(name):
                    samples.append((name, size))

        except sqlite3.OperationalError:
            continue

    random.shuffle(samples)
    return samples[:limit]


def sample_video_torrents(mag_conn: sqlite3.Connection, limit: int) -> list[tuple]:
    """Sample random video-looking torrents from magnetico."""
    # Try FTS first (much faster)
    samples = sample_video_torrents_fts(mag_conn, limit)
    if len(samples) >= limit:
        return samples

    # Fall back to random offset if FTS didn't return enough
    total = mag_conn.execute("SELECT COUNT(*) FROM torrents").fetchone()[0]
    attempts = 0
    max_attempts = limit * 10

    while len(samples) < limit and attempts < max_attempts:
        offset = random.randint(0, max(0, total - 1000))

        cursor = mag_conn.execute("""
            SELECT name, total_size FROM torrents
            LIMIT 100 OFFSET ?
        """, (offset,))

        for name_bytes, size in cursor:
            try:
                name = name_bytes.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                continue

            if not (VIDEO_SIZE_RANGE[0] <= size <= VIDEO_SIZE_RANGE[1]):
                continue

            if VIDEO_INDICATORS.search(name):
                samples.append((name, size))
                if len(samples) >= limit:
                    break

        attempts += 1

    return samples


def sample_audio_torrents(mag_conn: sqlite3.Connection, limit: int) -> list[tuple]:
    """Sample random audio-looking torrents from magnetico."""
    # Try FTS first
    samples = sample_audio_torrents_fts(mag_conn, limit)
    if len(samples) >= limit:
        return samples

    # Fall back to random offset
    total = mag_conn.execute("SELECT COUNT(*) FROM torrents").fetchone()[0]
    attempts = 0
    max_attempts = limit * 10

    while len(samples) < limit and attempts < max_attempts:
        offset = random.randint(0, max(0, total - 1000))

        cursor = mag_conn.execute("""
            SELECT name, total_size FROM torrents
            LIMIT 100 OFFSET ?
        """, (offset,))

        for name_bytes, size in cursor:
            try:
                name = name_bytes.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                continue

            if not (AUDIO_SIZE_RANGE[0] <= size <= AUDIO_SIZE_RANGE[1]):
                continue

            if AUDIO_INDICATORS.search(name):
                samples.append((name, size))
                if len(samples) >= limit:
                    break

        attempts += 1

    return samples


def process_video_samples(samples: list[tuple], gt_conn: sqlite3.Connection, tmdb_session: requests.Session) -> int:
    """Process video samples and store validated matches."""
    validated = 0

    for i, (name, size) in enumerate(samples):
        meta = extract_video_metadata(name)
        if not meta:
            continue

        # Try to validate
        if meta["type"] == "movie":
            result = validate_movie_tmdb(meta["title"], meta["year"], tmdb_session)
        else:
            result = validate_tv_tmdb(meta["title"], meta["year"], tmdb_session)

        if result:
            # Insert ground truth
            try:
                gt_conn.execute("""
                    INSERT OR IGNORE INTO ground_truth
                    (type, title, year, external_id, source, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result["type"],
                    result["title"],
                    result.get("year"),
                    result.get("tmdb_id"),
                    "tmdb",
                    json.dumps({"reverse_matched": True})
                ))

                gt_id = gt_conn.execute("SELECT id FROM ground_truth WHERE source='tmdb' AND external_id=?",
                                        (result.get("tmdb_id"),)).fetchone()

                if gt_id:
                    gt_conn.execute("""
                        INSERT OR IGNORE INTO matches
                        (ground_truth_id, torrent_name, torrent_size, match_score)
                        VALUES (?, ?, ?, ?)
                    """, (gt_id[0], name, size, 15.0))  # High score for validated matches

                    validated += 1
            except sqlite3.IntegrityError:
                pass

        if (i + 1) % 100 == 0:
            gt_conn.commit()
            print(f"  Video: {i+1}/{len(samples)} processed, {validated} validated")
            time.sleep(0.25)  # TMDB rate limit

    gt_conn.commit()
    return validated


def process_audio_samples(samples: list[tuple], gt_conn: sqlite3.Connection, mb_session: requests.Session) -> int:
    """Process audio samples and store validated matches."""
    validated = 0

    for i, (name, size) in enumerate(samples):
        meta = extract_audio_metadata(name)
        if not meta:
            continue

        result = validate_album_musicbrainz(meta["title"], meta["artist"], meta.get("year"), mb_session)

        if result:
            try:
                gt_conn.execute("""
                    INSERT OR IGNORE INTO ground_truth
                    (type, title, artist, year, external_id, source, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    "album",
                    result["title"],
                    result["artist"],
                    result.get("year"),
                    result.get("mb_id"),
                    "musicbrainz",
                    json.dumps({"reverse_matched": True})
                ))

                gt_id = gt_conn.execute("SELECT id FROM ground_truth WHERE source='musicbrainz' AND external_id=?",
                                        (result.get("mb_id"),)).fetchone()

                if gt_id:
                    gt_conn.execute("""
                        INSERT OR IGNORE INTO matches
                        (ground_truth_id, torrent_name, torrent_size, match_score)
                        VALUES (?, ?, ?, ?)
                    """, (gt_id[0], name, size, 15.0))

                    validated += 1
            except sqlite3.IntegrityError:
                pass

        if (i + 1) % 50 == 0:
            gt_conn.commit()
            print(f"  Audio: {i+1}/{len(samples)} processed, {validated} validated")
            time.sleep(1.1)  # MusicBrainz rate limit

    gt_conn.commit()
    return validated


def main():
    parser = argparse.ArgumentParser(description="Reverse-match magnetico torrents against metadata APIs")
    parser.add_argument("--video", type=int, default=5000, help="Number of video samples to process")
    parser.add_argument("--audio", type=int, default=5000, help="Number of audio samples to process")
    args = parser.parse_args()

    print(f"Magnetico DB: {MAGNETICO_DB}")
    print(f"Ground truth DB: {GROUND_TRUTH_DB}")
    print("-" * 60)

    # Connect to databases
    mag_conn = sqlite3.connect(MAGNETICO_DB)
    mag_conn.text_factory = bytes  # Get raw bytes, decode manually
    gt_conn = sqlite3.connect(GROUND_TRUTH_DB)

    # Create sessions
    tmdb_session = requests.Session()
    mb_session = requests.Session()
    mb_session.headers.update({
        "User-Agent": MB_USER_AGENT,
        "Accept": "application/json"
    })

    # Get initial counts
    initial_gt = gt_conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    initial_matches = gt_conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"Initial: {initial_gt} ground truth entries, {initial_matches} matches")

    # Sample and process video
    print(f"\nSampling {args.video} video torrents...")
    video_samples = sample_video_torrents(mag_conn, args.video)
    print(f"  Got {len(video_samples)} video samples")

    print("Processing video samples...")
    video_validated = process_video_samples(video_samples, gt_conn, tmdb_session)
    print(f"  Validated {video_validated} video torrents")

    # Sample and process audio
    print(f"\nSampling {args.audio} audio torrents...")
    audio_samples = sample_audio_torrents(mag_conn, args.audio)
    print(f"  Got {len(audio_samples)} audio samples")

    print("Processing audio samples (slower due to MusicBrainz rate limits)...")
    audio_validated = process_audio_samples(audio_samples, gt_conn, mb_session)
    print(f"  Validated {audio_validated} audio torrents")

    # Final summary
    final_gt = gt_conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    final_matches = gt_conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    print("-" * 60)
    print("Summary:")
    print(f"  New ground truth entries: {final_gt - initial_gt}")
    print(f"  New matches: {final_matches - initial_matches}")
    print(f"  Total ground truth: {final_gt}")
    print(f"  Total matches: {final_matches}")

    # Show by type
    cursor = gt_conn.execute("""
        SELECT gt.type, COUNT(DISTINCT gt.id), COUNT(m.id)
        FROM ground_truth gt
        LEFT JOIN matches m ON gt.id = m.ground_truth_id
        GROUP BY gt.type
    """)
    print("\nBy type:")
    for row in cursor:
        print(f"  {row[0]}: {row[1]} entries, {row[2]} matches")

    mag_conn.close()
    gt_conn.close()


if __name__ == "__main__":
    main()
