#!/usr/bin/env python3
"""
Validate audio samples against Spotify catalog.
Runs on homelab where spotify_vacuumed.sqlite3 is located.

Usage:
    cat audio_samples.jsonl | python validate_spotify.py > validated.jsonl
"""

import json
import sqlite3
import sys
from difflib import SequenceMatcher
from pathlib import Path

SPOTIFY_DB = Path.home() / "Downloads" / "spotify_vacuumed.sqlite3"


def normalize(s: str) -> str:
    """Normalize string for comparison."""
    s = s.lower()
    # Remove common suffixes
    s = s.replace("(deluxe)", "").replace("(remastered)", "").replace("(expanded)", "")
    s = s.replace("[deluxe]", "").replace("[remastered]", "").replace("[expanded]", "")
    # Remove punctuation
    import re
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def validate_album(conn: sqlite3.Connection, artist: str, album: str, year: int | None) -> dict | None:
    """Search Spotify catalog for matching album."""
    norm_artist = normalize(artist)
    norm_album = normalize(album)

    best_match = None
    best_score = 0

    # Strategy 1: Exact album name match (uses index)
    cursor = conn.execute("""
        SELECT a.name, ar.name, a.release_date, a.rowid
        FROM albums a
        JOIN artist_albums aa ON a.rowid = aa.album_rowid
        JOIN artists ar ON ar.rowid = aa.artist_rowid
        WHERE aa.is_appears_on = 0
        AND a.name = ?
        LIMIT 50
    """, (album,))

    for album_name, artist_name, release_date, album_rowid in cursor:
        album_sim = SequenceMatcher(None, norm_album, normalize(album_name)).ratio()
        artist_sim = SequenceMatcher(None, norm_artist, normalize(artist_name)).ratio()
        score = album_sim * 0.6 + artist_sim * 0.4

        result_year = None
        if release_date and len(release_date) >= 4:
            try:
                result_year = int(release_date[:4])
            except ValueError:
                pass

        if year and result_year and year == result_year:
            score += 0.1

        if score > best_score and artist_sim > 0.5:
            best_score = score
            best_match = {
                "album": album_name,
                "artist": artist_name,
                "year": result_year,
                "spotify_rowid": album_rowid,
                "score": score,
            }

    if best_score >= 0.85:
        return best_match

    # Strategy 2: Prefix match on album (uses index)
    album_prefix = album[:15] if len(album) > 15 else album
    cursor = conn.execute("""
        SELECT a.name, ar.name, a.release_date, a.rowid
        FROM albums a
        JOIN artist_albums aa ON a.rowid = aa.album_rowid
        JOIN artists ar ON ar.rowid = aa.artist_rowid
        WHERE aa.is_appears_on = 0
        AND a.name >= ? AND a.name < ?
        LIMIT 100
    """, (album_prefix, album_prefix + 'zzz'))

    for album_name, artist_name, release_date, album_rowid in cursor:
        album_sim = SequenceMatcher(None, norm_album, normalize(album_name)).ratio()
        artist_sim = SequenceMatcher(None, norm_artist, normalize(artist_name)).ratio()
        score = album_sim * 0.6 + artist_sim * 0.4

        result_year = None
        if release_date and len(release_date) >= 4:
            try:
                result_year = int(release_date[:4])
            except ValueError:
                pass

        if year and result_year and year == result_year:
            score += 0.1

        if score > best_score and album_sim > 0.6 and artist_sim > 0.5:
            best_score = score
            best_match = {
                "album": album_name,
                "artist": artist_name,
                "year": result_year,
                "spotify_rowid": album_rowid,
                "score": score,
            }

    if best_score >= 0.85:
        return best_match

    # Strategy 3: Exact artist name match (uses index)
    cursor = conn.execute("""
        SELECT a.name, ar.name, a.release_date, a.rowid
        FROM artists ar
        JOIN artist_albums aa ON ar.rowid = aa.artist_rowid
        JOIN albums a ON a.rowid = aa.album_rowid
        WHERE aa.is_appears_on = 0
        AND ar.name = ?
        LIMIT 100
    """, (artist,))

    for album_name, artist_name, release_date, album_rowid in cursor:
        album_sim = SequenceMatcher(None, norm_album, normalize(album_name)).ratio()
        artist_sim = SequenceMatcher(None, norm_artist, normalize(artist_name)).ratio()
        score = album_sim * 0.6 + artist_sim * 0.4

        result_year = None
        if release_date and len(release_date) >= 4:
            try:
                result_year = int(release_date[:4])
            except ValueError:
                pass

        if year and result_year and year == result_year:
            score += 0.1

        if score > best_score and album_sim > 0.6 and artist_sim > 0.5:
            best_score = score
            best_match = {
                "album": album_name,
                "artist": artist_name,
                "year": result_year,
                "spotify_rowid": album_rowid,
                "score": score,
            }

    return best_match if best_score > 0.7 else None


def main():
    if not SPOTIFY_DB.exists():
        print(f"Error: Spotify DB not found at {SPOTIFY_DB}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(SPOTIFY_DB)
    conn.execute("PRAGMA cache_size = -100000")  # 100MB cache

    validated = 0
    processed = 0

    for line in sys.stdin:
        try:
            sample = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        processed += 1

        result = validate_album(
            conn,
            sample.get("artist", ""),
            sample.get("album", ""),
            sample.get("year"),
        )

        if result:
            validated += 1
            output = {
                "torrent_name": sample["torrent_name"],
                "torrent_size": sample["torrent_size"],
                "spotify_album": result["album"],
                "spotify_artist": result["artist"],
                "spotify_year": result["year"],
                "match_score": result["score"],
            }
            print(json.dumps(output, ensure_ascii=False))

        if processed % 1000 == 0:
            print(f"Processed {processed}, validated {validated} ({validated/processed*100:.1f}%)", file=sys.stderr)

    print(f"\nDone: {processed} processed, {validated} validated ({validated/processed*100:.1f}%)", file=sys.stderr)
    conn.close()


if __name__ == "__main__":
    main()
