#!/usr/bin/env python3
"""
Fetch popular albums from MusicBrainz for ground-truth training data.

MusicBrainz doesn't have a "top albums" list, so we:
1. Fetch releases that have been tagged/rated most often
2. Use the "release-group" endpoint with ratings

Usage:
    python scripts/fetch_musicbrainz_metadata.py --albums 5000
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import requests

# MusicBrainz requires a user-agent with contact info
USER_AGENT = "MimmoTrainingDataFetcher/1.0 (https://github.com/user/mimmo)"
MB_BASE_URL = "https://musicbrainz.org/ws/2"

DB_PATH = Path(__file__).parent.parent / "data" / "training_ground_truth.db"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ground_truth (
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            original_title TEXT,
            artist TEXT,
            year INTEGER,
            external_id TEXT,
            source TEXT NOT NULL,
            extra_json TEXT,
            UNIQUE(source, external_id)
        )
    """)
    conn.commit()
    return conn


def fetch_mb_session() -> requests.Session:
    """Create a session with proper headers for MusicBrainz."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    })
    return session


def fetch_popular_artists(session: requests.Session, limit: int = 100) -> list[dict]:
    """Fetch artists with high ratings."""
    artists = []
    offset = 0

    print(f"Fetching top-rated artists...")

    while len(artists) < limit:
        url = f"{MB_BASE_URL}/artist"
        params = {
            "query": "rating:[4 TO 5]",
            "fmt": "json",
            "limit": 100,
            "offset": offset
        }

        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 503:
            print("  Rate limited, waiting 2s...")
            time.sleep(2)
            continue

        if resp.status_code != 200:
            print(f"  Error: {resp.status_code}")
            break

        data = resp.json()
        batch = data.get("artists", [])
        if not batch:
            break

        artists.extend(batch)
        offset += 100
        time.sleep(1.1)  # MusicBrainz rate limit: 1 req/sec

    print(f"  Found {len(artists)} artists")
    return artists[:limit]


def fetch_albums_by_artist(session: requests.Session, artist_id: str, artist_name: str) -> list[dict]:
    """Fetch release groups (albums) for an artist."""
    url = f"{MB_BASE_URL}/release-group"
    params = {
        "artist": artist_id,
        "type": "album",
        "fmt": "json",
        "limit": 100
    }

    resp = session.get(url, params=params, timeout=30)
    if resp.status_code == 503:
        time.sleep(2)
        resp = session.get(url, params=params, timeout=30)

    if resp.status_code != 200:
        return []

    data = resp.json()
    albums = []

    for rg in data.get("release-groups", []):
        albums.append({
            "id": rg.get("id"),
            "title": rg.get("title"),
            "artist": artist_name,
            "year": rg.get("first-release-date", "")[:4] if rg.get("first-release-date") else None,
            "type": rg.get("primary-type", "Album"),
        })

    return albums


def fetch_popular_releases(session: requests.Session, count: int, conn: sqlite3.Connection) -> int:
    """Fetch popular releases using a search query."""
    print(f"Fetching {count} popular albums from MusicBrainz...")

    # Search for highly-rated release groups
    # MusicBrainz doesn't have a "most popular" list, so we search for common keywords
    # and sort by score (relevance)

    inserted = 0
    offset = 0

    while inserted < count:
        url = f"{MB_BASE_URL}/release-group"
        params = {
            "query": "primarytype:album AND status:official",
            "fmt": "json",
            "limit": 100,
            "offset": offset
        }

        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 503:
            print("  Rate limited, waiting 2s...")
            time.sleep(2)
            continue

        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} - {resp.text[:200]}")
            break

        data = resp.json()
        release_groups = data.get("release-groups", [])

        if not release_groups:
            print(f"  No more results at offset {offset}")
            break

        for rg in release_groups:
            if inserted >= count:
                break

            title = rg.get("title", "")
            mb_id = rg.get("id", "")

            # Get artist from artist-credit
            artist_credit = rg.get("artist-credit", [])
            if artist_credit:
                artist = artist_credit[0].get("name", "") if artist_credit else ""
                # Handle "Various Artists" case
                if len(artist_credit) > 1:
                    artist = " / ".join(ac.get("name", "") for ac in artist_credit[:3])
            else:
                artist = ""

            # Get year from first-release-date
            first_release = rg.get("first-release-date", "")
            year = int(first_release[:4]) if first_release and len(first_release) >= 4 else None

            extra = {
                "primary_type": rg.get("primary-type"),
                "secondary_types": rg.get("secondary-types", []),
            }

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO ground_truth
                    (type, title, artist, year, external_id, source, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("album", title, artist, year, mb_id, "musicbrainz", json.dumps(extra)))

                if conn.total_changes > 0:
                    inserted += 1
                    if inserted % 100 == 0:
                        print(f"  Inserted {inserted}/{count} albums...")
                        conn.commit()
            except sqlite3.IntegrityError:
                pass

        offset += 100
        time.sleep(1.1)  # MusicBrainz strict rate limit

    conn.commit()
    print(f"  Done: {inserted} albums inserted")
    return inserted


def fetch_releases_by_decade(session: requests.Session, count_per_decade: int, conn: sqlite3.Connection) -> int:
    """Fetch releases across decades for variety."""
    print(f"Fetching albums by decade for variety...")

    decades = [
        ("1960", "1969"),
        ("1970", "1979"),
        ("1980", "1989"),
        ("1990", "1999"),
        ("2000", "2009"),
        ("2010", "2019"),
        ("2020", "2025"),
    ]

    total_inserted = 0

    for start_year, end_year in decades:
        print(f"  Decade {start_year}s...")
        inserted = 0
        offset = 0

        while inserted < count_per_decade:
            url = f"{MB_BASE_URL}/release-group"
            params = {
                "query": f"primarytype:album AND firstreleasedate:[{start_year} TO {end_year}]",
                "fmt": "json",
                "limit": 100,
                "offset": offset
            }

            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 503:
                time.sleep(2)
                continue

            if resp.status_code != 200:
                break

            data = resp.json()
            release_groups = data.get("release-groups", [])

            if not release_groups:
                break

            for rg in release_groups:
                if inserted >= count_per_decade:
                    break

                title = rg.get("title", "")
                mb_id = rg.get("id", "")

                artist_credit = rg.get("artist-credit", [])
                if artist_credit:
                    artist = artist_credit[0].get("name", "")
                else:
                    artist = ""

                first_release = rg.get("first-release-date", "")
                year = int(first_release[:4]) if first_release and len(first_release) >= 4 else None

                extra = {
                    "primary_type": rg.get("primary-type"),
                }

                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO ground_truth
                        (type, title, artist, year, external_id, source, extra_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, ("album", title, artist, year, mb_id, "musicbrainz", json.dumps(extra)))

                    if conn.total_changes > 0:
                        inserted += 1
                        total_inserted += 1
                except sqlite3.IntegrityError:
                    pass

            offset += 100
            time.sleep(1.1)

        print(f"    Got {inserted} from {start_year}s")
        conn.commit()

    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Fetch MusicBrainz metadata for training")
    parser.add_argument("--albums", type=int, default=5000, help="Number of albums to fetch")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = init_db(db_path)
    session = fetch_mb_session()

    print(f"Database: {db_path}")
    print("-" * 60)

    # Approach: fetch from multiple decades for variety
    # This gives us better coverage than just "most popular"
    count_per_decade = args.albums // 7

    total = fetch_releases_by_decade(session, count_per_decade, conn)

    # If we need more, fetch general popular releases
    remaining = args.albums - total
    if remaining > 0:
        total += fetch_popular_releases(session, remaining, conn)

    print("-" * 60)

    # Show summary
    cursor = conn.execute("SELECT type, COUNT(*) FROM ground_truth GROUP BY type")
    print("Summary:")
    for row in cursor:
        print(f"  {row[0]}: {row[1]}")

    total_count = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    print(f"  Total: {total_count}")

    # Show some album examples
    print("\nSample albums:")
    cursor = conn.execute("SELECT artist, title, year FROM ground_truth WHERE type='album' LIMIT 10")
    for row in cursor:
        print(f"  {row[0]} - {row[1]} ({row[2]})")

    conn.close()


if __name__ == "__main__":
    main()
