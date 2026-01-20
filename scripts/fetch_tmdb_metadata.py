#!/usr/bin/env python3
"""
Fetch top movies and TV series from TMDB for ground-truth training data.

Usage:
    python scripts/fetch_tmdb_metadata.py --movies 2000 --tv 500
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import requests

TMDB_API_KEY = "543ba71a128563c1afb45daaa84df1fb"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            ground_truth_id INTEGER REFERENCES ground_truth(id),
            torrent_name TEXT NOT NULL,
            torrent_size INTEGER,
            match_score REAL,
            UNIQUE(ground_truth_id, torrent_name)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY,
            input TEXT NOT NULL,
            label_json TEXT NOT NULL,
            source TEXT NOT NULL,
            match_id INTEGER REFERENCES matches(id)
        )
    """)
    conn.commit()
    return conn


def fetch_tmdb_movies(count: int, conn: sqlite3.Connection) -> int:
    """Fetch top-rated movies from TMDB."""
    print(f"Fetching top {count} movies from TMDB...")

    inserted = 0
    page = 1
    max_pages = (count // 20) + 1  # TMDB returns 20 per page

    while inserted < count and page <= max_pages:
        url = f"{TMDB_BASE_URL}/movie/top_rated"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "page": page
        }

        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} - {resp.text}")
            break

        data = resp.json()
        movies = data.get("results", [])

        if not movies:
            break

        for movie in movies:
            if inserted >= count:
                break

            title = movie.get("title", "")
            original_title = movie.get("original_title", "")
            release_date = movie.get("release_date", "")
            year = int(release_date[:4]) if release_date and len(release_date) >= 4 else None
            tmdb_id = str(movie.get("id", ""))

            extra = {
                "overview": movie.get("overview", ""),
                "popularity": movie.get("popularity", 0),
                "vote_average": movie.get("vote_average", 0),
                "genre_ids": movie.get("genre_ids", []),
            }

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO ground_truth
                    (type, title, original_title, year, external_id, source, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("movie", title, original_title, year, tmdb_id, "tmdb", json.dumps(extra)))

                if conn.total_changes > 0:
                    inserted += 1
                    if inserted % 100 == 0:
                        print(f"  Inserted {inserted}/{count} movies...")
                        conn.commit()
            except sqlite3.IntegrityError:
                pass  # Already exists

        page += 1
        time.sleep(0.25)  # Rate limit: 4 requests/sec

    conn.commit()
    print(f"  Done: {inserted} movies inserted")
    return inserted


def fetch_tmdb_tv(count: int, conn: sqlite3.Connection) -> int:
    """Fetch top-rated TV series from TMDB."""
    print(f"Fetching top {count} TV series from TMDB...")

    inserted = 0
    page = 1
    max_pages = (count // 20) + 1

    while inserted < count and page <= max_pages:
        url = f"{TMDB_BASE_URL}/tv/top_rated"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "page": page
        }

        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} - {resp.text}")
            break

        data = resp.json()
        shows = data.get("results", [])

        if not shows:
            break

        for show in shows:
            if inserted >= count:
                break

            title = show.get("name", "")
            original_title = show.get("original_name", "")
            first_air_date = show.get("first_air_date", "")
            year = int(first_air_date[:4]) if first_air_date and len(first_air_date) >= 4 else None
            tmdb_id = str(show.get("id", ""))

            extra = {
                "overview": show.get("overview", ""),
                "popularity": show.get("popularity", 0),
                "vote_average": show.get("vote_average", 0),
                "genre_ids": show.get("genre_ids", []),
            }

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO ground_truth
                    (type, title, original_title, year, external_id, source, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("tv", title, original_title, year, tmdb_id, "tmdb", json.dumps(extra)))

                if conn.total_changes > 0:
                    inserted += 1
                    if inserted % 100 == 0:
                        print(f"  Inserted {inserted}/{count} TV series...")
                        conn.commit()
            except sqlite3.IntegrityError:
                pass

        page += 1
        time.sleep(0.25)

    conn.commit()
    print(f"  Done: {inserted} TV series inserted")
    return inserted


def fetch_popular_movies(count: int, conn: sqlite3.Connection) -> int:
    """Fetch popular movies (different list than top_rated)."""
    print(f"Fetching {count} popular movies from TMDB...")

    inserted = 0
    page = 1
    max_pages = (count // 20) + 1

    while inserted < count and page <= max_pages:
        url = f"{TMDB_BASE_URL}/movie/popular"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "page": page
        }

        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code} - {resp.text}")
            break

        data = resp.json()
        movies = data.get("results", [])

        if not movies:
            break

        for movie in movies:
            if inserted >= count:
                break

            title = movie.get("title", "")
            original_title = movie.get("original_title", "")
            release_date = movie.get("release_date", "")
            year = int(release_date[:4]) if release_date and len(release_date) >= 4 else None
            tmdb_id = str(movie.get("id", ""))

            extra = {
                "overview": movie.get("overview", ""),
                "popularity": movie.get("popularity", 0),
                "vote_average": movie.get("vote_average", 0),
                "genre_ids": movie.get("genre_ids", []),
            }

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO ground_truth
                    (type, title, original_title, year, external_id, source, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("movie", title, original_title, year, tmdb_id, "tmdb", json.dumps(extra)))

                if conn.total_changes > 0:
                    inserted += 1
                    if inserted % 100 == 0:
                        print(f"  Inserted {inserted}/{count} popular movies...")
                        conn.commit()
            except sqlite3.IntegrityError:
                pass

        page += 1
        time.sleep(0.25)

    conn.commit()
    print(f"  Done: {inserted} popular movies inserted")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Fetch TMDB metadata for training")
    parser.add_argument("--movies", type=int, default=2000, help="Number of movies to fetch")
    parser.add_argument("--tv", type=int, default=500, help="Number of TV series to fetch")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = init_db(db_path)

    print(f"Database: {db_path}")
    print("-" * 60)

    # Fetch from multiple lists to get more variety
    total_movies = 0
    total_movies += fetch_tmdb_movies(args.movies // 2, conn)  # Half from top_rated
    total_movies += fetch_popular_movies(args.movies // 2, conn)  # Half from popular

    total_tv = fetch_tmdb_tv(args.tv, conn)

    print("-" * 60)

    # Show summary
    cursor = conn.execute("SELECT type, COUNT(*) FROM ground_truth GROUP BY type")
    print("Summary:")
    for row in cursor:
        print(f"  {row[0]}: {row[1]}")

    total = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    print(f"  Total: {total}")

    conn.close()


if __name__ == "__main__":
    main()
