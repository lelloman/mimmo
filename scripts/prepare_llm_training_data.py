#!/usr/bin/env python3
"""
Prepare training data for small LLM metadata extraction.

Output format per schema:
- [video/movie] → title | year
- [video/episode] → series_title | episode_title
- [video/season] → series_title
- [video/series] → series_title
- [audio/album] → album_name | artist | year
- [audio/track] → track_name | artist | year

Each schema has AT MOST one optional field (always last when present).
"""

import json
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "training_ground_truth.db"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "llm_training_data.jsonl"


def detect_video_subtype(torrent_name: str) -> str:
    """Detect video subtype from torrent name patterns."""
    name_lower = torrent_name.lower()

    # Check for episode pattern: S01E01, 1x01, etc.
    if re.search(r's\d{1,2}e\d{1,2}', name_lower) or re.search(r'\d{1,2}x\d{1,2}', name_lower):
        return "video/episode"

    # Check for multi-season (full series): S01-S07, Season 1-5, etc.
    if re.search(r's\d{1,2}\s*-\s*s?\d{1,2}', name_lower) or re.search(r'season\s*\d+\s*-\s*\d+', name_lower):
        return "video/series"

    # Check for season pack: S01.COMPLETE, Season 1, S01 (without episode)
    if re.search(r's\d{1,2}\.?complete', name_lower) or \
       re.search(r'season\s*\d+', name_lower) or \
       re.search(r'\.s\d{1,2}\.', name_lower):
        return "video/season"

    # Default for TV without clear pattern - assume series
    return "video/series"


def format_output(type_: str, subtype: str, title: str, artist: str | None, year: int | None) -> str:
    """Format the output string based on type/subtype."""

    if subtype == "video/movie":
        # title | year (year optional)
        if year:
            return f"{title} | {year}"
        return title

    elif subtype == "video/episode":
        # series_title | episode_title (episode_title optional)
        # We don't have episode_title in ground truth, so just return series_title
        return title

    elif subtype == "video/season":
        # series_title (no optional fields)
        return title

    elif subtype == "video/series":
        # series_title (no optional fields)
        return title

    elif subtype == "audio/album":
        # album_name | artist | year (year optional)
        if not artist:
            return None  # Skip - artist is required
        if year:
            return f"{title} | {artist} | {year}"
        return f"{title} | {artist}"

    elif subtype == "audio/track":
        # track_name | artist | year (year optional)
        if not artist:
            return None  # Skip - artist is required
        if year:
            return f"{title} | {artist} | {year}"
        return f"{title} | {artist}"

    return None


def main():
    conn = sqlite3.connect(DB_PATH)

    # Get all matches with ground truth
    cursor = conn.execute("""
        SELECT m.torrent_name, g.type, g.title, g.artist, g.year
        FROM matches m
        JOIN ground_truth g ON m.ground_truth_id = g.id
    """)

    samples = []
    stats = {
        "video/movie": 0,
        "video/episode": 0,
        "video/season": 0,
        "video/series": 0,
        "audio/album": 0,
        "audio/track": 0,
        "skipped": 0,
    }

    for torrent_name, type_, title, artist, year in cursor:
        # Determine subtype
        if type_ == "movie":
            subtype = "video/movie"
        elif type_ == "tv":
            subtype = detect_video_subtype(torrent_name)
        elif type_ == "album":
            subtype = "audio/album"
        else:
            stats["skipped"] += 1
            continue

        # Format output
        output = format_output(type_, subtype, title, artist, year)
        if output is None:
            stats["skipped"] += 1
            continue

        # Format input with task command and type prefix
        input_text = f"<|extract|>[{subtype}] {torrent_name}"

        samples.append({
            "input": input_text,
            "output": output,
            "subtype": subtype,
        })
        stats[subtype] += 1

    conn.close()

    # Write to JSONL
    with open(OUTPUT_PATH, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} training samples to {OUTPUT_PATH}")
    print("\nDistribution:")
    for subtype, count in sorted(stats.items()):
        print(f"  {subtype}: {count}")

    # Show some examples
    print("\nExamples:")
    import random
    for sample in random.sample(samples, min(10, len(samples))):
        print(f"\n  Input:  {sample['input'][:80]}...")
        print(f"  Output: {sample['output']}")


if __name__ == "__main__":
    main()
