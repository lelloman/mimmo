#!/usr/bin/env python3
"""
Album metadata extraction heuristics with iterative refinement.

Usage:
    python scripts/album_heuristics.py run      # Run extraction on todo, move successes to done
    python scripts/album_heuristics.py regress  # Verify all done items still pass
    python scripts/album_heuristics.py stats    # Show current stats
"""

import re
import sqlite3
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher

DB_PATH = "/home/lelloman/lelloprojects/mimmo/data/training_ground_truth.db"

# Current heuristic version - increment when updating rules
HEURISTIC_VERSION = 3


@dataclass
class Extraction:
    artist: str | None = None
    title: str | None = None
    year: int | None = None


def similarity(a: str | None, b: str | None) -> float:
    """Calculate similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    a = a.lower().strip()
    b = b.lower().strip()
    return SequenceMatcher(None, a, b).ratio()


def get_artist_aliases(conn: sqlite3.Connection, artist_name: str) -> list[str]:
    """Get all known aliases for an artist."""
    cursor = conn.execute(
        "SELECT alias FROM artist_aliases WHERE LOWER(canonical_name) = LOWER(?)",
        (artist_name,)
    )
    aliases = [row[0] for row in cursor]
    if not aliases:
        # Try matching by alias
        cursor = conn.execute(
            "SELECT canonical_name, alias FROM artist_aliases WHERE LOWER(alias) = LOWER(?)",
            (artist_name,)
        )
        for row in cursor:
            # Get all aliases for this canonical name
            cursor2 = conn.execute(
                "SELECT alias FROM artist_aliases WHERE canonical_name = ?",
                (row[0],)
            )
            aliases = [r[0] for r in cursor2]
            break
    return aliases if aliases else [artist_name]


def best_artist_similarity(extracted: str | None, expected: str, aliases: list[str]) -> float:
    """Get best similarity score across all artist aliases."""
    if not extracted:
        return 0.0
    return max(similarity(extracted, alias) for alias in aliases)


def clean_torrent_name(name: str) -> str:
    """Basic cleaning of torrent name."""
    # Remove common torrent site tags
    name = re.sub(r'\[rarbg\]|\[ettv\]|\[eztv\]|\[YTS.*?\]|\[TGx\]', '', name, flags=re.IGNORECASE)
    # Remove file extensions
    name = re.sub(r'\.(mp3|flac|wav|ogg|m4a|aac|zip|rar|tar|7z)$', '', name, flags=re.IGNORECASE)
    return name.strip()


def extract_year(text: str) -> tuple[int | None, str]:
    """Extract year from text, return (year, remaining_text)."""
    # Try parentheses first: (2023)
    match = re.search(r'\((\d{4})\)', text)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2030:
            remaining = text[:match.start()] + text[match.end():]
            return year, remaining.strip()

    # Try brackets: [2023]
    match = re.search(r'\[(\d{4})\]', text)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2030:
            remaining = text[:match.start()] + text[match.end():]
            return year, remaining.strip()

    # Try standalone year with word boundary
    match = re.search(r'(?:^|[\s\-_.])(\d{4})(?:[\s\-_.\[]|$)', text)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2030:
            remaining = text[:match.start()] + text[match.end():]
            return year, remaining.strip()

    return None, text


def extract_album_v1(name: str) -> Extraction:
    """
    Version 1: Basic pattern matching.

    Handles patterns like:
    - Artist - Album (Year) [Format]
    - Artist - Album Year [Format]
    """
    result = Extraction()

    # Clean the name
    name = clean_torrent_name(name)

    # Extract year first
    result.year, name = extract_year(name)

    # Remove format tags: [FLAC], [MP3 320], [24-96], etc.
    name = re.sub(r'\[.*?\]', '', name)
    name = re.sub(r'\((?:FLAC|MP3|AAC|OGG|WAV|320|256|192|V0|24[\-_]?(?:44|48|96|192)|16[\-_]?44).*?\)', '', name, flags=re.IGNORECASE)

    # Remove common quality/source tags
    name = re.sub(r'(?:FLAC|MP3|AAC|WEB|CD|VINYL|LP|Vinyl|vinyl|CD-?RIP|WEB-?DL|WEB-?FLAC)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(?:320|256|192|V0)\s*(?:kbps|cbr|vbr)?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(?:24|16)[\-_]?(?:44|48|96|192)\b', '', name)

    # Remove release group tags at end
    name = re.sub(r'[\-_]\s*[A-Z0-9]{2,10}$', '', name)

    # Remove uploader names in common formats
    name = re.sub(r'\s*[\-_]?\s*(?:PMEDIA|MiRCrew|GloDLS|HANNIBAL|SilverRG|NimitMak|Michi80|Big Papi)\s*', '', name, flags=re.IGNORECASE)

    # Clean up extra whitespace and punctuation
    name = re.sub(r'\s*[\-_]+\s*$', '', name)
    name = re.sub(r'^\s*[\-_]+\s*', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    # Try to split into Artist - Album
    # Pattern 1: "Artist - Album" with explicit dash separator
    match = re.match(r'^(.+?)\s*[-–—]\s*(.+)$', name)
    if match:
        result.artist = match.group(1).strip()
        result.title = match.group(2).strip()

        # Clean up common artifacts from title
        if result.title:
            result.title = re.sub(r'\s*[-–—]\s*$', '', result.title)
            result.title = re.sub(r'^\s*[-–—]\s*', '', result.title)

    return result


def extract_album_v2(name: str) -> Extraction:
    """
    Version 2: Improved year extraction.

    Improvements over v1:
    - Better year extraction (dates like [2024.11.22], years with genre like (2024 Pop))
    - Remove emojis
    - Remove curly brace content: {Label, Catalog}

    NOTE: Keeping v1's approach for everything else since it handles scene formats better.
    """
    result = Extraction()

    # Clean the name
    name = clean_torrent_name(name)

    # Remove emojis early (before any other processing)
    name = re.sub(r'[⭐️🎵🎶🔥💿📀😎]+', '', name)

    # NOTE: Not removing curly brace content because it sometimes contains
    # edition info that's part of the expected title (e.g., "{Deluxe Edition}")

    # Extract year - improved patterns (BEFORE removing brackets)
    # Try date format in brackets: [2024.11.22] or [2024-11-22]
    match = re.search(r'\[(\d{4})[.\-]\d{2}[.\-]\d{2}\]', name)
    if match:
        result.year = int(match.group(1))
        name = name[:match.start()] + name[match.end():]

    # Try year with catalog number: [2024 - 602475686767]
    if result.year is None:
        match = re.search(r'\[(\d{4})\s*[-–]\s*\d+\]', name)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2030:
                result.year = year
                name = name[:match.start()] + name[match.end():]

    # Try year with genre: (2024 Pop) or (2024 Hip Hop Rap)
    # But NOT years followed by remaster/release indicators
    # This pattern is intentionally narrow to avoid false positives with remaster years
    if result.year is None:
        # Only match if the content after the year looks like a genre, not a remaster label
        match = re.search(r'\((\d{4})\s+((?:Pop|Rock|Hip\s*Hop|Rap|Jazz|Blues|Country|Metal|Punk|R&B|Soul|Electronic|Dance|Classical|Indie|Alternative|Folk)[A-Za-z\s]*)\)', name, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2030:
                result.year = year
                name = name[:match.start()] + name[match.end():]

    # Standard year extraction if not found yet
    if result.year is None:
        result.year, name = extract_year(name)

    # Now use v1's cleaning logic which handles scene formats well
    # Remove format tags: [FLAC], [MP3 320], [24-96], etc.
    name = re.sub(r'\[.*?\]', '', name)
    name = re.sub(r'\((?:FLAC|MP3|AAC|OGG|WAV|320|256|192|V0|24[\-_]?(?:44|48|96|192)|16[\-_]?44).*?\)', '', name, flags=re.IGNORECASE)

    # Remove common quality/source tags
    name = re.sub(r'(?:FLAC|MP3|AAC|WEB|CD|VINYL|LP|Vinyl|vinyl|CD-?RIP|WEB-?DL|WEB-?FLAC)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(?:320|256|192|V0)\s*(?:kbps|cbr|vbr)?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(?:24|16)[\-_]?(?:44|48|96|192)\b', '', name)

    # Remove release group tags at end
    name = re.sub(r'[\-_]\s*[A-Z0-9]{2,10}$', '', name)

    # Remove uploader names in common formats
    name = re.sub(r'\s*[\-_]?\s*(?:PMEDIA|MiRCrew|GloDLS|HANNIBAL|SilverRG|NimitMak|Michi80|Big Papi)\s*', '', name, flags=re.IGNORECASE)

    # Clean up extra whitespace and punctuation
    name = re.sub(r'\s*[\-_]+\s*$', '', name)
    name = re.sub(r'^\s*[\-_]+\s*', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    # Try to split into Artist - Album
    # Pattern 1: "Artist - Album" with explicit dash separator
    match = re.match(r'^(.+?)\s*[-–—]\s*(.+)$', name)
    if match:
        result.artist = match.group(1).strip()
        result.title = match.group(2).strip()

        # Clean up common artifacts from title
        if result.title:
            result.title = re.sub(r'\s*[-–—]\s*$', '', result.title)
            result.title = re.sub(r'^\s*[-–—]\s*', '', result.title)

    return result


def extract_album_v3(name: str) -> Extraction:
    """
    Version 3: Safer release group removal + edition stripping.

    Improvements over v2:
    - Only remove release groups if they follow known release group patterns (EICHBAUM, MyDad, etc.)
      NOT short album names like GNX, SOS, UTOPIA
    - Strip edition/variant info: (Deluxe), (Store Exclusive), (with Isolated Vocals), (by emi)
    - Remove catalog info in curly braces: {Label, 602475451051}
    """
    result = Extraction()

    # Clean the name
    name = clean_torrent_name(name)

    # Remove emojis early
    name = re.sub(r'[⭐️🎵🎶🔥💿📀😎]+', '', name)

    # Only remove curly braces that clearly contain catalog/label info, not edition names
    # Keep: {Deluxe Edition}, {Expanded Edition}
    # Remove: {XO Republic Records, 0602475872252}, {602475451051}
    name = re.sub(r'\{[^}]*(?:Records|Label|Republic|Capitol|\d{10,})[^}]*\}', '', name, flags=re.IGNORECASE)

    # === Year extraction (same as v2) ===
    # Try date format in brackets: [2024.11.22] or [2024-11-22]
    match = re.search(r'\[(\d{4})[.\-]\d{2}[.\-]\d{2}\]', name)
    if match:
        result.year = int(match.group(1))
        name = name[:match.start()] + name[match.end():]

    # Try year with catalog number: [2024 - 602475686767]
    if result.year is None:
        match = re.search(r'\[(\d{4})\s*[-–]\s*\d+\]', name)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2030:
                result.year = year
                name = name[:match.start()] + name[match.end():]

    # Try year with genre (narrow pattern)
    if result.year is None:
        match = re.search(r'\((\d{4})\s+((?:Pop|Rock|Hip\s*Hop|Rap|Jazz|Blues|Country|Metal|Punk|R&B|Soul|Electronic|Dance|Classical|Indie|Alternative|Folk)[A-Za-z\s]*)\)', name, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2030:
                result.year = year
                name = name[:match.start()] + name[match.end():]

    # Standard year extraction
    if result.year is None:
        result.year, name = extract_year(name)

    # === Remove format tags ===
    name = re.sub(r'\[.*?\]', '', name)
    name = re.sub(r'\((?:FLAC|MP3|AAC|OGG|WAV|320|256|192|V0|24[\-_]?(?:44|48|96|192)|16[\-_]?44).*?\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(Explicit\)', '', name, flags=re.IGNORECASE)

    # Remove uploader/source in parentheses: (Big Papi), (ETTV), etc.
    name = re.sub(r'\((?:Big\s*Papi|ETTV|PMEDIA)\)', '', name, flags=re.IGNORECASE)

    # Remove uploader names that appear with spaces at end: "NimitMak SilverRG"
    name = re.sub(r'\s+(?:NimitMak|SilverRG|Big\s*Papi)\s*(?:NimitMak|SilverRG|Big\s*Papi)*\s*$', '', name, flags=re.IGNORECASE)

    # Remove common quality/source tags
    name = re.sub(r'(?:FLAC|MP3|AAC|WEB|CD|VINYL|LP|Vinyl|vinyl|CD-?RIP|WEB-?DL|WEB-?FLAC)\b', '', name, flags=re.IGNORECASE)
    # Handle @320Kbps format (common in some torrents)
    name = re.sub(r'@?\b(?:320|256|192|V0)\s*(?:kbps|cbr|vbr)?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(?:24|16)[\-_]?(?:44|48|96|192)\b', '', name)
    name = re.sub(r'\b(?:24Bit|16Bit|24BIT|16BIT|16BITS|24BITS)[-_]?\d*k?Hz?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b\d+\.?\d*\s*k?Hz\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\bHi-?Res\b', '', name, flags=re.IGNORECASE)

    # Remove edition/variant info from parentheses (v3 addition)
    # NOTE: Do NOT strip "(Deluxe)", "(Deluxe Edition)", "(Limited Edition)" etc.
    # These are often part of expected titles. Only strip patterns that are clearly extra.
    edition_pattern = r'\((?:' + '|'.join([
        r'00XO\s+Edition',          # 00XO Edition (The Weeknd specific)
        r'Exclusive\s+\d+\s+Track\s+[A-Za-z\s]+',  # Exclusive 21 Track Deluxe Digital
        r'Target\s+Exclusive',      # Target Exclusive
        r'Store\s+Exclusive',       # Store Exclusive
        r'Dolby\s+Atmos(?:\s+Edition)?', # Dolby Atmos, Dolby Atmos Edition
        r'with\s+[^)]+',            # with Isolated Vocals, with bonus tracks
        r'by\s+\w+',                # by emi, by XYZ
    ]) + r')\)'
    name = re.sub(edition_pattern, '', name, flags=re.IGNORECASE)

    # Collapse multiple consecutive dashes/underscores (left over from format removal)
    # Do this early to normalize the string before release group removal
    name = re.sub(r'[-_]{2,}', '-', name)

    # Remove known release group tags at end (explicit list, NOT general pattern)
    # This is safer than the v2 pattern which matched any 2-10 char uppercase string
    # List compiled from common patterns in dataset
    known_groups = [
        'EICHBAUM', 'MyDad', 'Sc4r3cr0w', 'PMEDIA', 'MiRCrew', 'GloDLS',
        'HANNIBAL', 'SilverRG', 'NimitMak', 'Michi80', 'PERFECT', 'ETRG',
        'JLM', 'FLAC', 'WEB', 'ETTV', 'CD', 'MT', 'TBS', 'RNS', 'MIXFIEND',
        'EAC', 'FNT', 'MFA', 'HHI', 'CR', 'HB', 'C4', 'MOD', 'ESC', 'P2P',
        'VAG', 'MARR', 'FLAWL3SS', 'TX', 'OMA', 'XRIP', 'BPM', 'ENRiCH',
        '401', 'sn3hdj3', 'Hunter', 'FTD', 'DGN', 'ePHEMERiD', 'JUST',
        'ENRAGED', 'CDS', '2CD', 'ADVANCE', 'RETAIL', 'CLEAN', 'REMASTER',
        'ALAC', 'INT', 'SHADOW', 'TOBLERONE', 'FAF', 'IDN', 'CREW'
    ]
    group_pattern = r'[\-_]\s*(?:' + '|'.join(known_groups) + r')$'
    # Apply multiple times to handle stacked patterns like -FLAC-JLM
    for _ in range(3):
        name = re.sub(group_pattern, '', name, flags=re.IGNORECASE)

    # Also clean up leftover format info at end: -ES, -WEB-ES, -CD-FLAC, etc.
    name = re.sub(r'[-_](?:WEB|CD|VINYL|LP)[-_]?(?:[A-Z]{2,3})?[-_]?(?:FLAC|MP3)?$', '', name, flags=re.IGNORECASE)

    # Scene release cleanup: remove language/country codes like -ES, -US, -UK at end
    name = re.sub(r'[-_](?:ES|US|UK|DE|FR|JP|KR)$', '', name, flags=re.IGNORECASE)

    # Clean up concatenation artifacts at end: -FLACJLM, -ESOMA, -WEBMOD, etc.
    # These are format tags concatenated with release groups due to year extraction
    format_prefix = r'(?:FLAC|WEB|CD|MP3|AAC|ADVANCE|RETAIL|2CD)'
    # Remove -FLACXYZ, -WEBXYZ, -ADVANCEXYZ, etc. at the end
    name = re.sub(rf'[-_]{format_prefix}[A-Z0-9]{{2,8}}$', '', name, flags=re.IGNORECASE)
    # Also handle formats where 2 items concatenated: -2CDBPM
    name = re.sub(r'[-_]2?CD[A-Z]{2,6}$', '', name, flags=re.IGNORECASE)

    # For scene releases, remove language code + release group concatenation
    # e.g., -ESOMA (ES + OMA), -USRNS (US + RNS)
    # Pattern requires the suffix after lang code to be ALL CAPS (not lowercase like "-Season")
    lang_codes = r'(?:ES|US|UK|DE|FR|JP|KR|EN|IT|NL|PT|PL|RU|SE|NO|DK|FI)'
    name = re.sub(rf'[-_]{lang_codes}[A-Z0-9]{{2,8}}$', '', name)  # No IGNORECASE - must be uppercase

    # Remove catalog numbers at end: MFSL45UD1S-006, SO-11601, Capitol SO-11601
    # Also remove Japanese catalog numbers like 8P-45, M-39
    # Don't use IGNORECASE - catalog prefixes and suffixes should be uppercase
    name = re.sub(r'[-_\s]+(?:MFSL|Capitol\s+SO)[A-Z0-9\-]+$', '', name)
    name = re.sub(r'\s+SO-\d+$', '', name)
    # Remove Japanese/release catalog suffixes: 8P-45, M-39
    name = re.sub(r'\s+[A-Z0-9]+-\d{1,3}$', '', name)

    # Remove trailing year with underscore (like _2023 in "2002_2023" where 2002 was extracted)
    name = re.sub(r'[-_]\d{4}$', '', name)

    # Clean up extra whitespace and punctuation
    name = re.sub(r'\s*[\-_]+\s*$', '', name)
    name = re.sub(r'^\s*[\-_]+\s*', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    # Try to split into Artist - Album
    match = re.match(r'^(.+?)\s*[-–—]\s*(.+)$', name)
    if match:
        result.artist = match.group(1).strip()
        result.title = match.group(2).strip()

        # Clean up common artifacts from title
        if result.title:
            result.title = re.sub(r'\s*[-–—]\s*$', '', result.title)
            result.title = re.sub(r'^\s*[-–—]\s*', '', result.title)

    return result


# Map of heuristic versions to extraction functions
EXTRACTORS = {
    1: extract_album_v1,
    2: extract_album_v2,
    3: extract_album_v3,
}


def extract(name: str, version: int = HEURISTIC_VERSION) -> Extraction:
    """Extract album metadata using specified heuristic version."""
    extractor = EXTRACTORS.get(version, extract_album_v1)
    return extractor(name)


def is_match(extraction: Extraction, expected_artist: str, expected_title: str,
             expected_year: int | None, aliases: list[str], threshold: float = 0.7) -> bool:
    """Check if extraction matches expected values."""
    # Year must match exactly (if expected)
    if expected_year is not None:
        if extraction.year != expected_year:
            return False

    # Artist similarity (check against all aliases)
    artist_sim = best_artist_similarity(extraction.artist, expected_artist, aliases)
    if artist_sim < threshold:
        return False

    # Title similarity
    title_sim = similarity(extraction.title, expected_title)
    if title_sim < threshold:
        return False

    return True


def run_on_todo(conn: sqlite3.Connection) -> tuple[int, int]:
    """Run extraction on todo items, move successes to done."""
    cursor = conn.execute("SELECT id, torrent_name, expected_artist, expected_title, expected_year FROM todo")
    rows = cursor.fetchall()

    moved = 0
    total = len(rows)

    for row in rows:
        id_, torrent_name, expected_artist, expected_title, expected_year = row

        # Get aliases for expected artist
        aliases = get_artist_aliases(conn, expected_artist) if expected_artist else [expected_artist]

        # Run extraction
        extraction = extract(torrent_name)

        # Check if it matches
        if is_match(extraction, expected_artist or "", expected_title or "", expected_year, aliases):
            # Move to done
            conn.execute("""
                INSERT INTO done (torrent_name, expected_artist, expected_title, expected_year,
                                  ground_truth_id, extracted_artist, extracted_title, extracted_year,
                                  heuristic_version)
                SELECT torrent_name, expected_artist, expected_title, expected_year, ground_truth_id,
                       ?, ?, ?, ?
                FROM todo WHERE id = ?
            """, (extraction.artist, extraction.title, extraction.year, HEURISTIC_VERSION, id_))

            conn.execute("DELETE FROM todo WHERE id = ?", (id_,))
            moved += 1

    conn.commit()
    return moved, total


def run_regression(conn: sqlite3.Connection) -> tuple[int, int, list[str]]:
    """Verify all done items still pass with current heuristics."""
    cursor = conn.execute("""
        SELECT id, torrent_name, expected_artist, expected_title, expected_year
        FROM done
    """)
    rows = cursor.fetchall()

    passed = 0
    failed_items = []

    for row in rows:
        id_, torrent_name, expected_artist, expected_title, expected_year = row

        aliases = get_artist_aliases(conn, expected_artist) if expected_artist else [expected_artist]
        extraction = extract(torrent_name)

        if is_match(extraction, expected_artist or "", expected_title or "", expected_year, aliases):
            passed += 1
        else:
            failed_items.append(torrent_name)

    return passed, len(rows), failed_items


def show_stats(conn: sqlite3.Connection):
    """Show current statistics."""
    todo_count = conn.execute("SELECT COUNT(*) FROM todo").fetchone()[0]
    done_count = conn.execute("SELECT COUNT(*) FROM done").fetchone()[0]
    total = todo_count + done_count

    print(f"Total samples: {total}")
    print(f"Done: {done_count} ({100*done_count/total:.1f}%)")
    print(f"Todo: {todo_count} ({100*todo_count/total:.1f}%)")
    print(f"Current heuristic version: {HEURISTIC_VERSION}")


def show_failures(conn: sqlite3.Connection, limit: int = 20):
    """Show sample of todo items with extraction attempts."""
    cursor = conn.execute("""
        SELECT torrent_name, expected_artist, expected_title, expected_year
        FROM todo LIMIT ?
    """, (limit,))

    print(f"\nSample failures (first {limit}):")
    print("-" * 80)

    for row in cursor:
        torrent_name, expected_artist, expected_title, expected_year = row
        extraction = extract(torrent_name)
        aliases = get_artist_aliases(conn, expected_artist) if expected_artist else [expected_artist]

        artist_sim = best_artist_similarity(extraction.artist, expected_artist or "", aliases)
        title_sim = similarity(extraction.title, expected_title)
        year_ok = extraction.year == expected_year

        print(f"Input: {torrent_name[:70]}...")
        print(f"  Expected: {expected_artist} / {expected_title} / {expected_year}")
        print(f"  Got:      {extraction.artist} / {extraction.title} / {extraction.year}")
        print(f"  Scores:   artist={artist_sim:.2f} title={title_sim:.2f} year={'OK' if year_ok else 'FAIL'}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python album_heuristics.py [run|regress|stats|failures]")
        sys.exit(1)

    cmd = sys.argv[1]
    conn = sqlite3.connect(DB_PATH)

    if cmd == "run":
        print(f"Running heuristics v{HEURISTIC_VERSION} on todo items...")
        moved, total = run_on_todo(conn)
        print(f"Moved {moved}/{total} items to done")
        show_stats(conn)

    elif cmd == "regress":
        print(f"Running regression test with heuristics v{HEURISTIC_VERSION}...")
        passed, total, failures = run_regression(conn)
        print(f"Passed: {passed}/{total}")
        if failures:
            print(f"REGRESSION DETECTED! {len(failures)} items failed:")
            for f in failures[:10]:
                print(f"  - {f[:60]}...")
        else:
            print("All items still pass!")

    elif cmd == "stats":
        show_stats(conn)

    elif cmd == "failures":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        show_failures(conn, limit)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    conn.close()


if __name__ == "__main__":
    main()
