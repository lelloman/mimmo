#!/usr/bin/env python3
"""Extract album name matching pairs from MusicBrainz JSON dump."""

import json
import random
import re
import tarfile
from pathlib import Path

# Stream directly from compressed archive
RELEASE_GROUP_ARCHIVE = Path(__file__).parent.parent / "release-group.tar.xz"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "album_pairs.jsonl"

TARGET_POSITIVE = 5000
TARGET_NEGATIVE = 5000


def normalize(name: str) -> str:
    """Normalize name for comparison."""
    return re.sub(r'[^\w\s]', '', name.lower()).strip()


def generate_synthetic_variation(title: str) -> str | None:
    """Generate a synthetic variation of an album title."""
    variations = []

    # Case variations
    if title != title.upper():
        variations.append(title.upper())
    if title != title.lower():
        variations.append(title.lower())

    # Remove "The" prefix
    if title.lower().startswith("the "):
        variations.append(title[4:])
    elif not title.lower().startswith("the "):
        variations.append("The " + title)

    # Remove punctuation
    no_punct = re.sub(r"['\-\.\:\,]", "", title)
    if no_punct != title:
        variations.append(no_punct)

    # Add common suffixes
    suffixes = [
        " (Remastered)",
        " (Deluxe Edition)",
        " [Remaster]",
        " - Deluxe",
        " (Anniversary Edition)",
    ]
    variations.append(title + random.choice(suffixes))

    # Remove common suffixes
    for suffix_pattern in [
        r'\s*\(Remastered\)',
        r'\s*\(Deluxe.*?\)',
        r'\s*\[Remaster\]',
        r'\s*-\s*Deluxe',
        r'\s*\(.*?Edition\)',
    ]:
        cleaned = re.sub(suffix_pattern, '', title, flags=re.IGNORECASE)
        if cleaned != title:
            variations.append(cleaned.strip())

    return random.choice(variations) if variations else None


def extract_pairs():
    """Extract album name pairs from the dump."""
    positive_pairs = []
    all_titles = []

    print(f"Streaming from {RELEASE_GROUP_ARCHIVE}...")

    with tarfile.open(RELEASE_GROUP_ARCHIVE, 'r:xz') as tar:
        # Find the release-group file in the archive
        for member in tar.getmembers():
            if 'release-group' in member.name and not member.isdir():
                print(f"  Found: {member.name} ({member.size:,} bytes)")
                f = tar.extractfile(member)
                break
        else:
            print("Could not find release-group file in archive")
            return

        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"  Processed {i:,} release groups, found {len(positive_pairs):,} positive pairs")

            try:
                rg = json.loads(line.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            title = rg.get('title', '').strip()
            if not title:
                continue

            all_titles.append(title)
            aliases = rg.get('aliases', [])

            # Create positive pairs from aliases
            for alias in aliases:
                alias_name = alias.get('name', '').strip()
                if alias_name and normalize(alias_name) != normalize(title):
                    positive_pairs.append({
                        'name1': title,
                        'name2': alias_name,
                        'match': True
                    })

            # Generate synthetic variation
            if len(positive_pairs) < TARGET_POSITIVE:
                synth = generate_synthetic_variation(title)
                if synth:
                    positive_pairs.append({
                        'name1': title,
                        'name2': synth,
                        'match': True
                    })

    print(f"Found {len(positive_pairs):,} positive pairs from {len(all_titles):,} release groups")

    # Limit positive pairs
    if len(positive_pairs) > TARGET_POSITIVE:
        positive_pairs = random.sample(positive_pairs, TARGET_POSITIVE)

    # Generate negative pairs
    print("Generating negative pairs...")
    negative_pairs = []
    attempts = 0
    max_attempts = TARGET_NEGATIVE * 10

    while len(negative_pairs) < TARGET_NEGATIVE and attempts < max_attempts:
        attempts += 1
        title1, title2 = random.sample(all_titles, 2)

        if normalize(title1) != normalize(title2):
            negative_pairs.append({
                'name1': title1,
                'name2': title2,
                'match': False
            })

    print(f"Generated {len(negative_pairs):,} negative pairs")

    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Wrote {len(all_pairs):,} pairs to {OUTPUT_FILE}")

    # Print samples
    print("\nSample positive pairs:")
    for p in random.sample([p for p in all_pairs if p['match']], min(5, len(positive_pairs))):
        print(f"  {p['name1']!r} <-> {p['name2']!r}")

    print("\nSample negative pairs:")
    for p in random.sample([p for p in all_pairs if not p['match']], min(5, len(negative_pairs))):
        print(f"  {p['name1']!r} <-> {p['name2']!r}")


if __name__ == '__main__':
    random.seed(42)
    extract_pairs()
