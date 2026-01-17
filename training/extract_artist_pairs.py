#!/usr/bin/env python3
"""Extract artist name matching pairs from MusicBrainz JSON dump."""

import json
import random
import re
import tarfile
from pathlib import Path

# Stream directly from compressed archive
ARTIST_ARCHIVE = Path(__file__).parent.parent / "artist.tar.xz"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "artist_pairs.jsonl"

# Target counts
TARGET_POSITIVE = 5000  # Half positive, half negative for 10k total
TARGET_NEGATIVE = 5000


def normalize(name: str) -> str:
    """Normalize name for comparison."""
    return re.sub(r'[^\w\s]', '', name.lower()).strip()


def generate_synthetic_variation(name: str) -> str | None:
    """Generate a synthetic variation of a name."""
    variations = []

    # Case variations
    if name != name.upper():
        variations.append(name.upper())
    if name != name.lower():
        variations.append(name.lower())
    if name != name.title():
        variations.append(name.title())

    # Remove "The" prefix
    if name.lower().startswith("the "):
        variations.append(name[4:])
    else:
        variations.append("The " + name)

    # Remove punctuation
    no_punct = re.sub(r"['\-\.]", "", name)
    if no_punct != name:
        variations.append(no_punct)

    # Replace & with "and"
    if "&" in name:
        variations.append(name.replace("&", "and"))
    if " and " in name.lower():
        variations.append(re.sub(r'\band\b', '&', name, flags=re.IGNORECASE))

    return random.choice(variations) if variations else None


def extract_pairs():
    """Extract artist name pairs from the dump."""
    positive_pairs = []
    all_names = []

    print(f"Streaming from {ARTIST_ARCHIVE}...")

    with tarfile.open(ARTIST_ARCHIVE, 'r:xz') as tar:
        # Find the artist file in the archive
        for member in tar.getmembers():
            if member.name.endswith('/artist') or member.name == 'artist':
                print(f"  Found: {member.name} ({member.size:,} bytes)")
                f = tar.extractfile(member)
                break
        else:
            print("Could not find artist file in archive")
            return

        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"  Processed {i:,} artists, found {len(positive_pairs):,} positive pairs")

            try:
                artist = json.loads(line.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            name = artist.get('name', '').strip()
            if not name:
                continue

            all_names.append(name)
            aliases = artist.get('aliases', [])

            # Create positive pairs from aliases
            for alias in aliases:
                alias_name = alias.get('name', '').strip()
                if alias_name and normalize(alias_name) != normalize(name):
                    positive_pairs.append({
                        'name1': name,
                        'name2': alias_name,
                        'match': True
                    })

                    if len(positive_pairs) >= TARGET_POSITIVE:
                        break

            # Add synthetic variations
            if len(positive_pairs) < TARGET_POSITIVE:
                synth = generate_synthetic_variation(name)
                if synth and normalize(synth) != normalize(name):
                    positive_pairs.append({
                        'name1': name,
                        'name2': synth,
                        'match': True
                    })

    print(f"Found {len(positive_pairs):,} positive pairs from {len(all_names):,} artists")

    # Limit positive pairs
    if len(positive_pairs) > TARGET_POSITIVE:
        positive_pairs = random.sample(positive_pairs, TARGET_POSITIVE)

    # Generate negative pairs (random non-matching artists)
    print("Generating negative pairs...")
    negative_pairs = []
    attempts = 0
    max_attempts = TARGET_NEGATIVE * 10

    while len(negative_pairs) < TARGET_NEGATIVE and attempts < max_attempts:
        attempts += 1
        name1, name2 = random.sample(all_names, 2)

        # Ensure they're actually different
        if normalize(name1) != normalize(name2):
            negative_pairs.append({
                'name1': name1,
                'name2': name2,
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
