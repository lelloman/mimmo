#!/usr/bin/env python3
"""Interactive script to fix samples with invalid labels in the consensus database."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "consensus_labels.db"
VALID_CATEGORIES = {"music", "video", "software", "book", "porn", "other"}

# Key shortcuts
SHORTCUTS = {
    "m": "music",
    "v": "video",
    "s": "software",
    "b": "book",
    "p": "porn",
    "o": "other",
    "x": "skip",  # skip this one
    "q": "quit",  # quit the script
}


def get_samples_with_invalid_labels(conn):
    """Get all samples that have at least one invalid label."""
    c = conn.cursor()

    query = """
    SELECT id, name, files_json, qwen, gemma, mistral, qwen3coder
    FROM samples
    WHERE (qwen IS NOT NULL AND qwen NOT IN ('music','video','software','book','porn','other'))
       OR (gemma IS NOT NULL AND gemma NOT IN ('music','video','software','book','porn','other'))
       OR (mistral IS NOT NULL AND mistral NOT IN ('music','video','software','book','porn','other'))
       OR (qwen3coder IS NOT NULL AND qwen3coder NOT IN ('music','video','software','book','porn','other'))
    ORDER BY id
    """
    return c.execute(query).fetchall()


def get_invalid_columns(qwen, gemma, mistral, qwen3coder):
    """Return list of (column_name, invalid_value) for columns with invalid labels."""
    invalid = []
    for col, val in [("qwen", qwen), ("gemma", gemma), ("mistral", mistral), ("qwen3coder", qwen3coder)]:
        if val is not None and val not in VALID_CATEGORIES:
            invalid.append((col, val))
    return invalid


def fix_sample(conn, sample_id, new_label):
    """Set all invalid labels for this sample to the new label."""
    c = conn.cursor()

    # Get current values
    row = c.execute(
        "SELECT qwen, gemma, mistral, qwen3coder FROM samples WHERE id = ?",
        (sample_id,)
    ).fetchone()

    qwen, gemma, mistral, qwen3coder = row
    updates = []

    for col, val in [("qwen", qwen), ("gemma", gemma), ("mistral", mistral), ("qwen3coder", qwen3coder)]:
        if val is not None and val not in VALID_CATEGORIES:
            updates.append(col)

    for col in updates:
        c.execute(f"UPDATE samples SET {col} = ? WHERE id = ?", (new_label, sample_id))

    conn.commit()
    return updates


def main():
    conn = sqlite3.connect(DB_PATH)

    print("=" * 60)
    print("INVALID LABEL FIXUP TOOL (per-sample)")
    print("=" * 60)
    print("\nShortcuts:")
    for key, value in SHORTCUTS.items():
        print(f"  {key} = {value}")
    print()

    samples = get_samples_with_invalid_labels(conn)
    total = len(samples)

    print(f"Found {total} samples with invalid labels to process\n")

    fixed = 0
    skipped = 0

    for i, (sid, name, files_json, qwen, gemma, mistral, qwen3coder) in enumerate(samples, 1):
        invalid_cols = get_invalid_columns(qwen, gemma, mistral, qwen3coder)

        # Parse files JSON
        import json
        try:
            files = json.loads(files_json) if files_json else []
        except json.JSONDecodeError:
            files = []

        print("-" * 60)
        print(f"[{i}/{total}] Sample ID: {sid}")
        print("-" * 60)
        print(f"\n  NAME: {name}\n")
        if files:
            print(f"  FILES:")
            for f in files[:15]:  # Show up to 15 files
                print(f"    {f}")
            if len(files) > 15:
                print(f"    ... and {len(files) - 15} more files")
        print()
        print(f"  LABELS:")
        print(f"    qwen:       {qwen}")
        print(f"    gemma:      {gemma}")
        print(f"    mistral:    {mistral}")
        print(f"    qwen3coder: {qwen3coder}")
        print()
        print(f"  INVALID: {', '.join(f'{col}={val!r}' for col, val in invalid_cols)}")
        print()
        print(f"  [m]usic [v]ideo [s]oftware [b]ook [p]orn [o]ther | [x]skip [q]uit")

        while True:
            choice = input("  > ").strip().lower()

            if choice == "q":
                print(f"\nQuitting. Fixed {fixed}, skipped {skipped}.")
                conn.close()
                return

            if choice == "x":
                print("  Skipped.")
                skipped += 1
                break

            if choice in SHORTCUTS and SHORTCUTS[choice] in VALID_CATEGORIES:
                new_label = SHORTCUTS[choice]
                updated_cols = fix_sample(conn, sid, new_label)
                print(f"  Fixed {', '.join(updated_cols)} -> {new_label}")
                fixed += 1
                break

            print("  Invalid choice. Use m/v/s/b/p/o/x/q")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Fixed: {fixed}")
    print(f"Skipped: {skipped}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
