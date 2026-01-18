#!/usr/bin/env python3
"""Monitor consensus labeling progress."""

import sqlite3
import time
from collections import Counter
from pathlib import Path
import argparse

OLD_DB_PATH = Path(__file__).parent.parent / "data" / "consensus_labels.db"
NEW_DB_PATH = Path(__file__).parent.parent / "data" / "medium_labels.db"
OLD_VALID = {"music", "video", "software", "book", "porn", "other"}
NEW_VALID = {"audio", "video", "software", "book", "other"}

# Track previous counts for rate calculation
prev_counts = {}
prev_time = None


def get_invalid_stats(c, col, valid_set):
    """Get count and set of invalid labels for a column."""
    rows = c.execute(f"SELECT {col} FROM samples WHERE {col} IS NOT NULL").fetchall()
    invalid = [r[0] for r in rows if r[0] not in valid_set]
    return len(invalid), set(invalid)


def calc_rate_eta(col, count, total, elapsed):
    """Calculate req/s and ETA for a model."""
    global prev_counts

    if elapsed <= 0:
        return "", ""

    prev = prev_counts.get(col, 0)
    diff = count - prev

    if diff <= 0:
        return "", ""

    rate = diff / elapsed
    remaining = total - count
    eta_secs = remaining / rate if rate > 0 else 0

    if eta_secs < 60:
        eta = f"{eta_secs:.0f}s"
    elif eta_secs < 3600:
        eta = f"{eta_secs/60:.1f}m"
    else:
        eta = f"{eta_secs/3600:.1f}h"

    return f"{rate:.1f}/s", eta


def monitor_old():
    """Monitor old 6-class consensus labeling."""
    global prev_counts, prev_time

    while True:
        try:
            conn = sqlite3.connect(OLD_DB_PATH)
            c = conn.cursor()

            total = c.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            qwen = c.execute("SELECT COUNT(*) FROM samples WHERE qwen IS NOT NULL").fetchone()[0]
            gemma = c.execute("SELECT COUNT(*) FROM samples WHERE gemma IS NOT NULL").fetchone()[0]
            mistral = c.execute("SELECT COUNT(*) FROM samples WHERE mistral IS NOT NULL").fetchone()[0]
            qwen3 = c.execute("SELECT COUNT(*) FROM samples WHERE qwen3coder IS NOT NULL").fetchone()[0]
            consensus = c.execute("SELECT COUNT(*) FROM samples WHERE consensus IS NOT NULL").fetchone()[0]
            majority = c.execute("SELECT COUNT(*) FROM samples WHERE majority IS NOT NULL").fetchone()[0]

            # Get invalid labels per model
            qwen_inv, _ = get_invalid_stats(c, "qwen", OLD_VALID)
            gemma_inv, _ = get_invalid_stats(c, "gemma", OLD_VALID)
            mistral_inv, _ = get_invalid_stats(c, "mistral", OLD_VALID)
            qwen3_inv, _ = get_invalid_stats(c, "qwen3coder", OLD_VALID)

            # Compute 3-model agreement on the fly
            three_agree = c.execute("""
                SELECT COUNT(*) FROM samples
                WHERE qwen IS NOT NULL AND gemma IS NOT NULL AND mistral IS NOT NULL
                AND qwen IN ('music','video','software','book','porn','other')
                AND gemma IN ('music','video','software','book','porn','other')
                AND mistral IN ('music','video','software','book','porn','other')
                AND qwen = gemma AND gemma = mistral
            """).fetchone()[0]

            three_labeled = c.execute("""
                SELECT COUNT(*) FROM samples
                WHERE qwen IS NOT NULL AND gemma IS NOT NULL AND mistral IS NOT NULL
            """).fetchone()[0]

            conn.close()

            # Calculate rates
            now = time.time()
            elapsed = now - prev_time if prev_time else 0

            qwen_rate, qwen_eta = calc_rate_eta("qwen", qwen, total, elapsed)
            gemma_rate, gemma_eta = calc_rate_eta("gemma", gemma, total, elapsed)
            mistral_rate, mistral_eta = calc_rate_eta("mistral", mistral, total, elapsed)
            qwen3_rate, qwen3_eta = calc_rate_eta("qwen3coder", qwen3, total, elapsed)

            # Update previous counts
            prev_counts = {"qwen": qwen, "gemma": gemma, "mistral": mistral, "qwen3coder": qwen3}
            prev_time = now

            def pct_invalid(inv, labeled):
                return f"({100*inv/labeled:.1f}% err)" if labeled > 0 else ""

            def format_rate_eta(rate, eta):
                if rate and eta:
                    return f"  {rate:>7} ETA {eta:>6}"
                return ""

            print(f"\033[2J\033[H")  # Clear screen
            print(f"=== OLD: 6-Class Consensus Labeling ===\n")
            print(f"Total samples: {total:,}\n")
            print(f"  qwen2.5:3b   {qwen:>6,} / {total:,}  ({100*qwen/total:>5.1f}%) {pct_invalid(qwen_inv, qwen):>12}{format_rate_eta(qwen_rate, qwen_eta)}")
            print(f"  gemma3:4b    {gemma:>6,} / {total:,}  ({100*gemma/total:>5.1f}%) {pct_invalid(gemma_inv, gemma):>12}{format_rate_eta(gemma_rate, gemma_eta)}")
            print(f"  mistral:7b   {mistral:>6,} / {total:,}  ({100*mistral/total:>5.1f}%) {pct_invalid(mistral_inv, mistral):>12}{format_rate_eta(mistral_rate, mistral_eta)}")
            print(f"  qwen3-coder  {qwen3:>6,} / {total:,}  ({100*qwen3/total:>5.1f}%) {pct_invalid(qwen3_inv, qwen3):>12}{format_rate_eta(qwen3_rate, qwen3_eta)}")

            print(f"\nResults (3 small models):")
            print(f"  All labeled: {three_labeled:>6,}")
            print(f"  All 3 agree: {three_agree:>6,} ({100*three_agree/three_labeled:.1f}%)" if three_labeled > 0 else f"  All 3 agree: {three_agree:>6,}")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(5)


def monitor_new():
    """Monitor new 5-class medium labeling."""
    global prev_counts, prev_time

    while True:
        try:
            if not NEW_DB_PATH.exists():
                print("New DB not created yet...")
                time.sleep(5)
                continue

            conn = sqlite3.connect(NEW_DB_PATH)
            c = conn.cursor()

            total = c.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

            # Old labels (converted)
            old_qwen = c.execute("SELECT COUNT(*) FROM samples WHERE old_qwen IS NOT NULL").fetchone()[0]
            old_gemma = c.execute("SELECT COUNT(*) FROM samples WHERE old_gemma IS NOT NULL").fetchone()[0]
            old_mistral = c.execute("SELECT COUNT(*) FROM samples WHERE old_mistral IS NOT NULL").fetchone()[0]
            old_qwen3 = c.execute("SELECT COUNT(*) FROM samples WHERE old_qwen3coder IS NOT NULL").fetchone()[0]

            # New labels
            new_qwen = c.execute("SELECT COUNT(*) FROM samples WHERE new_qwen IS NOT NULL").fetchone()[0]
            new_gemma = c.execute("SELECT COUNT(*) FROM samples WHERE new_gemma IS NOT NULL").fetchone()[0]
            new_mistral = c.execute("SELECT COUNT(*) FROM samples WHERE new_mistral IS NOT NULL").fetchone()[0]
            new_big = c.execute("SELECT COUNT(*) FROM samples WHERE new_big IS NOT NULL").fetchone()[0]

            # Consensus
            with_medium = c.execute("SELECT COUNT(*) FROM samples WHERE medium IS NOT NULL").fetchone()[0]

            # Invalid counts for new labels
            new_qwen_inv, _ = get_invalid_stats(c, "new_qwen", NEW_VALID)
            new_gemma_inv, _ = get_invalid_stats(c, "new_gemma", NEW_VALID)

            conn.close()

            # Calculate rates
            now = time.time()
            elapsed = now - prev_time if prev_time else 0

            qwen_rate, qwen_eta = calc_rate_eta("new_qwen", new_qwen, total, elapsed)
            gemma_rate, gemma_eta = calc_rate_eta("new_gemma", new_gemma, total, elapsed)
            mistral_rate, mistral_eta = calc_rate_eta("new_mistral", new_mistral, total, elapsed)
            big_rate, big_eta = calc_rate_eta("new_big", new_big, total, elapsed)

            # Update previous counts
            prev_counts = {"new_qwen": new_qwen, "new_gemma": new_gemma, "new_mistral": new_mistral, "new_big": new_big}
            prev_time = now

            def pct_invalid(inv, labeled):
                return f"({100*inv/labeled:.1f}% err)" if labeled > 0 else ""

            def format_rate_eta(rate, eta):
                if rate and eta:
                    return f"  {rate:>7} ETA {eta:>6}"
                return ""

            print(f"\033[2J\033[H")  # Clear screen
            print(f"=== NEW: 5-Class Medium Labeling ===\n")
            print(f"Total samples: {total:,}\n")

            print(f"Old labels (converted, NULL if was porn):")
            print(f"  old_qwen:      {old_qwen:>6,} / {total:,}  ({100*old_qwen/total:>5.1f}%)")
            print(f"  old_gemma:     {old_gemma:>6,} / {total:,}  ({100*old_gemma/total:>5.1f}%)")
            print(f"  old_mistral:   {old_mistral:>6,} / {total:,}  ({100*old_mistral/total:>5.1f}%)")
            print(f"  old_qwen3:     {old_qwen3:>6,} / {total:,}  ({100*old_qwen3/total:>5.1f}%)")

            print(f"\nNew labels (5-class prompt):")
            print(f"  new_qwen:      {new_qwen:>6,} / {total:,}  ({100*new_qwen/total:>5.1f}%) {pct_invalid(new_qwen_inv, new_qwen):>12}{format_rate_eta(qwen_rate, qwen_eta)}")
            print(f"  new_gemma:     {new_gemma:>6,} / {total:,}  ({100*new_gemma/total:>5.1f}%) {pct_invalid(new_gemma_inv, new_gemma):>12}{format_rate_eta(gemma_rate, gemma_eta)}")
            print(f"  new_mistral:   {new_mistral:>6,} / {total:,}  (cascade){format_rate_eta(mistral_rate, mistral_eta)}")
            print(f"  new_big:       {new_big:>6,} / {total:,}  (cascade){format_rate_eta(big_rate, big_eta)}")

            print(f"\nConsensus (3+ agree):")
            print(f"  With medium:   {with_medium:>6,} / {total:,}  ({100*with_medium/total:.1f}%)")
            print(f"  Without:       {total - with_medium:>6,}")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Monitor labeling progress")
    parser.add_argument("--new", action="store_true", help="Monitor new 5-class medium labeling")
    args = parser.parse_args()

    if args.new:
        monitor_new()
    else:
        monitor_old()


if __name__ == "__main__":
    main()
