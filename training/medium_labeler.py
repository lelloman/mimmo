#!/usr/bin/env python3
"""Medium-type labeler for BERT training data (5 classes: audio, video, software, book, other).

This script re-labels existing samples with a cleaner taxonomy that separates
medium type from content type (porn is now a separate binary classifier).

Cascade strategy:
1. Run qwen2.5:3b + gemma3:4b with new 5-class prompt
2. Compare with old 6-class labels (treating old "porn" as needing re-label)
3. If 3+ models agree (new + old non-porn) -> consensus
4. If disagreement -> run mistral:7b
5. Still no consensus -> run qpt-oss:120b

Usage:
    # Run 2 small models on all samples
    python medium_labeler.py --model small2

    # Run mistral on samples without consensus
    python medium_labeler.py --model mistral-cascade

    # Run big model on remaining samples
    python medium_labeler.py --model big-cascade

    # Show stats
    python medium_labeler.py --stats
"""

import argparse
import json
import sqlite3
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

OLD_DB = Path(__file__).parent.parent / "data" / "consensus_labels.db"
NEW_DB = Path(__file__).parent.parent / "data" / "medium_labels.db"

# Endpoints
RTX_OLLAMA_URL = "http://192.168.1.92:11434"      # RTX 4090 - small models
STRIX_HALO_URL = "http://192.168.1.102:8080"      # Strix Halo - big model

# New 5-class categories (medium type only)
CATEGORIES = ["audio", "video", "software", "book", "other"]

# Map old 6-class to new 5-class
OLD_TO_NEW = {
    "music": "audio",
    "video": "video",
    "software": "software",
    "book": "book",
    "porn": None,  # Needs re-labeling - can't infer medium from this
    "other": "other",
}

# Response normalization for new schema
NORMALIZE_MAP = {
    # Audio aliases
    "music": "audio", "soundtrack": "audio", "album": "audio",
    "mp3": "audio", "flac": "audio", "audiobook": "audio",
    # Video aliases
    "movie": "video", "movies": "video", "film": "video", "films": "video",
    "tv": "video", "tvshow": "video", "tv show": "video", "tv-show": "video",
    "television": "video", "anime": "video", "documentary": "video",
    "series": "video", "drama": "video", "sports": "video",
    "porn": "video", "adult": "video", "xxx": "video", "hentai": "video",  # Porn is video medium
    # Book aliases
    "comic": "book", "comics": "book", "manga": "book",
    "ebook": "book", "e-book": "book", "pdf": "book",
    # Software aliases
    "game": "software", "games": "software",
    "application": "software", "app": "software",
}

# =============================================================================
# UTILITIES
# =============================================================================

def human_size(n):
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.1f}{u}'
        n /= 1024
    return f'{n:.1f}PB'


def normalize_response(response):
    """Normalize LLM response to valid category."""
    response = response.strip().lower()

    if response in CATEGORIES:
        return response

    for key, val in NORMALIZE_MAP.items():
        if key in response:
            return val

    for cat in CATEGORIES:
        if cat in response:
            return cat

    return None


def convert_old_label(old_label):
    """Convert old 6-class label to new 5-class. Returns None if can't convert."""
    if old_label is None:
        return None
    old_label = old_label.lower().strip()
    if old_label in OLD_TO_NEW:
        return OLD_TO_NEW[old_label]
    # Check if it's already a valid new category
    if old_label in CATEGORIES:
        return old_label
    return None


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def build_prompt(name, files_json):
    """Build classification prompt with top 3 largest files."""
    files = json.loads(files_json) if files_json else []
    files_sorted = sorted(files, key=lambda x: x[0], reverse=True)[:3]

    files_str = "\n".join(
        f"  {Path(path).name} ({human_size(size)})"
        for size, path in files_sorted
    )

    return f"""Classify this torrent by MEDIUM TYPE. Reply with exactly one word: audio, video, software, book, or other

Note:
- "audio" = music, podcasts, audiobooks, sound files
- "video" = movies, TV shows, anime, documentaries, adult videos
- "software" = applications, games, operating systems
- "book" = ebooks, comics, manga, PDFs, documents
- "other" = images, archives, data, mixed content

Torrent: {name}
Top files:
{files_str}

Medium type:"""


# =============================================================================
# LLM CLASSIFICATION
# =============================================================================

def classify_ollama(prompt, model_name, timeout=60):
    """Classify using Ollama API (RTX)."""
    data = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 20, "num_ctx": 2048}
    }).encode()

    try:
        req = urllib.request.Request(
            f"{RTX_OLLAMA_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
            response = result.get("response", "").strip().lower()
        elapsed = time.time() - start

        normalized = normalize_response(response)
        if normalized:
            return normalized, elapsed
        return f"?({response[:20]})", elapsed
    except Exception as e:
        return f"ERR({str(e)[:15]})", 0.0


def classify_openai(prompt, model_name=None, timeout=60):
    """Classify using OpenAI API (Strix Halo)."""
    data = json.dumps({
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0.0
    }).encode()

    try:
        req = urllib.request.Request(
            f"{STRIX_HALO_URL}/v1/completions",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
            response = result.get("choices", [{}])[0].get("text", "").strip().lower()
        elapsed = time.time() - start

        normalized = normalize_response(response)
        if normalized:
            return normalized, elapsed
        return f"?({response[:20]})", elapsed
    except Exception as e:
        return f"ERR({str(e)[:15]})", 0.0


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def init_new_db():
    """Initialize new database, copying samples from old DB."""
    NEW_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(NEW_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            files_json TEXT,
            total_size INTEGER,
            -- Old labels (converted to 5-class, NULL if was porn)
            old_qwen TEXT,
            old_gemma TEXT,
            old_mistral TEXT,
            old_qwen3coder TEXT,
            -- New labels (5-class)
            new_qwen TEXT,
            new_qwen_time REAL,
            new_gemma TEXT,
            new_gemma_time REAL,
            new_mistral TEXT,
            new_mistral_time REAL,
            new_big TEXT,
            new_big_time REAL,
            -- Consensus
            medium TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_medium ON samples(medium)")
    conn.commit()
    return conn


def copy_from_old_db(conn):
    """Copy samples from old DB and convert labels."""
    # Check if already populated
    count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    if count > 0:
        print(f"New DB already has {count} samples")
        return

    old_conn = sqlite3.connect(OLD_DB)
    cursor = old_conn.execute("""
        SELECT id, name, files_json, total_size, qwen, gemma, mistral, qwen3coder
        FROM samples
    """)

    inserted = 0
    for row in cursor:
        sid, name, files_json, total_size, qwen, gemma, mistral, qwen3coder = row

        # Convert old labels to new 5-class
        old_qwen = convert_old_label(qwen)
        old_gemma = convert_old_label(gemma)
        old_mistral = convert_old_label(mistral)
        old_qwen3coder = convert_old_label(qwen3coder)

        conn.execute("""
            INSERT INTO samples (id, name, files_json, total_size,
                                 old_qwen, old_gemma, old_mistral, old_qwen3coder)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (sid, name, files_json, total_size,
              old_qwen, old_gemma, old_mistral, old_qwen3coder))
        inserted += 1

    conn.commit()
    old_conn.close()
    print(f"Copied {inserted} samples from old DB")


def label_with_model(conn, model_name, col_name, workers=4, use_openai=False):
    """Label samples with a single model."""
    cursor = conn.execute(
        f"SELECT id, name, files_json FROM samples WHERE {col_name} IS NULL"
    )
    rows = cursor.fetchall()

    if not rows:
        print(f"  All samples already labeled with {model_name}")
        return

    api_type = "OpenAI (Strix Halo)" if use_openai else "Ollama (RTX)"
    print(f"  Labeling {len(rows)} samples with {model_name} via {api_type} (workers={workers})...")
    start = time.time()
    completed = 0

    def process(row):
        sid, name, files_json = row
        prompt = build_prompt(name, files_json)
        if use_openai:
            label, elapsed = classify_openai(prompt, model_name)
        else:
            label, elapsed = classify_ollama(prompt, model_name)
        return sid, label, elapsed

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, row): row for row in rows}
        for future in as_completed(futures):
            sid, label, elapsed = future.result()

            for attempt in range(10):
                try:
                    conn.execute(
                        f"UPDATE samples SET {col_name} = ?, {col_name}_time = ? WHERE id = ?",
                        (label, elapsed, sid)
                    )
                    conn.commit()
                    break
                except sqlite3.OperationalError:
                    time.sleep(0.1 * (attempt + 1))

            completed += 1
            if completed % 100 == 0:
                rate = completed / (time.time() - start)
                remaining = (len(rows) - completed) / rate if rate > 0 else 0
                print(f"    {completed}/{len(rows)} ({rate:.1f} req/s, ~{remaining/60:.1f}m left)")

    elapsed = time.time() - start
    print(f"    Done: {completed} samples in {elapsed:.1f}s ({completed/elapsed:.1f} req/s)")


def compute_consensus(conn):
    """Compute medium consensus using cascade logic.

    For each sample, collect all valid labels (old converted + new):
    - old_qwen, old_gemma, old_mistral, old_qwen3coder (if not None = wasn't porn)
    - new_qwen, new_gemma
    - new_mistral (if run)
    - new_big (if run)

    If 3+ agree on same label -> consensus
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, old_qwen, old_gemma, old_mistral, old_qwen3coder, new_qwen, new_gemma, new_mistral, new_big FROM samples")

    updated = 0
    for row in cursor.fetchall():
        sid = row[0]
        labels = []

        # Collect all valid labels
        for label in row[1:]:
            if label and label in CATEGORIES:
                labels.append(label)

        medium = None
        if labels:
            counts = Counter(labels)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= 3:
                medium = most_common[0]

        conn.execute("UPDATE samples SET medium = ? WHERE id = ?", (medium, sid))
        updated += 1

    conn.commit()
    print(f"Updated consensus for {updated} samples")


def get_samples_needing_mistral(conn):
    """Get samples that don't have consensus after 2 small models."""
    cursor = conn.execute("""
        SELECT id FROM samples
        WHERE new_qwen IS NOT NULL AND new_gemma IS NOT NULL
        AND new_mistral IS NULL
        AND medium IS NULL
    """)
    return [row[0] for row in cursor.fetchall()]


def get_samples_needing_big(conn):
    """Get samples that don't have consensus after mistral."""
    cursor = conn.execute("""
        SELECT id FROM samples
        WHERE new_qwen IS NOT NULL AND new_gemma IS NOT NULL AND new_mistral IS NOT NULL
        AND new_big IS NULL
        AND medium IS NULL
    """)
    return [row[0] for row in cursor.fetchall()]


def label_cascade_mistral(conn, workers=4):
    """Run mistral on samples without consensus."""
    # First compute consensus to identify which samples need more labeling
    compute_consensus(conn)

    ids = get_samples_needing_mistral(conn)
    if not ids:
        print("  No samples need mistral labeling")
        return

    cursor = conn.execute(
        f"SELECT id, name, files_json FROM samples WHERE id IN ({','.join(map(str, ids))})"
    )
    rows = cursor.fetchall()

    print(f"  Labeling {len(rows)} samples with mistral:7b (cascade)...")
    start = time.time()
    completed = 0

    def process(row):
        sid, name, files_json = row
        prompt = build_prompt(name, files_json)
        label, elapsed = classify_ollama(prompt, "mistral:7b")
        return sid, label, elapsed

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, row): row for row in rows}
        for future in as_completed(futures):
            sid, label, elapsed = future.result()

            for attempt in range(10):
                try:
                    conn.execute(
                        "UPDATE samples SET new_mistral = ?, new_mistral_time = ? WHERE id = ?",
                        (label, elapsed, sid)
                    )
                    conn.commit()
                    break
                except sqlite3.OperationalError:
                    time.sleep(0.1 * (attempt + 1))

            completed += 1
            if completed % 100 == 0:
                rate = completed / (time.time() - start)
                remaining = (len(rows) - completed) / rate if rate > 0 else 0
                print(f"    {completed}/{len(rows)} ({rate:.1f} req/s, ~{remaining/60:.1f}m left)")

    elapsed = time.time() - start
    print(f"    Done: {completed} samples in {elapsed:.1f}s ({completed/elapsed:.1f} req/s)")

    # Recompute consensus
    compute_consensus(conn)


def label_cascade_big(conn, workers=4):
    """Run big model on samples still without consensus."""
    # First compute consensus
    compute_consensus(conn)

    ids = get_samples_needing_big(conn)
    if not ids:
        print("  No samples need big model labeling")
        return

    cursor = conn.execute(
        f"SELECT id, name, files_json FROM samples WHERE id IN ({','.join(map(str, ids))})"
    )
    rows = cursor.fetchall()

    print(f"  Labeling {len(rows)} samples with qpt-oss:120b (cascade)...")
    start = time.time()
    completed = 0

    def process(row):
        sid, name, files_json = row
        prompt = build_prompt(name, files_json)
        label, elapsed = classify_openai(prompt)
        return sid, label, elapsed

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, row): row for row in rows}
        for future in as_completed(futures):
            sid, label, elapsed = future.result()

            for attempt in range(10):
                try:
                    conn.execute(
                        "UPDATE samples SET new_big = ?, new_big_time = ? WHERE id = ?",
                        (label, elapsed, sid)
                    )
                    conn.commit()
                    break
                except sqlite3.OperationalError:
                    time.sleep(0.1 * (attempt + 1))

            completed += 1
            if completed % 100 == 0:
                rate = completed / (time.time() - start)
                remaining = (len(rows) - completed) / rate if rate > 0 else 0
                print(f"    {completed}/{len(rows)} ({rate:.1f} req/s, ~{remaining/60:.1f}m left)")

    elapsed = time.time() - start
    print(f"    Done: {completed} samples in {elapsed:.1f}s ({completed/elapsed:.1f} req/s)")

    # Recompute consensus
    compute_consensus(conn)


def print_stats(conn):
    """Print labeling statistics."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM samples")
    total = cursor.fetchone()[0]

    if total == 0:
        print("No samples in database")
        return

    # Old labels stats
    cursor.execute("SELECT COUNT(*) FROM samples WHERE old_qwen IS NOT NULL")
    old_qwen = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM samples WHERE old_qwen IS NULL")
    old_porn = cursor.fetchone()[0]  # These were porn in old schema

    # New labels stats
    cursor.execute("SELECT COUNT(*) FROM samples WHERE new_qwen IS NOT NULL")
    new_qwen = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM samples WHERE new_gemma IS NOT NULL")
    new_gemma = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM samples WHERE new_mistral IS NOT NULL")
    new_mistral = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM samples WHERE new_big IS NOT NULL")
    new_big = cursor.fetchone()[0]

    # Consensus
    cursor.execute("SELECT COUNT(*) FROM samples WHERE medium IS NOT NULL")
    with_consensus = cursor.fetchone()[0]

    print(f"\n{'='*60}")
    print("MEDIUM LABELING STATS")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"\nOld labels (converted to 5-class):")
    print(f"  Usable (non-porn): {old_qwen}")
    print(f"  Was porn (needs re-label): {old_porn}")
    print(f"\nNew labels:")
    print(f"  new_qwen:    {new_qwen:>6} / {total}")
    print(f"  new_gemma:   {new_gemma:>6} / {total}")
    print(f"  new_mistral: {new_mistral:>6} (cascade)")
    print(f"  new_big:     {new_big:>6} (cascade)")
    print(f"\nConsensus:")
    print(f"  With consensus: {with_consensus:>6} ({100*with_consensus/total:.1f}%)")
    print(f"  Without:        {total - with_consensus:>6}")

    # Distribution
    print(f"\nMedium distribution:")
    cursor.execute("""
        SELECT medium, COUNT(*) FROM samples
        WHERE medium IS NOT NULL
        GROUP BY medium ORDER BY COUNT(*) DESC
    """)
    for cat, cnt in cursor.fetchall():
        print(f"  {cat}: {cnt}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Medium-type labeler (5-class)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", choices=[
        "small2", "qwen", "gemma", "gemma-halo", "mistral-cascade", "big-cascade"
    ], default="small2")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    conn = init_new_db()

    # Copy samples from old DB if needed
    copy_from_old_db(conn)

    if args.stats:
        print_stats(conn)
        conn.close()
        return

    if args.model == "small2":
        label_with_model(conn, "qwen2.5:3b", "new_qwen", args.workers, use_openai=False)
        label_with_model(conn, "gemma3:4b", "new_gemma", args.workers, use_openai=False)
        compute_consensus(conn)

    elif args.model == "qwen":
        label_with_model(conn, "qwen2.5:3b", "new_qwen", args.workers, use_openai=False)

    elif args.model == "gemma":
        label_with_model(conn, "gemma3:4b", "new_gemma", args.workers, use_openai=False)

    elif args.model == "gemma-halo":
        # Run gemma on Strix Halo via OpenAI API
        label_with_model(conn, "gemma-3-4b-it-Q4_K_M.gguf", "new_gemma", args.workers, use_openai=True)

    elif args.model == "mistral-cascade":
        label_cascade_mistral(conn, args.workers)

    elif args.model == "big-cascade":
        label_cascade_big(conn, args.workers)

    print_stats(conn)
    conn.close()
    print(f"\nDatabase saved to {NEW_DB}")


if __name__ == "__main__":
    main()
