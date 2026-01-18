#!/usr/bin/env python3
"""Multi-LLM consensus labeler for BERT training data.

Strategy:
- Run 4 LLMs on each sample:
  - 3 small models on RTX (Ollama): qwen2.5:3b, gemma3:4b, mistral:7b
  - 1 big model on Strix Halo (OpenAI API): qwen3-coder:30b
- Use consensus voting:
  - All 4 agree = high confidence (~60%)
  - 3v1 majority = usable with majority vote (~27%)
  - Total usable: ~87%

Performance (from 1000-sample benchmark):
- RTX Ollama: ~31 req/s with 4 workers
- Strix Halo OpenAI: ~4 req/s
- 50k samples: ~80min for small models, ~3.5h for big model
- Run small models in parallel with big model for efficiency

Usage:
    # Full run: sample 50k torrents and label with all 4 models
    python consensus_labeler.py -n 50000

    # Run only small models (RTX)
    python consensus_labeler.py --skip-sampling --model small3

    # Run only big model (Strix Halo)
    python consensus_labeler.py --skip-sampling --model qwen3coder

    # Export training data
    python consensus_labeler.py --export
"""

import argparse
import json
import random
import sqlite3
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"
OUTPUT_DB = Path(__file__).parent.parent / "data" / "consensus_labels.db"

# Endpoints
RTX_OLLAMA_URL = "http://192.168.1.92:11434"      # RTX 4090 - small models
STRIX_HALO_URL = "http://192.168.1.102:8080"      # Strix Halo - big model (OpenAI API)

# Small models (RTX via Ollama)
SMALL_MODELS = [
    ("qwen2.5:3b", "qwen"),
    ("gemma3:4b", "gemma"),
    ("mistral:7b", "mistral"),
]

# Big model (Strix Halo via OpenAI API)
BIG_MODEL = ("qwen3-coder:30b", "qwen3coder")

# All models for iteration
ALL_MODELS = SMALL_MODELS + [BIG_MODEL]

CATEGORIES = ["music", "video", "software", "book", "porn", "other"]

# Response normalization: map common LLM outputs to valid categories
NORMALIZE_MAP = {
    # Video aliases
    "movie": "video", "movies": "video", "film": "video", "films": "video",
    "tv": "video", "tvshow": "video", "tv show": "video", "tv-show": "video",
    "television": "video", "television_show": "video",
    "anime": "video", "documentary": "video", "series": "video", "drama": "video",
    "sports": "video",
    # Book aliases
    "comic": "book", "comics": "book", "manga": "book",
    "ebook": "book", "e-book": "book", "audiobook": "book",
    # Software aliases
    "game": "software", "games": "software",
    "application": "software", "app": "software",
    # Porn aliases
    "adult": "porn", "xxx": "porn", "hentai": "porn", "nsfw": "porn",
    # Music aliases
    "audio": "music", "soundtrack": "music", "album": "music",
}

# =============================================================================
# UTILITIES
# =============================================================================

def human_size(n):
    """Convert bytes to human-readable size."""
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.1f}{u}'
        n /= 1024
    return f'{n:.1f}PB'


def ensure_str(s):
    """Ensure value is a string (handle bytes from SQLite)."""
    return s.decode('utf-8', errors='replace') if isinstance(s, bytes) else s


def normalize_response(response):
    """Normalize LLM response to valid category."""
    response = response.strip().lower()

    # Direct match
    if response in CATEGORIES:
        return response

    # Check normalization map
    for key, val in NORMALIZE_MAP.items():
        if key in response:
            return val

    # Check if any category is mentioned
    for cat in CATEGORIES:
        if cat in response:
            return cat

    return None  # Invalid response


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

    return f"""Classify this torrent. Reply with exactly one word: music, video, software, book, porn, or other

Note: "video" includes movies, TV shows, anime, documentaries. "book" includes comics/manga.

Torrent: {name}
Top files:
{files_str}

Category:"""


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


def classify_openai(prompt, timeout=60):
    """Classify using OpenAI API (Strix Halo - qwen3-coder)."""
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

def init_output_db():
    """Initialize output database with WAL mode for better concurrency."""
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUTPUT_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            files_json TEXT,
            total_size INTEGER,
            qwen TEXT,
            qwen_time REAL,
            gemma TEXT,
            gemma_time REAL,
            mistral TEXT,
            mistral_time REAL,
            qwen3coder TEXT,
            qwen3coder_time REAL,
            consensus TEXT,
            majority TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Add qwen3coder columns if they don't exist (for migration)
    try:
        conn.execute("ALTER TABLE samples ADD COLUMN qwen3coder TEXT")
        conn.execute("ALTER TABLE samples ADD COLUMN qwen3coder_time REAL")
    except sqlite3.OperationalError:
        pass  # Columns already exist
    conn.execute("CREATE INDEX IF NOT EXISTS idx_consensus ON samples(consensus)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_majority ON samples(majority)")
    conn.commit()
    return conn


def get_existing_ids(conn):
    """Get IDs already processed."""
    cursor = conn.execute("SELECT id FROM samples")
    return set(row[0] for row in cursor.fetchall())


def sample_torrents(n, existing_ids, seed=42):
    """Sample n random torrents not already processed."""
    print(f"Sampling {n} torrents from DHT database...")
    random.seed(seed)

    src_conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    src_conn.text_factory = bytes

    cursor = src_conn.execute("SELECT MAX(id) FROM torrents")
    max_id = cursor.fetchone()[0]

    torrents = []
    attempts = 0
    max_attempts = n * 10

    while len(torrents) < n and attempts < max_attempts:
        tid = random.randint(1, max_id)
        attempts += 1

        if tid in existing_ids:
            continue

        cursor = src_conn.execute(
            "SELECT id, name, total_size FROM torrents WHERE id = ?", (tid,)
        )
        row = cursor.fetchone()
        if not row:
            continue

        tid, name, total_size = row
        cursor = src_conn.execute(
            "SELECT size, path FROM files WHERE torrent_id = ?", (tid,)
        )
        files = cursor.fetchall()

        if not files:
            continue

        name = ensure_str(name)
        files_list = [[size, ensure_str(path)] for size, path in files[:10]]

        existing_ids.add(tid)
        torrents.append((tid, name, json.dumps(files_list), total_size))

        if len(torrents) % 1000 == 0:
            print(f"  {len(torrents)}/{n}")

    src_conn.close()
    return torrents


def insert_samples(conn, torrents):
    """Insert sampled torrents into output database."""
    cursor = conn.cursor()
    for tid, name, files_json, total_size in torrents:
        cursor.execute(
            "INSERT INTO samples (id, name, files_json, total_size) VALUES (?, ?, ?, ?)",
            (tid, name, files_json, total_size)
        )
    conn.commit()
    print(f"Inserted {len(torrents)} samples")


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
            label, elapsed = classify_openai(prompt)
        else:
            label, elapsed = classify_ollama(prompt, model_name)
        return sid, label, elapsed

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, row): row for row in rows}
        for future in as_completed(futures):
            sid, label, elapsed = future.result()

            # Retry on SQLite lock
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


def label_disagreements_with_qwen3coder(conn, workers=4, use_openai=True, batch="all"):
    """Label only samples where 3 small models disagree.

    Args:
        use_openai: True for Strix Halo (OpenAI API), False for RTX (Ollama)
        batch: "odd", "even", or "all" - allows splitting work between machines
    """
    # Get samples where all 3 small models have valid labels but don't all agree
    base_query = """
        SELECT id, name, files_json FROM samples
        WHERE qwen IS NOT NULL AND gemma IS NOT NULL AND mistral IS NOT NULL
        AND qwen IN ('music','video','software','book','porn','other')
        AND gemma IN ('music','video','software','book','porn','other')
        AND mistral IN ('music','video','software','book','porn','other')
        AND qwen3coder IS NULL
        AND NOT (qwen = gemma AND gemma = mistral)
    """
    if batch == "odd":
        base_query += " AND id % 2 = 1"
    elif batch == "even":
        base_query += " AND id % 2 = 0"

    cursor = conn.execute(base_query)
    rows = cursor.fetchall()

    if not rows:
        print(f"  No disagreements to label with qwen3-coder")
        return

    api_type = "OpenAI (Strix Halo)" if use_openai else "Ollama (RTX)"
    model_name = "qwen3-coder:30b"
    print(f"  Labeling {len(rows)} disagreements ({batch}) with {model_name} via {api_type} (workers={workers})...")
    start = time.time()
    completed = 0

    def process(row):
        sid, name, files_json = row
        prompt = build_prompt(name, files_json)
        if use_openai:
            label, elapsed = classify_openai(prompt)
        else:
            label, elapsed = classify_ollama(prompt, "qwen3-coder:30b")
        return sid, label, elapsed

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, row): row for row in rows}
        for future in as_completed(futures):
            sid, label, elapsed = future.result()

            # Retry on SQLite lock
            for attempt in range(10):
                try:
                    conn.execute(
                        "UPDATE samples SET qwen3coder = ?, qwen3coder_time = ? WHERE id = ?",
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


def label_disagreements_dual(conn, rtx_workers=4, halo_workers=4):
    """Label disagreements using both RTX and Strix Halo in parallel.

    Main thread handles DB, workers only do HTTP requests.
    """
    # Get samples where all 3 small models have valid labels but don't all agree
    cursor = conn.execute("""
        SELECT id, name, files_json FROM samples
        WHERE qwen IS NOT NULL AND gemma IS NOT NULL AND mistral IS NOT NULL
        AND qwen IN ('music','video','software','book','porn','other')
        AND gemma IN ('music','video','software','book','porn','other')
        AND mistral IN ('music','video','software','book','porn','other')
        AND qwen3coder IS NULL
        AND NOT (qwen = gemma AND gemma = mistral)
    """)
    rows = list(cursor.fetchall())

    if not rows:
        print(f"  No disagreements to label")
        return

    total = len(rows)
    print(f"  Labeling {total} disagreements with qwen3-coder:30b via RTX ({rtx_workers}w) + Strix Halo ({halo_workers}w)...")

    start = time.time()
    completed = 0
    rtx_count = 0
    halo_count = 0

    def process_rtx(row):
        sid, name, files_json = row
        prompt = build_prompt(name, files_json)
        label, elapsed = classify_ollama(prompt, "qwen3-coder:30b")
        return sid, label, elapsed, "rtx"

    def process_halo(row):
        sid, name, files_json = row
        prompt = build_prompt(name, files_json)
        label, elapsed = classify_openai(prompt)
        return sid, label, elapsed, "halo"

    # Use two thread pools, main thread collects results
    with ThreadPoolExecutor(max_workers=rtx_workers) as rtx_pool, \
         ThreadPoolExecutor(max_workers=halo_workers) as halo_pool:

        # Submit all work - alternate between pools to balance initially
        futures = []
        for i, row in enumerate(rows):
            if i % 2 == 0:
                futures.append(rtx_pool.submit(process_rtx, row))
            else:
                futures.append(halo_pool.submit(process_halo, row))

        # Main thread collects results and writes to DB
        for future in as_completed(futures):
            sid, label, elapsed, source = future.result()

            # Save to DB (main thread only)
            for attempt in range(10):
                try:
                    conn.execute(
                        "UPDATE samples SET qwen3coder = ?, qwen3coder_time = ? WHERE id = ?",
                        (label, elapsed, sid)
                    )
                    conn.commit()
                    break
                except sqlite3.OperationalError:
                    time.sleep(0.1 * (attempt + 1))

            completed += 1
            if source == "rtx":
                rtx_count += 1
            else:
                halo_count += 1

            if completed % 100 == 0:
                rate = completed / (time.time() - start)
                remaining = (total - completed) / rate if rate > 0 else 0
                print(f"    {completed}/{total} ({rate:.1f} req/s, ~{remaining/60:.1f}m left) [RTX:{rtx_count} Halo:{halo_count}]")

    elapsed = time.time() - start
    print(f"    Done: {completed} samples in {elapsed:.1f}s ({completed/elapsed:.1f} req/s)")
    print(f"    RTX: {rtx_count}, Strix Halo: {halo_count}")


def compute_consensus(conn):
    """Compute consensus and majority vote labels.

    Handles two cases:
    1. All 3 small models agree -> consensus without needing qwen3coder
    2. Disagreement with qwen3coder -> use majority voting
    """
    from collections import Counter
    cursor = conn.cursor()

    # Get all samples with at least 3 small model labels
    cursor.execute("""
        SELECT id, qwen, gemma, mistral, qwen3coder FROM samples
        WHERE qwen IS NOT NULL AND gemma IS NOT NULL AND mistral IS NOT NULL
    """)

    for row in cursor.fetchall():
        sid, qwen, gemma, mistral, qwen3coder = row

        # Filter valid small model labels
        small_labels = [l for l in [qwen, gemma, mistral] if l in CATEGORIES]

        consensus = None
        majority = None

        if len(small_labels) == 3:
            if small_labels[0] == small_labels[1] == small_labels[2]:
                # All 3 small models agree - consensus!
                consensus = small_labels[0]
                majority = small_labels[0]
            elif qwen3coder and qwen3coder in CATEGORIES:
                # Disagreement among small models, use qwen3coder to break tie
                all_labels = small_labels + [qwen3coder]
                counts = Counter(all_labels)
                most_common = counts.most_common(1)[0]
                if most_common[1] >= 3:
                    # 3v1 split - use majority
                    majority = most_common[0]
                elif most_common[1] == 2:
                    # Check for 2v2 vs 2v1v1
                    top_two = counts.most_common(2)
                    if len(top_two) >= 2 and top_two[1][1] == 2:
                        # 2v2 split - no clear majority
                        majority = None
                    else:
                        # 2v1v1 - use the 2
                        majority = most_common[0]

        conn.execute(
            "UPDATE samples SET consensus = ?, majority = ? WHERE id = ?",
            (consensus, majority, sid)
        )

    conn.commit()


def print_stats(conn):
    """Print labeling statistics."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM samples")
    total = cursor.fetchone()[0]

    if total == 0:
        print("No samples in database")
        return

    # Count samples with all 4 labels
    cursor.execute("""
        SELECT COUNT(*) FROM samples
        WHERE qwen IS NOT NULL AND gemma IS NOT NULL
          AND mistral IS NOT NULL AND qwen3coder IS NOT NULL
    """)
    fully_labeled = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM samples WHERE consensus IS NOT NULL")
    consensus_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM samples WHERE majority IS NOT NULL")
    majority_count = cursor.fetchone()[0]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Fully labeled (4 models): {fully_labeled}")
    if fully_labeled > 0:
        print(f"All 4 agree (consensus): {consensus_count} ({consensus_count/fully_labeled*100:.1f}%)")
        print(f"3+ agree (majority):     {majority_count} ({majority_count/fully_labeled*100:.1f}%)")
        print(f"Disagreements (2v2):     {fully_labeled - majority_count} ({(fully_labeled-majority_count)/fully_labeled*100:.1f}%)")

    print(f"\nConsensus category distribution:")
    cursor.execute("""
        SELECT consensus, COUNT(*) FROM samples
        WHERE consensus IS NOT NULL
        GROUP BY consensus ORDER BY COUNT(*) DESC
    """)
    for cat, cnt in cursor.fetchall():
        print(f"  {cat}: {cnt}")

    print(f"\nMajority category distribution:")
    cursor.execute("""
        SELECT majority, COUNT(*) FROM samples
        WHERE majority IS NOT NULL
        GROUP BY majority ORDER BY COUNT(*) DESC
    """)
    for cat, cnt in cursor.fetchall():
        print(f"  {cat}: {cnt}")

    # Per-model stats
    print(f"\nPer-model category distribution:")
    for _, col in ALL_MODELS:
        cursor.execute(f"SELECT {col}, COUNT(*) FROM samples WHERE {col} IS NOT NULL GROUP BY {col}")
        dist = {r[0]: r[1] for r in cursor.fetchall()}
        if dist:
            summary = " ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
            print(f"  {col:12}: {summary}")


def export_training_data(conn, output_file, use_majority=True):
    """Export labeled data for BERT training.

    Combines torrent name with top 3 biggest files for richer context.
    """
    label_col = "majority" if use_majority else "consensus"

    cursor = conn.execute(f"""
        SELECT name, files_json, {label_col} FROM samples
        WHERE {label_col} IS NOT NULL
    """)

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for name, files_json, label in cursor:
            # Parse files and get top 3 biggest
            try:
                files = json.loads(files_json) if files_json else []
            except json.JSONDecodeError:
                files = []

            # Build text: name + top 3 file names
            # files format: [[size, filename], [size, filename], ...]
            if files:
                top_file_names = [f[1] for f in files[:3]]
                text = f"{name} | {' | '.join(top_file_names)}"
            else:
                text = name

            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + '\n')
            count += 1

    print(f"Exported {count} samples to {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-LLM consensus labeler")
    parser.add_argument("-n", "--samples", type=int, default=50000,
                       help="Number of samples to label")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--workers", type=int, default=4,
                       help="Parallel workers for LLM requests (4 optimal for RTX)")
    parser.add_argument("--skip-sampling", action="store_true",
                       help="Skip sampling, continue from existing database")
    parser.add_argument("--model", choices=[
        "qwen", "gemma", "mistral", "qwen3coder", "small3", "all", "gemma-halo",
        "qwen3coder-disagree", "qwen3coder-disagree-rtx", "qwen3coder-disagree-dual",
        "qwen3coder-disagree-halo-odd", "qwen3coder-disagree-halo-even",
        "qwen3coder-disagree-rtx-odd", "qwen3coder-disagree-rtx-even"
    ], default="all", help="Which model(s) to run")
    parser.add_argument("--export", action="store_true",
                       help="Export training data after labeling")
    parser.add_argument("--stats", action="store_true",
                       help="Just print stats and exit")
    args = parser.parse_args()

    conn = init_output_db()

    if args.stats:
        print_stats(conn)
        conn.close()
        return

    if args.export:
        # Export only - don't run any labeling
        compute_consensus(conn)
        export_file = OUTPUT_DB.parent / "training_data_consensus.jsonl"
        export_training_data(conn, export_file, use_majority=True)
        conn.close()
        return

    # Step 1: Sample torrents
    if not args.skip_sampling:
        if not DB_PATH.exists():
            print(f"Source database not found: {DB_PATH}")
            return

        existing_ids = get_existing_ids(conn)
        remaining = args.samples - len(existing_ids)

        if remaining > 0:
            torrents = sample_torrents(remaining, existing_ids, args.seed + len(existing_ids))
            insert_samples(conn, torrents)
        else:
            print(f"Target sample count already reached ({len(existing_ids)} samples)")

    # Step 2: Label with models
    print(f"\nLabeling samples...")

    if args.model == "all":
        # Run all 4 models
        for model_name, col_name in SMALL_MODELS:
            label_with_model(conn, model_name, col_name, args.workers, use_openai=False)
        model_name, col_name = BIG_MODEL
        label_with_model(conn, model_name, col_name, args.workers, use_openai=True)

    elif args.model == "small3":
        # Run only 3 small models (RTX)
        for model_name, col_name in SMALL_MODELS:
            label_with_model(conn, model_name, col_name, args.workers, use_openai=False)

    elif args.model == "qwen3coder":
        # Run only big model (Strix Halo)
        model_name, col_name = BIG_MODEL
        label_with_model(conn, model_name, col_name, args.workers, use_openai=True)

    elif args.model == "qwen3coder-disagree":
        # Run big model only on samples where 3 small models disagree (Strix Halo)
        label_disagreements_with_qwen3coder(conn, args.workers, use_openai=True, batch="all")

    elif args.model == "qwen3coder-disagree-rtx":
        # Run big model only on samples where 3 small models disagree (RTX Ollama)
        label_disagreements_with_qwen3coder(conn, args.workers, use_openai=False, batch="all")

    elif args.model == "qwen3coder-disagree-dual":
        # Run big model on both RTX and Strix Halo in parallel
        label_disagreements_dual(conn, rtx_workers=4, halo_workers=4)

    elif args.model == "qwen3coder-disagree-halo-odd":
        # Run on Strix Halo for odd IDs only
        label_disagreements_with_qwen3coder(conn, args.workers, use_openai=True, batch="odd")

    elif args.model == "qwen3coder-disagree-halo-even":
        # Run on Strix Halo for even IDs only
        label_disagreements_with_qwen3coder(conn, args.workers, use_openai=True, batch="even")

    elif args.model == "qwen3coder-disagree-rtx-odd":
        # Run on RTX for odd IDs only (qwen2.5-coder:32b via Ollama)
        label_disagreements_with_qwen3coder(conn, args.workers, use_openai=False, batch="odd")

    elif args.model == "qwen3coder-disagree-rtx-even":
        # Run on RTX for even IDs only (qwen2.5-coder:32b via Ollama)
        label_disagreements_with_qwen3coder(conn, args.workers, use_openai=False, batch="even")

    elif args.model == "gemma-halo":
        # Run gemma on Strix Halo via OpenAI API
        label_with_model(conn, "gemma-3-4b-it", "gemma", args.workers, use_openai=True)

    else:
        # Run single small model
        model_map = {col: (model, col) for model, col in SMALL_MODELS}
        model_name, col_name = model_map[args.model]
        label_with_model(conn, model_name, col_name, args.workers, use_openai=False)

    # Step 3: Compute consensus
    print("\nComputing consensus labels...")
    compute_consensus(conn)

    # Step 4: Print stats
    print_stats(conn)

    # Step 5: Export if requested
    if args.export:
        export_file = OUTPUT_DB.parent / "training_data_consensus.jsonl"
        export_training_data(conn, export_file, use_majority=True)

    conn.close()
    print(f"\nDatabase saved to {OUTPUT_DB}")


if __name__ == "__main__":
    main()
