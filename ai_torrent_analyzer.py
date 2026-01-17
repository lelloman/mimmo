#!/usr/bin/env python3
"""Select a torrent from Magnetico dump and analyze it with an AI endpoint."""

import sqlite3
import sys
import os
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional
import argparse
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse

DB_PATH = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"

DEFAULT_SYSTEM_PROMPT = """You are a music metadata extraction assistant. Given a torrent name and file listing, extract structured metadata.

Respond ONLY with valid JSON in this exact format:
{
  "type": "album" | "collection" | "track" | "audiobook" | "other",
  "artist": "Artist Name" | null,
  "album": "Album Title" | null,
  "year": 1234 | null,
  "tracks": [{"num": 1, "name": "Track Name"}, ...] | []
}

Rules:
- "album": Single album by one artist
- "collection": Discography, compilation, or multiple albums
- "track": Single song/file
- "audiobook": Spoken word, podcast, radio drama
- "other": Non-music or unclassifiable
- Extract artist/album/year from folder name or file patterns
- For tracks, extract track number and name from filenames
- If uncertain, use null"""

DEFAULT_TEMPLATE = """Torrent: {torrent_name}
Size: {torrent_size}
Files: {file_count}

Directory Tree:
{torrent_tree}

Extract the metadata as JSON:"""

DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"


def human_size(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def build_tree(files: list[tuple[int, str]]) -> str:
    tree = {}
    for size, path in files:
        parts = path.split('/')
        current = tree
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = size

    lines = []
    def render(node, prefix="", is_last=True):
        items = sorted(node.items(), key=lambda x: (isinstance(x[1], dict), x[0].lower()))
        for i, (name, value) in enumerate(items):
            is_last_item = i == len(items) - 1
            connector = "└── " if is_last_item else "├── "

            if isinstance(value, dict):
                lines.append(f"{prefix}{connector}{name}/")
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                render(value, new_prefix, is_last_item)
            else:
                lines.append(f"{prefix}{connector}{name}  [{human_size(value)}]")

    render(tree)
    return "\n".join(lines)


def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[tuple]:
    cursor = conn.execute("""
        SELECT t.id, t.name, t.total_size,
               datetime(t.discovered_on, 'unixepoch') as discovered,
               (SELECT COUNT(*) FROM files WHERE torrent_id = t.id) as file_count
        FROM torrents t
        JOIN torrents_idx idx ON t.id = idx.rowid
        WHERE torrents_idx MATCH ?
        ORDER BY t.total_size DESC
        LIMIT ?
    """, (query, limit))
    return cursor.fetchall()


def get_files(conn: sqlite3.Connection, torrent_id: int) -> list[tuple]:
    cursor = conn.execute("""
        SELECT size, path FROM files
        WHERE torrent_id = ?
        ORDER BY path
    """, (torrent_id,))
    return cursor.fetchall()


def get_torrent(conn: sqlite3.Connection, torrent_id: int) -> Optional[tuple]:
    cursor = conn.execute("""
        SELECT id, name, total_size, datetime(discovered_on, 'unixepoch')
        FROM torrents WHERE id = ?
    """, (torrent_id,))
    return cursor.fetchone()


def get_random_torrent(conn: sqlite3.Connection) -> Optional[tuple]:
    cursor = conn.execute("""
        SELECT id FROM torrents
        ORDER BY RANDOM()
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result:
        torrent_id = result[0]
        return get_torrent(conn, torrent_id)
    return None


def get_max_torrent_id(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("SELECT MAX(id) FROM torrents")
    result = cursor.fetchone()
    return result[0] if result and result[0] else 0


def load_template(template_path: Optional[Path]) -> str:
    if template_path is None:
        return DEFAULT_TEMPLATE

    if not template_path.exists():
        if DEFAULT_PROMPTS_DIR.exists():
            fallback_path = DEFAULT_PROMPTS_DIR / template_path.name
            if fallback_path.exists():
                template_path = fallback_path
            else:
                print(f"Warning: Template not found at {template_path} or {fallback_path}, using default")
                return DEFAULT_TEMPLATE
        else:
            print(f"Warning: Template not found at {template_path}, using default")
            return DEFAULT_TEMPLATE

    return template_path.read_text(encoding='utf-8')


def build_prompt(template: str, torrent: tuple, tree: str, file_count: int) -> str:
    name = torrent[1]
    size = human_size(torrent[2])

    placeholders = {
        'torrent_name': name,
        'torrent_size': size,
        'torrent_tree': tree,
        'file_count': file_count,
        'torrent_id': torrent[0],
        'torrent_discovered': torrent[3]
    }

    return template.format(**placeholders)


def query_ollama(base_url: str, model: str, prompt: str, system: str = None, force_json: bool = True) -> tuple[str, dict]:
    url = f"{base_url.rstrip('/')}/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    if system:
        data["system"] = system
    if force_json:
        data["format"] = "json"
    body = json.dumps(data).encode('utf-8')
    headers = {'Content-Type': 'application/json'}

    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        stats = {
            'prompt_tokens': result.get('prompt_eval_count', 0),
            'completion_tokens': result.get('eval_count', 0),
            'total_duration_ns': result.get('total_duration', 0),
            'eval_duration_ns': result.get('eval_duration', 0),
            'load_duration_ns': result.get('load_duration', 0)
        }
        # Qwen3 models put output in "thinking" field, others use "response"
        text = result.get('response', '') or result.get('thinking', '')
        return text, stats


def query_openai(base_url: str, model: str, api_key: str, prompt: str, system: str = None) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    body = json.dumps(data).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        return result['choices'][0]['message']['content']


def interactive_search(conn: sqlite3.Connection) -> Optional[int]:
    print("\n🔍 DHT Torrent Search (type 'q' to quit)\n")

    while True:
        try:
            query = input("Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return None

        if not query or query.lower() == 'q':
            return None

        if query.isdigit():
            torrent_id = int(query)
            torrent = get_torrent(conn, torrent_id)
            if torrent:
                print(f"\n📦 {torrent[1]}")
                print(f"   Size: {human_size(torrent[2])} | Discovered: {torrent[3]}")
                print(f"   ID: {torrent[0]}\n")
                return torrent[0]
            else:
                print(f"Torrent ID {torrent_id} not found\n")
            continue

        results = search(conn, query)

        if not results:
            print("No results found.\n")
            continue

        print(f"\n{'ID':<10} {'Size':<10} {'Files':<8} Name")
        print("-" * 80)
        for row in results:
            tid, name, size, discovered, file_count = row
            name = name.decode('utf-8', errors='replace') if isinstance(name, bytes) else name
            display_name = name[:55] + "..." if len(name) > 58 else name
            print(f"{tid:<10} {human_size(size):<10} {file_count:<8} {display_name}")
        print(f"\nEnter ID to select, or search again:\n")


def main():
    parser = argparse.ArgumentParser(
        description="Select a torrent from Magnetico dump and analyze it with an AI endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --random
  %(prog)s --search --template classify.txt
  %(prog)s --id 12345 --prompt "Analyze this structure"
  %(prog)s --random --preview-only
        """
    )
    parser.add_argument('--db', type=Path, default=DB_PATH, help='Path to Magnetico database')
    parser.add_argument('--random', action='store_true', help='Select a random torrent')
    parser.add_argument('--id', type=int, help='Select torrent by ID')
    parser.add_argument('--search', action='store_true', help='Interactive search mode')
    parser.add_argument('--template', type=Path, help='Path to prompt template file')
    parser.add_argument('--prompt', type=str, help='Custom prompt text (overrides template)')
    parser.add_argument('--preview-only', action='store_true', help='Only show the prepared prompt, no AI query')
    parser.add_argument('--output', type=Path, help='Save AI response to file')
    
    ai_group = parser.add_argument_group('AI Configuration')
    ai_group.add_argument('--ai-type', choices=['ollama', 'openai'], help='AI endpoint type (env: AI_ENDPOINT_TYPE)')
    ai_group.add_argument('--ai-url', help='AI endpoint URL (env: AI_BASE_URL)')
    ai_group.add_argument('--ai-model', help='AI model name (env: AI_MODEL)')
    ai_group.add_argument('--ai-key', help='API key for OpenAI (env: AI_API_KEY)')

    args = parser.parse_args()

    if not any([args.random, args.id, args.search]):
        parser.print_help()
        sys.exit(1)

    if not args.db.exists():
        print(f"Error: Database not found at {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.text_factory = lambda x: x.decode('utf-8', errors='replace')

    try:
        torrent = None
        torrent_id = None

        if args.id:
            torrent = get_torrent(conn, args.id)
            if not torrent:
                print(f"Error: Torrent ID {args.id} not found")
                sys.exit(1)
            torrent_id = args.id

        elif args.random:
            torrent = get_random_torrent(conn)
            if not torrent:
                print("Error: Could not find any torrent in database")
                sys.exit(1)
            torrent_id = torrent[0]
            print(f"Selected random torrent ID: {torrent_id}")

        elif args.search:
            torrent_id = interactive_search(conn)
            if torrent_id is None:
                print("No torrent selected")
                sys.exit(0)
            torrent = get_torrent(conn, torrent_id)

        files = get_files(conn, torrent_id)
        file_count = len(files)

        print(f"\n{'='*60}")
        print(f"📦 {torrent[1]}")
        print(f"   Size: {human_size(torrent[2])} | Files: {file_count}")
        print(f"{'='*60}\n")

        if files:
            tree = build_tree(files)
            print(tree)
        else:
            tree = "(no files)"
            print(tree)
        print()

        if args.prompt:
            prompt_text = args.prompt
        else:
            template = load_template(args.template)
            prompt_text = build_prompt(template, torrent, tree, file_count)

        print("\n" + "="*60)
        print("SYSTEM PROMPT:")
        print("="*60)
        print(DEFAULT_SYSTEM_PROMPT)
        print("="*60)
        print("\nUSER PROMPT:")
        print("="*60)
        print(prompt_text)
        print("="*60 + "\n")

        if args.preview_only:
            print("(Preview mode - no AI query)")
            sys.exit(0)

        ai_type = args.ai_type or os.environ.get('AI_ENDPOINT_TYPE', 'ollama')
        ai_url = args.ai_url or os.environ.get('AI_BASE_URL',
                                              'http://192.168.1.92:11434' if ai_type == 'ollama' else 'https://api.openai.com')
        ai_model = args.ai_model or os.environ.get('AI_MODEL', 
                                                   'llama3.2' if ai_type == 'ollama' else 'gpt-4o-mini')
        ai_key = args.ai_key or os.environ.get('AI_API_KEY', '')

        if ai_type == 'openai' and not ai_key:
            print("Error: API key required for OpenAI endpoint (use --ai-key or AI_API_KEY env var)")
            sys.exit(1)

        print(f"Querying {ai_type} endpoint: {ai_url}")
        print(f"Model: {ai_model}\n")

        try:
            if ai_type == 'ollama':
                response, stats = query_ollama(ai_url, ai_model, prompt_text, DEFAULT_SYSTEM_PROMPT)
            else:
                response = query_openai(ai_url, ai_model, ai_key, prompt_text, DEFAULT_SYSTEM_PROMPT)
                stats = None

            print("="*60)
            print("AI RESPONSE:")
            print("="*60)
            print(response)
            print("="*60)

            if stats:
                print(f"\n📊 Statistics:")
                print(f"   Prompt tokens: {stats['prompt_tokens']:,}")
                print(f"   Completion tokens: {stats['completion_tokens']:,}")
                print(f"   Total time: {stats['total_duration_ns'] / 1e9:.2f}s")
                print(f"   Generation time: {stats['eval_duration_ns'] / 1e9:.2f}s")
                print(f"   Load time: {stats['load_duration_ns'] / 1e9:.2f}s")

            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(response, encoding='utf-8')
                print(f"\nResponse saved to {args.output}")

        except urllib.error.URLError as e:
            print(f"Error connecting to AI endpoint: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error querying AI: {e}")
            sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
