#!/home/lelloman/lelloprojects/mimmo/.venv/bin/python3
"""TUI for searching the Magnetico DHT dump and viewing torrent structures."""

import sqlite3
import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Input, DataTable, Static, TextArea
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.binding import Binding

DB_PATH = Path.home() / "Downloads/dht-magnetico-dump/32.5M-BitTorrent-DHT-Dump-Magnetico-Maiti.info/database.sqlite3"


def human_size(size: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def build_tree(files: list[tuple[int, str]]) -> str:
    """Build a tree structure from file paths."""
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


class FileTreeScreen(ModalScreen):
    """Screen to display file tree for a torrent."""

    BINDINGS = [
        Binding("escape,q", "app.pop_screen", "Close"),
    ]

    def __init__(self, torrent_name: str, files: list[tuple[int, str]]):
        super().__init__()
        self.torrent_name = torrent_name
        self.files = files

    def compose(self) -> ComposeResult:
        tree_text = build_tree(self.files) if self.files else "(no files)"
        with Vertical():
            yield Static(f"📦 {self.torrent_name}", id="torrent-title")
            yield TextArea(tree_text, id="file-tree", read_only=True)

    def on_mount(self) -> None:
        self.query_one("#file-tree", TextArea).cursor_position = (0, 0)


class DHTSearchApp(App):
    """Main TUI app for DHT search."""

    CSS = """
    #search-container {
        height: 5;
        dock: bottom;
    }

    #input {
        width: 1fr;
    }

    #results-container {
        height: 1fr;
    }

    DataTable {
        height: 1fr;
    }

    #torrent-title {
        text-align: center;
        text-style: bold;
        margin: 1;
    }

    #file-tree {
        height: 1fr;
        border: solid $primary;
    }

    .scrolling {
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "app.pop_screen", "Close"),
    ]

    conn: sqlite3.Connection

    def __init__(self):
        super().__init__()
        if not DB_PATH.exists():
            print(f"Database not found: {DB_PATH}")
            sys.exit(1)
        self.conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        self.conn.text_factory = lambda x: x.decode('utf-8', errors='replace')

    def search_db(self, query: str, limit: int = 50) -> list[tuple]:
        """Search torrents using FTS5."""
        cursor = self.conn.execute("""
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

    def get_files(self, torrent_id: int) -> list[tuple]:
        """Get all files for a torrent."""
        cursor = self.conn.execute("""
            SELECT size, path FROM files
            WHERE torrent_id = ?
            ORDER BY path
        """, (torrent_id,))
        return cursor.fetchall()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="results-container"):
            yield DataTable(id="results-table")
        with Horizontal(id="search-container"):
            yield Input(placeholder="Search torrents (or enter ID to view files)...", id="input")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("ID", width=10)
        table.add_column("Size", width=10)
        table.add_column("Files", width=8)
        table.add_column("Name", width=100)
        table.cursor_type = "row"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()

        if not query:
            return

        if query.isdigit():
            torrent_id = int(query)
            self.show_torrent_files(torrent_id)
        else:
            self.perform_search(query)

        event.input.value = ""

    def perform_search(self, query: str) -> None:
        table = self.query_one(DataTable)
        table.clear()

        results = self.search_db(query)

        if not results:
            table.add_row("", "", "", "No results found")
            return

        for row in results:
            tid, name, size, discovered, file_count = row
            name = name.decode('utf-8', errors='replace') if isinstance(name, bytes) else name
            table.add_row(str(tid), human_size(size), str(file_count), name)

    def show_torrent_files(self, torrent_id: int) -> None:
        cursor = self.conn.execute("""
            SELECT id, name FROM torrents WHERE id = ?
        """, (torrent_id,))
        torrent = cursor.fetchone()

        if not torrent:
            table = self.query_one(DataTable)
            table.clear()
            table.add_row("", "", "", f"Torrent ID {torrent_id} not found")
            return

        files = self.get_files(torrent_id)
        name = torrent[1].decode('utf-8', errors='replace') if isinstance(torrent[1], bytes) else torrent[1]
        self.push_screen(FileTreeScreen(name, files))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one(DataTable)
        row_key = event.row_key
        if row_key is not None:
            row = table.get_row(row_key)
            if row and row[0].isdigit():
                self.show_torrent_files(int(row[0]))


def main():
    app = DHTSearchApp()
    app.run()


if __name__ == "__main__":
    main()
