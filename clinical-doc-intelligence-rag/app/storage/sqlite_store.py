from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.models.schemas import Chunk, EvidenceBlock


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init(self) -> None:
        with self._connect() as con:
            con.executescript('''
            CREATE TABLE IF NOT EXISTS documents(
                document_id TEXT PRIMARY KEY, filename TEXT, pages INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS blocks(
                block_id TEXT PRIMARY KEY, document_id TEXT, page_number INTEGER, block_type TEXT,
                text TEXT, bbox TEXT, confidence REAL
            );
            CREATE TABLE IF NOT EXISTS chunks(
                chunk_id TEXT PRIMARY KEY, document_id TEXT, page_numbers TEXT, text TEXT,
                source_blocks TEXT, source_type TEXT
            );
            ''')

    def save_document(self, document_id: str, filename: str, pages: int) -> None:
        with self._connect() as con:
            con.execute('INSERT OR REPLACE INTO documents(document_id, filename, pages) VALUES (?, ?, ?)', (document_id, filename, pages))

    def save_blocks(self, blocks: list[EvidenceBlock]) -> None:
        with self._connect() as con:
            con.executemany(
                'INSERT OR REPLACE INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?)',
                [(b.block_id, b.document_id, b.page_number, b.block_type, b.text, json.dumps(b.bbox), b.confidence) for b in blocks],
            )

    def save_chunks(self, chunks: list[Chunk]) -> None:
        with self._connect() as con:
            con.executemany(
                'INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?)',
                [(c.chunk_id, c.document_id, json.dumps(c.page_numbers), c.text, json.dumps(c.source_blocks), c.source_type) for c in chunks],
            )

    def get_chunks(self) -> list[Chunk]:
        with self._connect() as con:
            rows = con.execute('SELECT chunk_id, document_id, page_numbers, text, source_blocks, source_type FROM chunks').fetchall()
        return [Chunk(chunk_id=r[0], document_id=r[1], page_numbers=json.loads(r[2]), text=r[3], source_blocks=json.loads(r[4]), source_type=r[5]) for r in rows]

    def list_documents(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute('SELECT document_id, filename, pages, created_at FROM documents ORDER BY created_at DESC').fetchall()
        return [{'document_id': r[0], 'filename': r[1], 'pages': r[2], 'created_at': r[3]} for r in rows]
