# src/skill_cache/sqlite_store.py
"""
SQLite storage for skill cache with deterministic hashing.

Provides:
- SQLiteStore.compute_problem_sha(text) to generate normalized SHA used across build/query.
- insert_skill/get_by_problem_sha/iter_all utilities.
"""

import sqlite3
import json
import hashlib
import os
import re
from typing import Optional, Dict, Any, Iterator

def _normalize_text_for_hash(s: Optional[str]) -> str:
    """Normalize text for deterministic hashing: strip, collapse whitespace, lowercase."""
    if not s:
        return ""
    s2 = re.sub(r"\s+", " ", s.strip())
    return s2.lower()

class SQLiteStore:
    """SQLite backend for skill cache."""

    def __init__(self, db_path: str):
        if not db_path:
            raise ValueError("SQLiteStore requires a non-empty db_path")
        self.db_path = str(db_path)
        parent = os.path.dirname(self.db_path) or "."
        os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        cur = self._conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS skill_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_sha TEXT NOT NULL UNIQUE,
            topic TEXT,
            problem_text TEXT,
            composed_text TEXT,
            simhash INTEGER,
            solution_json TEXT,
            constraints_json TEXT,
            provenance_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON skill_entries(topic)")
        self._conn.commit()

    @staticmethod
    def compute_problem_sha(text: Optional[str]) -> str:
        """Public helper to compute the SHA used by this store (normalized)."""
        normalized = _normalize_text_for_hash(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def insert_skill(
        self,
        topic: str,
        problem_text: str,
        composed_text: str,
        simhash: Optional[int],
        solution: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert or replace a skill entry. Returns the id."""
        base = problem_text if (problem_text and str(problem_text).strip()) else composed_text
        problem_sha = self.compute_problem_sha(base)
        cur = self._conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO skill_entries
            (problem_sha, topic, problem_text, composed_text, simhash, solution_json, constraints_json, provenance_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            problem_sha,
            topic,
            problem_text,
            composed_text,
            int(simhash) if simhash is not None else None,
            json.dumps(solution, ensure_ascii=False),
            json.dumps(constraints or {}, ensure_ascii=False),
            json.dumps(provenance or {}, ensure_ascii=False),
        ))
        self._conn.commit()
        row = cur.execute("SELECT id FROM skill_entries WHERE problem_sha = ?", (problem_sha,)).fetchone()
        return int(row["id"]) if row else cur.lastrowid or 0

    def get_by_problem_sha(self, problem_sha: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        row = cur.execute("SELECT * FROM skill_entries WHERE problem_sha = ? LIMIT 1", (problem_sha,)).fetchone()
        if not row:
            return None
        d = dict(row)
        # Decode JSON columns
        for k in ("solution_json", "constraints_json", "provenance_json"):
            if k in d and d[k] is not None:
                try:
                    d[k.replace("_json","")] = json.loads(d[k])
                except Exception:
                    d[k.replace("_json","")] = d[k]
            else:
                d[k.replace("_json","")] = None
        return d

    def iter_all(self) -> Iterator[Dict[str, Any]]:
        cur = self._conn.cursor()
        for row in cur.execute("SELECT * FROM skill_entries"):
            d = dict(row)
            try:
                d["solution"] = json.loads(d.get("solution_json") or "{}")
            except Exception:
                d["solution"] = d.get("solution_json")
            yield d

    def count(self) -> int:
        cur = self._conn.cursor()
        r = cur.execute("SELECT COUNT(*) FROM skill_entries").fetchone()
        return int(r[0])

    def close(self):
        try:
            self._conn.commit()
            self._conn.close()
        except Exception:
            pass