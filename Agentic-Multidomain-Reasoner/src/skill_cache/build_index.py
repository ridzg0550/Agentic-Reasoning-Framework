# src/skill_cache/build_index.py
"""
Build the skill cache: ingest train.csv, normalize text, compute embedding vectors,
write SQLite rows, create FAISS index and save id->sqlite_id mapping.

Usage:
    python -m src.skill_cache.build_index /path/to/train.csv /path/to/cache_dir
"""

import os
import sys
import csv
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .sqlite_store import SQLiteStore

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_FILENAME = "skills.index"
ID_MAP_FILENAME = "skills_id_map.pkl"

def _compose_text(row: dict) -> str:
    """Compose canonical text for embedding/search: ONLY topic + problem_statement (no options)."""
    topic = row.get("topic") or ""
    problem = row.get("problem_statement") or row.get("problem_text") or ""
    parts = [topic.strip(), problem.strip()]
    return " | ".join([p for p in parts if p])

def build(train_csv_path: str, cache_dir: str):
    train_csv = Path(train_csv_path)
    cache_dir = Path(cache_dir)
    sqlite_dir = cache_dir / "sqlite"
    faiss_dir = cache_dir / "faiss"
    sqlite_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = sqlite_dir / "skills.db"
    index_path = faiss_dir / FAISS_INDEX_FILENAME
    id_map_path = faiss_dir / ID_MAP_FILENAME

    print(f"[build_index] Loading embedder {EMB_MODEL_NAME} ...")
    embedder = SentenceTransformer(EMB_MODEL_NAME)

    store = SQLiteStore(str(sqlite_path))
    vectors = []
    id_map = []  # list of sqlite ids, in same order as vectors

    print(f"[build_index] Reading train CSV: {train_csv}")
    with open(train_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            composed = _compose_text(row)
            topic = row.get("topic") or ""
            problem = row.get("problem_statement") or row.get("problem_text") or ""

            # Base solution text (existing behavior)
            solution_text = row.get("solution") or ""

            # Collect answer_option_1..N (if present) into an ordered list
            options = []
            # assume 1..50 is plenty — you can increase if needed
            for j in range(1, 6):
                k = f"answer_option_{j}"
                if k in row and row.get(k) not in (None, ""):
                    options.append(row.get(k))
                else:
                    # if the CSV has holes you want to skip (e.g. option_1, option_3),
                    # comment out the `else: break` to keep scanning; currently we keep scanning
                    # continue
                    pass

            # Also check common alternate key names (if your CSV uses them)
            if not options:
                # maybe options are stored as a single JSON/string column 'options'
                alt = row.get("options") or row.get("answer_options") or row.get("choices")
                if alt:
                    # try parse JSON list, else split on '||' or '|' heuristically
                    try:
                        import json
                        parsed = json.loads(alt)
                        if isinstance(parsed, (list, tuple)):
                            options = [str(x) for x in parsed]
                    except Exception:
                        # fallback split (adjust delimiter as needed)
                        if "||" in alt:
                            options = [s.strip() for s in alt.split("||") if s.strip()]
                        else:
                            options = [s.strip() for s in alt.split("|") if s.strip()]

            # Normalize correct option number if present
            raw_correct = row.get("correct_option_number") or row.get("correct_answer") or row.get("answer_index")
            correct_idx = None
            if raw_correct not in (None, ""):
                try:
                    # try integer-like
                    correct_idx = int(str(raw_correct).strip())
                except Exception:
                    # try letter like 'A'/'B' -> 1/2
                    rc = str(raw_correct).strip()
                    if len(rc) == 1 and rc.isalpha():
                        correct_idx = ord(rc.upper()) - ord("A") + 1
                    else:
                        # leave None if unparseable
                        correct_idx = None

            # Compose solution object to store in DB (extendable)
            solution_obj = {
                "final_answer_text": solution_text,
                # store both list and raw fields for compatibility
                "options": options,                        # [] if none found
                "correct_option_number": correct_idx       # None if not present/unparseable
            }

            # Insert into sqlite - uses normalized sha
            sid = store.insert_skill(
                topic=topic,
                problem_text=problem,
                composed_text=composed,
                simhash=None,
                solution=solution_obj,
                constraints=None,
                provenance=None
            )

            id_map.append(sid)
            vectors.append(composed)

    if not vectors:
        print("[build_index] No rows found in train CSV. Exiting.")
        return

    print(f"[build_index] Computing embeddings for {len(vectors)} examples ...")
    emb = embedder.encode(vectors, show_progress_bar=True, convert_to_numpy=True, batch_size=64)
    # L2 normalize for dot-product cosine search
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    d = emb.shape[1]
    print(f"[build_index] Building FAISS index (dim={d}) ...")
    index = faiss.IndexFlatIP(d)
    index.add(emb.astype("float32"))
    faiss.write_index(index, str(index_path))
    with open(id_map_path, "wb") as f:
        pickle.dump(id_map, f)
    print(f"[build_index] Wrote FAISS index to {index_path} and id_map to {id_map_path}")
    print(f"[build_index] SQLite DB at {sqlite_path} has {store.count()} rows.")