# src/skill_cache/index_store.py

import os
import pickle
from typing import Optional, List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .sqlite_store import SQLiteStore

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_FILENAME = "skills.index"
ID_MAP_FILENAME = "skills_id_map.pkl"

class IndexStore:
    def __init__(self, sqlite_db_path: str, faiss_index_name: str = FAISS_INDEX_FILENAME):
        # Allow a cache directory OR a specific DB file
        self.sqlite_db_path = str(sqlite_db_path)
        if os.path.isdir(self.sqlite_db_path):
            cache_dir = self.sqlite_db_path
            self.sqlite_db_path = os.path.join(cache_dir, "sqlite", "skills.db")
        else:
            cache_dir = os.path.dirname(os.path.dirname(self.sqlite_db_path))

        # Ensure FAISS paths exist regardless of branch
        faiss_dir = os.path.join(cache_dir, "faiss")
        os.makedirs(faiss_dir, exist_ok=True)
        self.faiss_path = os.path.join(faiss_dir, faiss_index_name)
        self.id_map_path = os.path.join(faiss_dir, ID_MAP_FILENAME)

        # Initialize SQLite store
        self.store = SQLiteStore(self.sqlite_db_path)
        print(f"[IndexStore] Using SQLite DB: {self.sqlite_db_path}")

        # Initialize embedder
        self.embedder = SentenceTransformer(EMB_MODEL_NAME)

        # Load FAISS index if present
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
            print(f"[IndexStore] Loaded FAISS index from {self.faiss_path}")
        else:
            self.index = None
            print(f"[IndexStore] WARNING: FAISS index not found at {self.faiss_path}")

        # Load id_map if present (list of sqlite row ids aligned with FAISS vectors)
        if os.path.exists(self.id_map_path):
            try:
                with open(self.id_map_path, "rb") as f:
                    self.id_map = pickle.load(f)
                if not isinstance(self.id_map, list):
                    self.id_map = list(self.id_map)
                print(f"[IndexStore] Loaded id_map with {len(self.id_map)} entries from {self.id_map_path}")
            except Exception as e:
                print(f"[IndexStore] WARNING: failed to load id_map: {e}")
                self.id_map = []
        else:
            self.id_map = []
            print(f"[IndexStore] WARNING: id_map not found at {self.id_map_path}")

    def exact_lookup(self, problem_sha: str) -> Optional[Dict[str, Any]]:
        rec = self.store.get_by_problem_sha(problem_sha)
        return rec

    def retrieve_similar(self, problem_text: str, topic: Optional[str] = None, top_k: int = 3, min_score: float = 0.75) -> List[Dict[str, Any]]:
        """
        Return list of {"score": float, "record": dict} sorted by score desc.
        topic: optional filter (case-insensitive); if given, prefer same-topic but still return others if scores high.
        
        FIXED: Only embeds topic + problem_text (no options) for similarity comparison.
        """
        import numpy as np
        import json
        import traceback

        try:
            print(f"[IndexStore.retrieve_similar] Called (top_k={top_k}, min_score={min_score}, topic={topic})")
            if not problem_text:
                print("[IndexStore] Empty problem_text provided -> returning []")
                return []

            # Check embedder
            if not hasattr(self, "embedder") or self.embedder is None:
                print("[IndexStore] ERROR: embedder is not initialized on this IndexStore instance.")
                return []

            # FIXED: Compose query text with ONLY topic + problem_text (matching build_index logic)
            query_parts = []
            if topic:
                query_parts.append(topic.strip())
            query_parts.append(problem_text.strip())
            query_text = " | ".join([p for p in query_parts if p])
            
            print(f"[IndexStore] Query text (topic+problem only): {query_text[:200]}...")

            # Encode query
            try:
                emb = self.embedder.encode([query_text], convert_to_numpy=True)
            except Exception as e:
                print(f"[IndexStore] ERROR: embedding failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                return []

            # normalize
            try:
                norm = np.linalg.norm(emb, axis=1, keepdims=True)
                norm[norm == 0] = 1.0
                emb = emb / norm
                emb = emb.astype("float32")
            except Exception as e:
                print(f"[IndexStore] ERROR during normalization: {type(e).__name__}: {e}")
                traceback.print_exc()
                return []

            # Check index
            if not hasattr(self, "index") or self.index is None:
                print("[IndexStore] ❌ FAISS index not loaded (self.index is None).")
                # helpful debug: if we have a path attribute, show it
                if hasattr(self, "faiss_path"):
                    print(f"[IndexStore] Expected FAISS path: {getattr(self, 'faiss_path')}")
                return []

            total = getattr(self.index, "ntotal", None)
            print(f"[IndexStore] FAISS index ntotal = {total}")
            if total is None or total == 0:
                print("[IndexStore] FAISS index empty (ntotal == 0) -> returning []")
                return []

            k = min(top_k, total)
            if k <= 0:
                print("[IndexStore] Computed k <= 0 -> returning []")
                return []

            # Search
            try:
                print(f"[IndexStore] 🔍 Searching FAISS (k={k})...")
                D, I = self.index.search(emb, k)
            except Exception as e:
                print(f"[IndexStore] ERROR during FAISS search: {type(e).__name__}: {e}")
                traceback.print_exc()
                return []

            # Log raw results
            try:
                raw_D = D.tolist() if hasattr(D, "tolist") else D
                raw_I = I.tolist() if hasattr(I, "tolist") else I
                print(f"[IndexStore] Raw FAISS distances: {raw_D}")
                print(f"[IndexStore] Raw FAISS indices: {raw_I}")
            except Exception:
                print("[IndexStore] Could not convert FAISS outputs to list for logging.")

            scores = D[0].tolist()
            idxs = I[0].tolist()

            results = []
            id_map_len = len(getattr(self, "id_map", []))
            for sc, idx in zip(scores, idxs):
                print(f"[IndexStore] Candidate -> idx={idx}, raw_score={sc}")
                if idx is None:
                    print("[IndexStore] -> idx is None, skipping")
                    continue
                if idx < 0 or idx >= id_map_len:
                    print(f"[IndexStore] -> idx {idx} out of id_map range (0..{id_map_len-1}), skipping")
                    continue
                sqlite_id = self.id_map[idx]
                try:
                    rec = self.store._conn.execute("SELECT * FROM skill_entries WHERE id = ?", (sqlite_id,)).fetchone()
                except Exception as e:
                    print(f"[IndexStore] ERROR fetching sqlite_id {sqlite_id}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    rec = None
                if not rec:
                    print(f"[IndexStore] -> No sqlite record found for id {sqlite_id}, skipping")
                    continue

                recd = dict(rec)
                # decode solution_json to dict if possible
                try:
                    recd["solution"] = json.loads(recd.get("solution_json") or "{}")
                except Exception:
                    recd["solution"] = recd.get("solution_json")

                # topic match info (informational only)
                if topic and recd.get("topic"):
                    try:
                        topic_match = str(recd.get("topic")).strip().lower() == str(topic).strip().lower()
                    except Exception:
                        topic_match = False
                    print(f"[IndexStore] -> record sqlite_id={sqlite_id}, topic='{recd.get('topic')}', topic_match={topic_match}")

                # append candidate
                results.append({"score": float(sc), "record": recd})

            # sort and filter
            results.sort(key=lambda x: x["score"], reverse=True)
            print(f"[IndexStore] Candidates fetched from DB: {len(results)} (before min_score filter)")

            filtered = [r for r in results if r["score"] >= float(min_score)]
            print(f"[IndexStore] Candidates with score >= {min_score}: {len(filtered)}")

            if len(filtered) == 0:
                # For debugging, show the top-k pre-filtered results briefly
                if len(results) > 0:
                    print("[IndexStore] No candidates passed the min_score threshold. Top pre-filter candidates:")
                    for i, r in enumerate(results[:top_k]):
                        rec = r["record"]
                        sol_preview = str(rec.get("solution"))[:200]
                        print(f"  {i+1}. score={r['score']:.4f}, sqlite_id={rec.get('id')}, topic={rec.get('topic')}, solution_preview={sol_preview!r}")
                else:
                    print("[IndexStore] No candidates were found at all (pre-filter).")
            else:
                print(f"[IndexStore] Returning top {min(len(filtered), top_k)} candidates (score >= {min_score}):")
                for i, r in enumerate(filtered[:top_k]):
                    rec = r["record"]
                    sol_preview = str(rec.get("solution"))[:200]
                    print(f"  {i+1}. score={r['score']:.4f}, sqlite_id={rec.get('id')}, topic={rec.get('topic')}, solution_preview={sol_preview!r}")

            return filtered[:top_k]
        except Exception as e:
            print(f"[IndexStore] Exception in retrieve_similar: {type(e).__name__}: {e}")
            traceback.print_exc()
            return []