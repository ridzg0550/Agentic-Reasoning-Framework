"""Retriever tool for fetching similar examples from skill cache."""
import traceback
from typing import Dict, Any, List
from ..agent.params import RETRIEVER_TOP_K_EXEMPLARS
from ..skill_cache.index_store import IndexStore

class RetrieverTool:
    """Retrieve similar examples from skill cache."""

    def __init__(self, cache_path: str):
        print(f"[RetrieverTool.__init__] Initializing retriever with cache_path={cache_path}")
        self.cache_path = cache_path
        try:
            self.index_store = IndexStore(self.cache_path)
            print(f"[RetrieverTool] ✅ IndexStore initialized successfully: {self.index_store}")
        except Exception as e:
            print(f"[RetrieverTool] ❌ Failed to initialize IndexStore: {type(e).__name__}: {e}")
            traceback.print_exc()
            self.index_store = None

    def execute(self, problem: str, topic: str, k: int = RETRIEVER_TOP_K_EXEMPLARS) -> Dict[str, Any]:
        """Main entrypoint: fetch top-k similar exemplars for a given problem."""
        print(f"\n[RetrieverTool.execute] Invoked")
        print(f"  ↳ problem (truncated 200 chars): {problem[:200]!r}")
        print(f"  ↳ topic: {topic!r}")
        print(f"  ↳ k={k}")

        if not self.index_store:
            print("[RetrieverTool] ❌ index_store is None (not loaded).")
            return {"success": False, "data": {}, "error": "Index not loaded"}

        try:
            print("[RetrieverTool] 🔍 Calling index_store.retrieve_similar() ...")
            exemplars = self.index_store.retrieve_similar(problem_text=problem, topic=topic, top_k=k)

            if exemplars is None:
                print("[RetrieverTool] ❌ retrieve_similar() returned None")
                return {"success": False, "data": {}, "error": "retrieve_similar() returned None"}

            print(f"[RetrieverTool] ✅ retrieve_similar() returned {len(exemplars)} exemplar(s)")

            formatted: List[Dict[str, Any]] = []
            for i, ex in enumerate(exemplars or []):
                rec = ex.get("record", {})
                score = ex.get("score", ex.get("similarity", 0.0))
                formatted_entry = {
                    "topic": rec.get("topic", ""),
                    "problem": rec.get("problem_text", ""),
                    "approach": rec.get("solution_summary", "")[:200],
                    "similarity": score,
                    "exact": bool(ex.get("exact", False))
                }
                formatted.append(formatted_entry)
                print(f"  [RetrieverTool] Exemplar {i+1}: score={score:.4f}, topic={formatted_entry['topic']!r}, "
                      f"problem_preview={formatted_entry['problem'][:100]!r}")

            print(f"[RetrieverTool] 🧩 Formatted {len(formatted)} exemplars ready for output.")

            return {
                "success": True,
                "data": {"exemplars": formatted, "count": len(formatted)},
                "error": ""
            }

        except Exception as e:
            print(f"[RetrieverTool] ❌ Exception during execute(): {type(e).__name__}: {e}")
            traceback.print_exc()
            return {"success": False, "data": {}, "error": f"{type(e).__name__}: {str(e)}"}