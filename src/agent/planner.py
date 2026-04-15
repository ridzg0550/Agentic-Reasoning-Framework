# src/agent/planner.py
"""
Planner that emits up to MAX_PLANNER_GOALS short goals. If exemplars are not passed
explicitly, it will consult self.last_exemplars (which run_batch attaches).
"""

import os
import json
from typing import List, Optional

from .llm_client import OllamaClient
from .params import MAX_PLANNER_GOALS

SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "schemas", "planner_goals.schema.json")

def _load_schema_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return json.dumps({"type":"object","properties":{"goals":{"type":"array","items":{"type":"string"},"maxItems":5}},"required":["goals"]})

class Planner:
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.schema_text = _load_schema_text(SCHEMA_FILE)
        self.last_exemplars = None

    def generate_goals(self, problem: str, topic: str, options: List[str], exemplars: Optional[List[dict]] = None) -> List[str]:
        # if exemplars not passed, use last_exemplars attached by run_batch
        if exemplars is None and getattr(self, "last_exemplars", None):
            exemplars = self.last_exemplars

        exemplar_lines = []
        if exemplars:
            for ex in exemplars[:3]:
                ct = ex.get("composed_text") or ex.get("problem_text") or ""
                sol = ""
                sol_obj = ex.get("solution") or {}
                if isinstance(sol_obj, dict):
                    sol = sol_obj.get("final_answer_text") or sol_obj.get("final_answer") or ""
                sol = sol or ex.get("solution_text") or ""
                exemplar_lines.append(f"Example: {ct[:400]} -> Answer: {str(sol)[:200]}")

        exemplars_text = "\n".join(exemplar_lines)

        prompt = (
            f"Produce up to {MAX_PLANNER_GOALS} concise goals (each <=10 words) as a single JSON object matching the provided schema.\n\n"
            f"Topic: {topic}\n"
            f"{('Exemplars:\\n' + exemplars_text + '\\n') if exemplars_text else ''}"
            f"Problem: {problem[:1500]}\n"
            f"Options count: {len(options)}\n"
            "Respond ONLY with a single JSON object and nothing else.\n"
            f"Schema:\n{self.schema_text}\n"
        )

        try:
            res = self.llm.generate_json(system_hint="You are a planner. Produce only a JSON object matching the schema. Keep goals short.",
                                         user_prompt=prompt, schema_text=self.schema_text, max_tokens=256, temperature=0.0)
            if res and isinstance(res, dict) and "goals" in res:
                goals = [g.strip() for g in res.get("goals") if isinstance(g, str) and g.strip()]
                return goals[:MAX_PLANNER_GOALS]
        except Exception:
            pass

        # fallback simple goals
        fallback = ["parse options", "extract numbers", "try python", "verify answer"]
        return fallback[:MAX_PLANNER_GOALS]