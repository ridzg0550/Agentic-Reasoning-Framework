# src/agent/context_manager.py
"""
Context manager for compact ReAct history and variable extraction.
"""
from typing import List, Dict, Any

class ContextManager:
    def __init__(self, max_chars: int = 4000, keep_last_steps: int = 3):
        self.max_chars = max_chars
        self.keep_last_steps = keep_last_steps
        self.history: List[Dict[str, Any]] = []
        self.variables: Dict[str, Any] = {}
        self.key_facts: Dict[str, Any] = {}

    def add_step(self, thought: str, action: Dict[str, Any], observation: Dict[str, Any]) -> None:
        step = {"thought": thought or "", "action": action or {}, "observation": observation or {}}
        self.history.append(step)
        self._extract_vars(step)

    def _extract_vars(self, step: Dict[str, Any]) -> None:
        obs = step.get("observation", {}) or {}
        if obs.get("ok") and isinstance(obs.get("data"), dict):
            data = obs["data"]
            if "result" in data:
                tool = obs.get("tool", "unknown")
                self.variables[tool] = data["result"]
            for k in ("constraints", "numbers", "parsed"):
                if k in data and isinstance(data[k], dict):
                    self.key_facts.update(data[k])

    def build_context(self, problem: str, topic: str, options: List[str]) -> str:
        # Base header
        header = f"Topic: {topic}\nProblem: {problem[:1200]}\nOptions: {len(options)} choices.\n"
        # Variables snapshot
        vars_str = ""
        if self.variables:
            pairs = []
            for k, v in list(self.variables.items())[-8:]:
                pairs.append(f"{k}={repr(v)[:120]}")
            vars_str = "Vars: " + ", ".join(pairs) + "\n"
        # Recent steps
        steps = self.history[-self.keep_last_steps:]
        lines: List[str] = []
        for i, st in enumerate(steps, 1):
            t = st.get("thought", "")
            a = st.get("action", {})
            o = st.get("observation", {})
            tool = a.get("tool") or o.get("tool") or ""
            ok = o.get("ok")
            brief = ""
            data = o.get("data") or {}
            if isinstance(data, dict) and "result" in data:
                brief = f" -> result={repr(data['result'])[:120]}"
            lines.append(f"[{i}] tool={tool} ok={ok} thought={t[:160]}{brief}")
        body = "\n".join(lines) + ("\n" if lines else "")
        ctx = header + vars_str + body
        if len(ctx) > self.max_chars:
            ctx = ctx[-self.max_chars:]
        return ctx
