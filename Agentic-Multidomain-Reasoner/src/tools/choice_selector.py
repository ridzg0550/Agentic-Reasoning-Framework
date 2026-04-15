# src/tools/choice_selector.py
"""
ChoiceSelector: map computed text -> option index robustly.
Returns structure: {"data": {"selected_index": int, "confidence": float}, "success": True/False}
"""
import difflib
import re

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}
_ORDINAL_SUFFIX_RE = re.compile(r"^(\d+)(st|nd|rd|th)?$", re.IGNORECASE)

class ChoiceSelector:
    def __init__(self):
        pass

    def _text_to_index(self, text):
        if not text:
            return None
        t = text.strip().lower()
        if t.isdigit():
            try:
                return int(t)
            except Exception:
                return None
        if t in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[t]
        m = _ORDINAL_SUFFIX_RE.match(t)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def execute(self, computed_result, options):
        # try numeric/ordinal
        idx = self._text_to_index(str(computed_result))
        if idx and 1 <= idx <= max(1, len(options)):
            return {"success": True, "data": {"selected_index": idx, "confidence": 0.95}}
        # fuzzy match to options
        cand = str(computed_result or "").strip().lower()
        if not cand:
            return {"success": False, "data": {"selected_index": None, "confidence": 0.0}}
        best_score = 0.0
        best_idx = None
        for i, opt in enumerate(options):
            if not opt:
                continue
            opt_s = str(opt).strip().lower()
            if cand == opt_s or cand in opt_s or opt_s in cand:
                return {"success": True, "data": {"selected_index": i+1, "confidence": 0.9}}
            score = difflib.SequenceMatcher(None, cand, opt_s).ratio()
            if score > best_score:
                best_score = score
                best_idx = i+1
        if best_score >= 0.72 and best_idx:
            return {"success": True, "data": {"selected_index": best_idx, "confidence": best_score}}
        return {"success": False, "data": {"selected_index": None, "confidence": best_score}}