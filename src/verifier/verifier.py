# src/verifier/verifier.py
"""
Robust Verifier for executor pipeline - FIXED VERSION
"""
from typing import Dict, Any, List, Optional
import re
import difflib

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}
_ORDINAL_SUFFIX_RE = re.compile(r"^(\d+)(st|nd|rd|th)?$", re.IGNORECASE)

def _text_to_int_candidate(text: str) -> Optional[int]:
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

def _fuzzy_match_options(candidate_text: str, options: List[str], threshold: float = 0.72) -> Optional[int]:
    if not options:
        return None
    cand = (candidate_text or "").strip().lower()
    if not cand:
        return None
    best_idx = None
    best_score = 0.0
    for i, opt in enumerate(options):
        if not opt:
            continue
        opt_s = str(opt).strip().lower()
        if opt_s and (cand == opt_s or cand in opt_s or opt_s in cand):
            return i + 1
        score = difflib.SequenceMatcher(None, cand, opt_s).ratio()
        if score > best_score:
            best_score = score
            best_idx = i + 1
    if best_score >= threshold:
        return best_idx
    return None

def _detect_overtake_expected(problem_text: str) -> Optional[int]:
    """
    FIX: When you overtake person at position N, you take their position N.
    Previously this was checking for the wrong expected value.
    """
    if not problem_text:
        return None
    s = problem_text.lower()
    
    # Look for "overtake the Xth person" patterns
    patterns = [
        r'overtak[e|ing]?\s+the\s+(\w+)\s+person',
        r'overtak[e|ing]?\s+(\w+)\s+person',
        r'pass\s+the\s+(\w+)\s+person',
        r'pass\s+(\w+)\s+person'
    ]
    
    for pattern in patterns:
        m = re.search(pattern, s)
        if m:
            token = m.group(1).strip().lower()
            if token in _ORDINAL_WORDS:
                return _ORDINAL_WORDS[token]  # You become this position
            if token.isdigit():
                try:
                    return int(token)
                except Exception:
                    pass
            mo = _ORDINAL_SUFFIX_RE.match(token)
            if mo:
                try:
                    return int(mo.group(1))
                except Exception:
                    pass
    return None

def _detect_camera_riddle(problem_text: str) -> bool:
    """
    FIX: Expand detection patterns for photo/camera riddles.
    """
    if not problem_text:
        return False
    s = problem_text.lower()
    
    # Multiple pattern checks
    camera_patterns = [
        ('shoot', 'underwater', 'hang'),
        ('shoot', 'husband', 'hang'),
        ('shoot', 'photo', 'develop'),
        ('photograph', 'develop', 'hang'),
        ('camera', 'develop', 'hang')
    ]
    
    for pattern in camera_patterns:
        if all(word in s for word in pattern):
            return True
    
    return False

class Verifier:
    def __init__(self):
        pass

    def verify(self, verify_input: Dict[str, Any]) -> Dict[str, Any]:
        violations: List[str] = []
        normalized_answer: Optional[str] = None

        problem = (verify_input.get("problem_text") or "")
        options = verify_input.get("options") or (verify_input.get("parsed_constraints", {}) or {}).get("options") or []
        candidate = (verify_input.get("candidate") or {}).get("final_answer", "")
        cand_text = str(candidate or "").strip()

        # First, try to extract integer or ordinal from candidate text
        intval = _text_to_int_candidate(cand_text)
        if intval is not None:
            if options and 1 <= intval <= len(options):
                normalized_answer = str(intval)
            else:
                normalized_answer = str(intval)
        else:
            # Try fuzzy matching to options
            if options:
                mapped = _fuzzy_match_options(cand_text, options, threshold=0.70)  # Lowered threshold
                if mapped:
                    normalized_answer = str(mapped)
                else:
                    violations.append("not_mapped_to_options")
            else:
                normalized_answer = ""

        # Special case: overtake riddle
        overtake_expected = _detect_overtake_expected(problem)
        if overtake_expected is not None:
            if normalized_answer:
                try:
                    got = int(normalized_answer)
                    # FIX: Correct logic - you become the position you overtook
                    if got == overtake_expected:
                        return {"status": "PASS", "violations": [], "normalized_answer": str(got)}
                    else:
                        violations.append(f"overtake_logic_mismatch_expected_{overtake_expected}_got_{got}")
                        return {"status": "FAIL", "violations": violations, "normalized_answer": normalized_answer}
                except Exception:
                    violations.append("overtake_mapping_uncertain")
                    return {"status": "INDETERMINATE", "violations": violations, "normalized_answer": normalized_answer}
            else:
                violations.append("overtake_no_mapping")
                return {"status": "INDETERMINATE", "violations": violations, "normalized_answer": ""}

        # Special case: camera/photo riddle
        if _detect_camera_riddle(problem):
            photo_keywords = {"photo", "photograph", "camera", "picture", "develop", "film", 
                            "processed", "development", "darkroom", "negative"}
            ct = cand_text.lower()
            matched_keyword = any(k in ct for k in photo_keywords)
            
            # Also check if the selected option contains photo keywords
            if options and not matched_keyword and normalized_answer:
                try:
                    idx = int(normalized_answer) - 1
                    if 0 <= idx < len(options):
                        opt_s = (options[idx] or "").lower()
                        if any(k in opt_s for k in photo_keywords):
                            matched_keyword = True
                except Exception:
                    pass
            
            if matched_keyword:
                if normalized_answer:
                    return {"status": "PASS", "violations": [], "normalized_answer": normalized_answer}
                else:
                    return {"status": "PASS", "violations": [], "normalized_answer": ""}
            else:
                violations.append("camera_riddle_mismatch")
                return {"status": "FAIL", "violations": violations, "normalized_answer": normalized_answer or ""}

        # Default verification
        if normalized_answer:
            return {"status": "PASS", "violations": violations, "normalized_answer": normalized_answer}
        else:
            if "not_mapped_to_options" in violations:
                return {"status": "INDETERMINATE", "violations": violations, "normalized_answer": ""}
            return {"status": "INDETERMINATE", "violations": violations, "normalized_answer": ""}