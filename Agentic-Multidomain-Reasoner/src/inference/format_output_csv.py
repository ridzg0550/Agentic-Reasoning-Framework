# src/inference/format_output_csv.py
"""
Utilities to normalize output dataframe and map a predicted solution string to one of the
provided answer options. The map_answer_to_option function returns 1-based index (1..N)
or None if mapping failed.
"""

import re
import difflib
from typing import List, Optional

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # remove surrounding punctuation and multiple spaces
    s = re.sub(r"\s+", " ", s)
    s = s.strip(' "\'')
    return s.lower()

def map_answer_to_option(solution_text: Optional[str], options: List[str]) -> Optional[int]:
    """
    Try to map a free-form solution_text to one of the provided options.
    Returns 1-based index if matched else None.
    Strategies (in order):
      1) If solution_text equals an option exactly (case-insensitive) -> that index.
      2) If solution_text looks like an option key (e.g., 'A', '1', '2') -> map by position.
      3) Fuzzy match using SequenceMatcher, threshold 0.7 -> best index.
    """
    if not solution_text:
        return None
    sol = _normalize_text(solution_text)

    # direct equality (strip, lower)
    for i, opt in enumerate(options):
        if _normalize_text(opt) == sol:
            return i + 1

    # If solution_text is short like 'A' or 'B' or '1' or '2' map accordingly
    token = sol.strip().upper()
    if token.isalpha() and len(token) == 1:
        # map A->1, B->2 ...
        idx = ord(token) - ord("A")
        if 0 <= idx < len(options):
            return idx + 1
    if token.isdigit():
        idx = int(token) - 1
        if 0 <= idx < len(options):
            return idx + 1

    # fuzzy matching
    best_ratio = 0.0
    best_idx = None
    for i, opt in enumerate(options):
        ratio = difflib.SequenceMatcher(None, _normalize_text(opt), sol).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    if best_ratio >= 0.70:
        return best_idx + 1

    # last-ditch: check if an option substring is contained in solution_text
    for i, opt in enumerate(options):
        if _normalize_text(opt) in sol or sol in _normalize_text(opt):
            return i + 1

    return None

def normalize_dataframe(df):
    """
    Ensure output dataframe contains required columns and canonical types for output.csv.
    This is a small helper - keep simple.
    """
    # ensure columns exist
    if "solution" not in df.columns:
        df["solution"] = None
    if "correct_option_number_pred" not in df.columns:
        df["correct_option_number_pred"] = None
    return df