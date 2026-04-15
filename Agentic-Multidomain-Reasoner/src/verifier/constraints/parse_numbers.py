"""Parse numeric values and constraints from text."""

from typing import List, Dict, Any
import re


def parse_numbers(text: str) -> Dict[str, Any]:
    """
    Extract numeric values and relationships from text.
    
    Args:
        text: Input text
        
    Returns:
        Dict of parsed numeric information
    """
    parsed = {
        "integers": [],
        "floats": [],
        "fractions": [],
        "percentages": [],
        "ranges": []
    }
    
    # Extract integers
    integers = re.findall(r'\b\d+\b', text)
    parsed["integers"] = [int(i) for i in integers]
    
    # Extract floats
    floats = re.findall(r'\b\d+\.\d+\b', text)
    parsed["floats"] = [float(f) for f in floats]
    
    # Extract percentages
    percentages = re.findall(r'\b(\d+\.?\d*)\s*%', text)
    parsed["percentages"] = [float(p) / 100 for p in percentages]
    
    # Extract ranges
    ranges = re.findall(r'(\d+)\s*(?:to|-)\s*(\d+)', text)
    parsed["ranges"] = [(int(a), int(b)) for a, b in ranges]
    
    return parsed
