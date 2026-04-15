"""Parse MCQ options from problem text."""

from typing import List, Dict, Any
import re


def parse_options(problem_text: str, options: List[str]) -> Dict[str, Any]:
    """
    Parse and structure MCQ options.
    
    Args:
        problem_text: Problem statement
        options: List of option strings
        
    Returns:
        Parsed options dict
    """
    parsed = {
        "count": len(options),
        "options": [],
        "has_numeric": False,
        "has_text": False
    }
    
    for i, opt in enumerate(options, 1):
        opt_info = {
            "number": i,
            "text": opt,
            "numeric_values": [],
            "type": "text"
        }
        
        # Extract numbers
        numbers = re.findall(r'-?\d+\.?\d*', opt)
        if numbers:
            opt_info["numeric_values"] = [float(n) for n in numbers]
            opt_info["type"] = "numeric"
            parsed["has_numeric"] = True
        else:
            parsed["has_text"] = True
        
        parsed["options"].append(opt_info)
    
    return parsed
