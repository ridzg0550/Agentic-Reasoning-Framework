"""Sanity checks for output validation."""

import pandas as pd
from typing import List, Dict, Any


def check_output_format(df: pd.DataFrame) -> List[str]:
    """
    Check if output DataFrame has correct format.
    
    Args:
        df: Output DataFrame
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required columns
    required_cols = ["topic", "problem_statement", "solution", "correct_option_number"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    if errors:
        return errors
    
    # Check data types
    if not pd.api.types.is_integer_dtype(df["correct_option_number"]):
        try:
            df["correct_option_number"] = df["correct_option_number"].astype(int)
        except:
            errors.append("correct_option_number must be integer")
    
    # Check value ranges
    invalid_options = df[
        (df["correct_option_number"] < 1) | 
        (df["correct_option_number"] > 5)
    ]
    if len(invalid_options) > 0:
        errors.append(
            f"{len(invalid_options)} rows have invalid option numbers "
            f"(must be 1-5)"
        )
    
    # Check for missing values
    for col in required_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            errors.append(f"{missing} missing values in column: {col}")
    
    return errors


def validate_output_csv(csv_path: str) -> bool:
    """
    Validate output CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Cannot read CSV: {e}")
        return False
    
    errors = check_output_format(df)
    
    if errors:
        print("VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ Output CSV is valid")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique topics: {df['topic'].nunique()}")
        return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sanity_checks.py <output.csv>")
        sys.exit(1)
    
    valid = validate_output_csv(sys.argv[1])
    sys.exit(0 if valid else 1)
