"""Compute Macro F1 score for evaluation."""

import sys
import pandas as pd
import argparse
from sklearn.metrics import f1_score
from typing import Dict, List


def compute_macro_f1(true_labels: List[int], pred_labels: List[int], 
                     topics: List[str] = None) -> Dict[str, float]:
    """
    Compute Macro F1 score.
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        topics: Optional topic labels for per-topic F1
        
    Returns:
        Dict with macro_f1 and optional per-topic scores
    """
    # Global Macro F1
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    
    results = {
        "macro_f1": macro_f1
    }
    
    # Per-topic F1 if topics provided
    if topics:
        unique_topics = sorted(set(topics))
        per_topic = {}
        
        for topic in unique_topics:
            topic_mask = [t == topic for t in topics]
            topic_true = [t for t, m in zip(true_labels, topic_mask) if m]
            topic_pred = [p for p, m in zip(pred_labels, topic_mask) if m]
            
            if topic_true:
                topic_f1 = f1_score(topic_true, topic_pred, average='macro')
                per_topic[topic] = topic_f1
        
        results["per_topic"] = per_topic
    
    return results


def evaluate_predictions(ground_truth_csv: str, predictions_csv: str):
    """
    Evaluate predictions against ground truth.
    
    Args:
        ground_truth_csv: Path to ground truth CSV (train.csv format)
        predictions_csv: Path to predictions CSV (output.csv format)
    """
    print(f"Loading ground truth from {ground_truth_csv}...")
    gt_df = pd.read_csv(ground_truth_csv)
    
    print(f"Loading predictions from {predictions_csv}...")
    pred_df = pd.read_csv(predictions_csv)
    
    # Match by problem statement
    merged = gt_df.merge(
        pred_df, 
        on="problem_statement", 
        suffixes=("_true", "_pred")
    )
    
    print(f"Matched {len(merged)} problems")
    
    if len(merged) == 0:
        print("ERROR: No matching problems found!")
        return
    
    # Extract labels
    true_labels = merged["correct_option_number"].tolist()
    pred_labels = merged["correct_option_number_pred"].tolist()
    topics = merged["topic_true"].tolist()
    
    # Compute metrics
    print("\nComputing metrics...")
    results = compute_macro_f1(true_labels, pred_labels, topics)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Macro F1: {results['macro_f1']:.4f}")
    
    if "per_topic" in results:
        print("\nPer-Topic F1:")
        for topic, f1 in sorted(results["per_topic"].items()):
            print(f"  {topic}: {f1:.4f}")
    
    # Accuracy
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / len(true_labels)
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(true_labels)})")


def main():
    parser = argparse.ArgumentParser(description="Compute Macro F1 score")
    parser.add_argument("ground_truth_csv", help="Path to ground truth CSV")
    parser.add_argument("predictions_csv", help="Path to predictions CSV")
    
    args = parser.parse_args()
    
    evaluate_predictions(args.ground_truth_csv, args.predictions_csv)


if __name__ == "__main__":
    main()
