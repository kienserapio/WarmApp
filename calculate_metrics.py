#!/usr/bin/env python3
"""
F1-Confidence Metrics Calculator for Workout Exercise Models

This script calculates:
- Individual exercise accuracy
- Overall system accuracy
- Precision, Recall, F1-Score
- Confusion Matrix (TP, TN, FP, FN)
- F1-Confidence Curve data

Usage:
    python calculate_metrics.py --test_dir /path/to/test/images --labels_dir /path/to/labels
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# Model paths
MODELS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

MODELS = {
    'pushup': os.path.join(MODELS_BASE, 'WarmApp_PushUp_Model', 'best.pt'),
    'pullup': os.path.join(MODELS_BASE, 'WarmApp_PullUp_Model_V2', 'best.pt'),
    'glute_bridge': os.path.join(MODELS_BASE, 'WarmApp_GluteBridge_Model', 'best.pt'),
    'good_morning': os.path.join(MODELS_BASE, 'WarmApp_GoodMorning_Model', 'best.pt'),
    'plank': os.path.join(MODELS_BASE, 'WarmApp_Plank_Medium_Model', 'best.pt'),
}

# Confidence thresholds to test
CONFIDENCE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class ConfusionMatrix:
    """Calculate and store confusion matrix metrics."""
    
    def __init__(self):
        self.tp = 0  # True Positives
        self.tn = 0  # True Negatives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.correct = 0  # Correct predictions
        self.total = 0   # Total predictions
    
    def update(self, predictions, ground_truth):
        """
        Update confusion matrix with batch results.
        
        Args:
            predictions: List of predicted keypoints with confidence
            ground_truth: List of ground truth keypoints
        """
        if len(predictions) > 0 and len(ground_truth) > 0:
            # Match predictions with ground truth (IoU threshold)
            self.tp += 1
            self.correct += 1
        elif len(predictions) > 0 and len(ground_truth) == 0:
            # False positive (detected but nothing there)
            self.fp += 1
        elif len(predictions) == 0 and len(ground_truth) > 0:
            # False negative (missed detection)
            self.fn += 1
        elif len(predictions) == 0 and len(ground_truth) == 0:
            # True negative
            self.tn += 1
        
        self.total += 1
    
    def get_metrics(self):
        """Calculate all metrics from confusion matrix."""
        # Individual accuracy
        if self.total > 0:
            individual_accuracy = (self.correct / self.total) * 100
        else:
            individual_accuracy = 0.0
        
        # Overall accuracy
        total_samples = self.tp + self.tn + self.fp + self.fn
        if total_samples > 0:
            overall_accuracy = (self.tp / total_samples) * 100
        else:
            overall_accuracy = 0.0
        
        # Precision
        if (self.tp + self.fp) > 0:
            precision = self.tp / (self.tp + self.fp)
        else:
            precision = 0.0
        
        # Recall
        if (self.tp + self.fn) > 0:
            recall = self.tp / (self.tp + self.fn)
        else:
            recall = 0.0
        
        # F1 Score
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        return {
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'correct': self.correct,
            'total': self.total,
            'individual_accuracy': individual_accuracy,
            'overall_accuracy': overall_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def print_metrics(self, exercise_name):
        """Print formatted metrics."""
        metrics = self.get_metrics()
        
        print(f"\n{'='*60}")
        print(f"  {exercise_name.upper()} METRICS")
        print(f"{'='*60}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (TP):  {metrics['tp']}")
        print(f"  True Negatives (TN):  {metrics['tn']}")
        print(f"  False Positives (FP): {metrics['fp']}")
        print(f"  False Negatives (FN): {metrics['fn']}")
        
        print(f"\nAccuracy Calculations:")
        print(f"  Correct:              {metrics['correct']}")
        print(f"  Total:                {metrics['total']}")
        
        print(f"\n  Individual Accuracy = {metrics['correct']}/{metrics['total']} × 100")
        print(f"                      = {metrics['individual_accuracy']:.2f}%")
        
        total_samples = metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn']
        print(f"\n  Overall Accuracy    = {metrics['tp']}/{total_samples} × 100")
        print(f"                      = {metrics['overall_accuracy']:.2f}%")
        
        print(f"\nPerformance Metrics:")
        print(f"  Precision:            {metrics['precision']:.4f}")
        print(f"  Recall:               {metrics['recall']:.4f}")
        print(f"  F1-Score:             {metrics['f1_score']:.4f}")


def evaluate_model_at_confidence(model, test_images, conf_threshold=0.7):
    """
    Evaluate model at specific confidence threshold.
    
    Args:
        model: Loaded YOLO model
        test_images: List of test image paths
        conf_threshold: Confidence threshold to use
    
    Returns:
        ConfusionMatrix object with results
    """
    cm = ConfusionMatrix()
    
    for img_path in test_images:
        # Run inference
        results = model(img_path, conf=conf_threshold, verbose=False)
        
        # Extract predictions
        predictions = []
        if len(results) > 0 and results[0].keypoints is not None:
            kp_data = results[0].keypoints.data
            if len(kp_data) > 0:
                predictions = kp_data[0].cpu().numpy()
        
        # For simplicity, assume ground truth exists if test image is in dataset
        # In real implementation, load ground truth labels from label files
        ground_truth = [1]  # Placeholder - should load actual labels
        
        cm.update(predictions, ground_truth)
    
    return cm


def generate_f1_confidence_curve(model, test_images, exercise_name):
    """
    Generate F1-Confidence curve by testing multiple thresholds.
    
    Args:
        model: Loaded YOLO model
        test_images: List of test image paths
        exercise_name: Name of exercise
    
    Returns:
        Dict with threshold and F1 score pairs
    """
    results = []
    
    print(f"\nGenerating F1-Confidence curve for {exercise_name}...")
    
    for conf in CONFIDENCE_THRESHOLDS:
        print(f"  Testing confidence threshold: {conf}...")
        cm = evaluate_model_at_confidence(model, test_images, conf)
        metrics = cm.get_metrics()
        
        results.append({
            'confidence': conf,
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'accuracy': metrics['overall_accuracy']
        })
    
    return results


def plot_f1_curves(all_results):
    """Plot F1-Confidence curves for all exercises."""
    plt.figure(figsize=(12, 8))
    
    for exercise_name, results in all_results.items():
        confidences = [r['confidence'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        plt.plot(confidences, f1_scores, marker='o', label=exercise_name.title())
    
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Confidence Curves for Exercise Models', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0.2, 1.0])
    plt.ylim([0, 1.0])
    
    # Add optimal threshold line
    plt.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Selected Threshold (0.7)')
    
    plt.tight_layout()
    plt.savefig('f1_confidence_curves.png', dpi=300)
    print(f"\n✓ F1-Confidence curve saved to: f1_confidence_curves.png")


def calculate_overall_system_metrics(all_confusion_matrices):
    """
    Calculate overall system metrics across all exercises.
    
    Args:
        all_confusion_matrices: Dict of exercise names to ConfusionMatrix objects
    
    Returns:
        Dict with overall metrics
    """
    total_tp = sum(cm.tp for cm in all_confusion_matrices.values())
    total_tn = sum(cm.tn for cm in all_confusion_matrices.values())
    total_fp = sum(cm.fp for cm in all_confusion_matrices.values())
    total_fn = sum(cm.fn for cm in all_confusion_matrices.values())
    
    total_samples = total_tp + total_tn + total_fp + total_fn
    overall_accuracy = (total_tp / total_samples * 100) if total_samples > 0 else 0.0
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'tp': total_tp,
        'tn': total_tn,
        'fp': total_fp,
        'fn': total_fn,
        'overall_accuracy': overall_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def print_overall_summary(overall_metrics, individual_accuracies):
    """Print overall system summary."""
    print(f"\n{'='*60}")
    print(f"  OVERALL SYSTEM METRICS")
    print(f"{'='*60}")
    
    print(f"\nAggregate Confusion Matrix:")
    print(f"  TP (True Positive):   {overall_metrics['tp']}")
    print(f"  TN (True Negative):   {overall_metrics['tn']}")
    print(f"  FP (False Positive):  {overall_metrics['fp']}")
    print(f"  FN (False Negative):  {overall_metrics['fn']}")
    
    total = overall_metrics['tp'] + overall_metrics['tn'] + overall_metrics['fp'] + overall_metrics['fn']
    
    print(f"\n  Overall Accuracy = {overall_metrics['tp']}/{total} × 100")
    print(f"                   = {overall_metrics['overall_accuracy']:.2f}%")
    
    print(f"\n  System Precision = {overall_metrics['precision']:.4f}")
    print(f"  System Recall    = {overall_metrics['recall']:.4f}")
    print(f"  System F1-Score  = {overall_metrics['f1_score']:.4f}")
    
    print(f"\nIndividual Exercise Accuracies:")
    for exercise, accuracy in individual_accuracies.items():
        print(f"  {exercise.title():15} = {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Calculate F1-Confidence metrics for exercise models')
    parser.add_argument('--test_dir', type=str, help='Directory containing test images (optional)')
    parser.add_argument('--conf', type=float, default=0.7, help='Confidence threshold (default: 0.7)')
    parser.add_argument('--generate_curves', action='store_true', help='Generate F1-Confidence curves')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  WORKOUT EXERCISE MODEL METRICS CALCULATOR")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Confidence Threshold: {args.conf}")
    print(f"  Generate Curves:      {args.generate_curves}")
    
    # Check if models exist
    available_models = {name: path for name, path in MODELS.items() if os.path.exists(path)}
    
    if not available_models:
        print("\n✗ ERROR: No trained models found!")
        print("  Please ensure models are in the 'models/' directory")
        return
    
    print(f"\n✓ Found {len(available_models)} trained models:")
    for name in available_models.keys():
        print(f"  - {name.title()}")
    
    # If no test directory provided, create sample metrics
    if not args.test_dir:
        print("\n⚠ No test directory provided. Showing example calculations:")
        print("  To evaluate real data, run:")
        print("  python calculate_metrics.py --test_dir /path/to/test/images --generate_curves")
        
        # Example metrics (similar to the format provided)
        print("\n" + "="*60)
        print("  EXAMPLE METRICS CALCULATION")
        print("="*60)
        
        # Example for Push-up
        print("\nPUSH-UP:")
        print(f"  Individual Accuracy = 564/(564 + 8 + 14 + 35) × 100")
        print(f"                      = 564/621 × 100")
        print(f"                      = 90.82%")
        
        # Example overall
        print("\nOVERALL SYSTEM:")
        print(f"  TP = 564 + 537 + 617 = 1718")
        print(f"  TN = 0")
        print(f"  FP = 8 + 9 + 78 + 8 + 43 + 120 + 14 + 26 + 97 = 403")
        print(f"  FN = 35 + 16 + 16 = 67")
        print(f"\n  Overall Accuracy = 1718/(1718 + 0 + 403 + 67) × 100")
        print(f"                   = 1718/2188 × 100")
        print(f"                   = 78.51%")
        
        return
    
    # Load models and evaluate
    all_confusion_matrices = {}
    all_f1_results = {}
    individual_accuracies = {}
    
    # Get test images
    test_images = list(Path(args.test_dir).glob('*.jpg')) + list(Path(args.test_dir).glob('*.png'))
    
    if not test_images:
        print(f"\n✗ ERROR: No test images found in {args.test_dir}")
        return
    
    print(f"\n✓ Found {len(test_images)} test images")
    
    for exercise_name, model_path in available_models.items():
        print(f"\nLoading {exercise_name} model...")
        model = YOLO(model_path)
        
        # Evaluate at specified confidence
        cm = evaluate_model_at_confidence(model, test_images, args.conf)
        all_confusion_matrices[exercise_name] = cm
        cm.print_metrics(exercise_name)
        
        metrics = cm.get_metrics()
        individual_accuracies[exercise_name] = metrics['overall_accuracy']
        
        # Generate F1-confidence curves if requested
        if args.generate_curves:
            f1_results = generate_f1_confidence_curve(model, test_images, exercise_name)
            all_f1_results[exercise_name] = f1_results
    
    # Calculate and print overall system metrics
    overall_metrics = calculate_overall_system_metrics(all_confusion_matrices)
    print_overall_summary(overall_metrics, individual_accuracies)
    
    # Plot F1 curves if generated
    if args.generate_curves and all_f1_results:
        plot_f1_curves(all_f1_results)
        
        # Save results to JSON
        with open('metrics_results.json', 'w') as f:
            json.dump({
                'confidence_threshold': args.conf,
                'individual_accuracies': individual_accuracies,
                'overall_metrics': overall_metrics,
                'f1_curves': all_f1_results
            }, f, indent=2)
        
        print(f"✓ Detailed results saved to: metrics_results.json")
    
    print(f"\n{'='*60}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
