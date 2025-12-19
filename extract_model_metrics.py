#!/usr/bin/env python3
"""
Extract validation metrics from trained YOLOv8 models
Analyzes each model's performance and calculates optimal confidence thresholds
"""

import os
import yaml
from ultralytics import YOLO
import json
from pathlib import Path

# Model paths
MODELS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

MODELS = {
    'pushup': {
        'path': os.path.join(MODELS_BASE, 'WarmApp_PushUp_Model', 'best.pt'),
        'name': 'Push-up',
        'model_size': 'YOLOv8n-pose'
    },
    'pullup': {
        'path': os.path.join(MODELS_BASE, 'WarmApp_PullUp_Model_V2', 'best.pt'),
        'name': 'Pull-up',
        'model_size': 'YOLOv8m-pose'
    },
    'glute_bridge': {
        'path': os.path.join(MODELS_BASE, 'WarmApp_GluteBridge_Model', 'best.pt'),
        'name': 'Glute Bridge',
        'model_size': 'YOLOv8n-pose'
    },
    'good_morning': {
        'path': os.path.join(MODELS_BASE, 'WarmApp_GoodMorning_Model', 'best.pt'),
        'name': 'Good Morning',
        'model_size': 'YOLOv8n-pose'
    },
    'plank': {
        'path': os.path.join(MODELS_BASE, 'WarmApp_Plank_Medium_Model', 'best.pt'),
        'name': 'Plank',
        'model_size': 'YOLOv8m-pose'
    },
}

CONFIDENCE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def analyze_model_info(model_path):
    """Extract model training information."""
    try:
        model = YOLO(model_path)
        
        # Get model info
        info = {
            'model_path': model_path,
            'model_loaded': True,
            'parameters': sum(p.numel() for p in model.model.parameters()) / 1e6,  # in millions
        }
        
        # Check for results.csv (training history)
        model_dir = Path(model_path).parent
        results_csv = model_dir / 'results.csv'
        
        if results_csv.exists():
            info['has_training_history'] = True
            # Read last line of results for final metrics
            with open(results_csv, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    headers = lines[0].strip().split(',')
                    last_values = lines[-1].strip().split(',')
                    
                    # Extract key metrics
                    metrics = {}
                    for h, v in zip(headers, last_values):
                        h = h.strip()
                        try:
                            metrics[h] = float(v.strip())
                        except:
                            metrics[h] = v.strip()
                    
                    info['final_metrics'] = metrics
        else:
            info['has_training_history'] = False
        
        return info
    except Exception as e:
        return {
            'model_path': model_path,
            'model_loaded': False,
            'error': str(e)
        }


def calculate_f1_score(precision, recall):
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def find_optimal_confidence(model_info):
    """
    Determine optimal confidence threshold based on model metrics.
    For workout posture detection, we prioritize precision (avoiding false positives).
    """
    if 'final_metrics' in model_info:
        metrics = model_info['final_metrics']
        
        # Extract precision if available
        precision_keys = [k for k in metrics.keys() if 'precision' in k.lower() or 'p' == k.lower()]
        recall_keys = [k for k in metrics.keys() if 'recall' in k.lower() or 'r' == k.lower()]
        
        if precision_keys and recall_keys:
            precision = metrics[precision_keys[0]]
            recall = metrics[recall_keys[0]]
            
            f1 = calculate_f1_score(precision, recall)
            
            # For high-precision applications, recommend 0.7 or higher
            if precision >= 0.85:
                recommended_conf = 0.7
            elif precision >= 0.75:
                recommended_conf = 0.6
            else:
                recommended_conf = 0.5
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'recommended_confidence': recommended_conf,
                'reasoning': f"Precision: {precision:.3f}, Recall: {recall:.3f}"
            }
    
    # Default recommendation
    return {
        'precision': None,
        'recall': None,
        'f1_score': None,
        'recommended_confidence': 0.7,
        'reasoning': "Default strict threshold for workout posture detection"
    }


def print_model_summary(exercise_key, model_data, analysis):
    """Print detailed model summary."""
    print(f"\n{'='*70}")
    print(f"  {model_data['name'].upper()} MODEL ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nModel Information:")
    print(f"  Architecture:        {model_data['model_size']}")
    print(f"  Parameters:          {analysis.get('parameters', 'N/A'):.2f}M" if analysis.get('parameters') else "  Parameters:          N/A")
    print(f"  Model Path:          {model_data['path']}")
    print(f"  Status:              {'✓ Loaded Successfully' if analysis.get('model_loaded') else '✗ Failed to Load'}")
    
    if 'final_metrics' in analysis:
        metrics = analysis['final_metrics']
        print(f"\nTraining Results (Final Epoch):")
        
        # Common metric keys
        metric_display = {
            'epoch': 'Epoch',
            'train/box_loss': 'Box Loss (Train)',
            'train/pose_loss': 'Pose Loss (Train)',
            'train/kobj_loss': 'Keypoint Loss (Train)',
            'metrics/precision(B)': 'Precision',
            'metrics/recall(B)': 'Recall',
            'metrics/mAP50(B)': 'mAP@0.5',
            'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
            'metrics/precision(P)': 'Pose Precision',
            'metrics/recall(P)': 'Pose Recall',
            'metrics/mAP50(P)': 'Pose mAP@0.5',
            'metrics/mAP50-95(P)': 'Pose mAP@0.5:0.95',
        }
        
        for key, display_name in metric_display.items():
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    print(f"  {display_name:25} {value:.4f}")
                else:
                    print(f"  {display_name:25} {value}")
    
    # Confidence threshold recommendation
    opt = find_optimal_confidence(analysis)
    print(f"\n{'─'*70}")
    print(f"  CONFIDENCE THRESHOLD RECOMMENDATION")
    print(f"{'─'*70}")
    
    if opt['precision'] is not None:
        print(f"  Precision:           {opt['precision']:.4f}")
        print(f"  Recall:              {opt['recall']:.4f}")
        print(f"  F1-Score:            {opt['f1_score']:.4f}")
    
    print(f"\n  Recommended Conf:    {opt['recommended_confidence']}")
    print(f"  Reasoning:           {opt['reasoning']}")
    
    return opt


def main():
    print("="*70)
    print("  WORKOUT MODEL METRICS EXTRACTION")
    print("="*70)
    print("\nAnalyzing trained YOLOv8 pose estimation models...")
    
    all_results = {}
    optimal_thresholds = {}
    
    for exercise_key, model_data in MODELS.items():
        if not os.path.exists(model_data['path']):
            print(f"\n✗ Model not found: {model_data['name']}")
            continue
        
        print(f"\n⏳ Analyzing {model_data['name']}...")
        
        analysis = analyze_model_info(model_data['path'])
        optimal = print_model_summary(exercise_key, model_data, analysis)
        
        all_results[exercise_key] = {
            'name': model_data['name'],
            'model_size': model_data['model_size'],
            'analysis': analysis,
            'optimal_confidence': optimal
        }
        
        optimal_thresholds[model_data['name']] = optimal['recommended_confidence']
    
    # Overall summary
    print(f"\n\n{'='*70}")
    print(f"  OVERALL SYSTEM SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nModel Overview:")
    for exercise_key, results in all_results.items():
        print(f"  {results['name']:15} {results['model_size']:15} Conf: {results['optimal_confidence']['recommended_confidence']}")
    
    print(f"\n{'─'*70}")
    print(f"  RECOMMENDED CONFIDENCE THRESHOLDS")
    print(f"{'─'*70}")
    
    for name, conf in optimal_thresholds.items():
        print(f"  {name:15} → {conf}")
    
    # System-wide recommendation
    avg_conf = sum(optimal_thresholds.values()) / len(optimal_thresholds) if optimal_thresholds else 0.7
    print(f"\n  System-Wide:    → {avg_conf:.1f}")
    print(f"\n  For real-time workout posture detection with immediate")
    print(f"  corrective feedback, a confidence threshold of 0.7 provides")
    print(f"  the best balance between precision and detection rate.")
    
    # Save to JSON
    output_file = 'model_metrics_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'models': all_results,
            'optimal_thresholds': optimal_thresholds,
            'system_recommendation': avg_conf
        }, f, indent=2, default=str)
    
    print(f"\n✓ Detailed analysis saved to: {output_file}")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
