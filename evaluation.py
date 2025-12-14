"""
Experimental Evaluation Module for PDA-based AV Block Classification

Implements comprehensive evaluation metrics, baseline comparisons, and result analysis.

Authors: Salma Bellaou, Asma Khamri
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import time

from pda_classifier import PDAClassifier, AVBlockType as PDABlockType
from data_generator import AVBlockType as DataBlockType


class BaselineClassifier:
    """
    Baseline classifier implementations for comparison.
    """
    
    def __init__(self, method: str = "rule_based"):
        """
        Args:
            method: Baseline method ('rule_based', 'regex', or 'fsa')
        """
        self.method = method
    
    def classify_rule_based(self, sequence: List[str]) -> str:
        """
        Simple rule-based classifier using if-then rules.
        Does not use stack memory.
        """
        # Count pattern occurrences
        pr_long_count = sequence.count("PRlong")
        pr_increase_count = sequence.count("PRincrease")
        p_count = sequence.count("P")
        r_count = sequence.count("R")
        
        # Simple rules (these will miss complex temporal patterns)
        if pr_long_count >= 3 and pr_increase_count == 0:
            # Guess first-degree if many PRlong
            return "first_degree"
        elif pr_increase_count >= 2:
            # Guess Mobitz I if multiple increases
            return "mobitz_i"
        elif pr_long_count > 0 and p_count > r_count:
            # Guess Mobitz II if some drops
            return "mobitz_ii"
        elif p_count > r_count + 2:
            # Guess third-degree if many more P than R
            return "third_degree"
        else:
            return "normal"
    
    def classify_fsa(self, sequence: List[str]) -> str:
        """
        Finite State Automaton (no stack memory).
        Limited to regular patterns.
        """
        # FSA without stack cannot properly track context
        # This will fail on patterns requiring memory
        
        state = "START"
        pr_long_seen = 0
        consecutive_p = 0
        
        for symbol in sequence:
            if symbol == "P":
                consecutive_p += 1
            elif symbol == "R":
                consecutive_p = 0
            elif symbol == "PRlong":
                pr_long_seen += 1
            elif symbol == "PRincrease":
                # FSA cannot properly track progression
                return "mobitz_i"
        
        # Without stack, cannot accurately classify
        if pr_long_seen >= 3:
            return "first_degree"
        elif consecutive_p >= 2:
            return "mobitz_ii"  # Might confuse with third-degree
        else:
            return "normal"
    
    def classify(self, sequence: List[str]) -> str:
        """Classify using selected baseline method"""
        if self.method == "rule_based":
            return self.classify_rule_based(sequence)
        elif self.method == "fsa":
            return self.classify_fsa(sequence)
        else:
            return self.classify_rule_based(sequence)


class Evaluator:
    """
    Comprehensive evaluation framework for AV block classification.
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.label_mapping = {
            "first_degree": PDABlockType.FIRST_DEGREE.value,
            "mobitz_i": PDABlockType.MOBITZ_I.value,
            "mobitz_ii": PDABlockType.MOBITZ_II.value,
            "third_degree": PDABlockType.THIRD_DEGREE.value,
            "normal": PDABlockType.NORMAL.value
        }
    
    def evaluate_model(self, 
                      classifier, 
                      dataset: List[dict],
                      model_name: str = "PDA") -> Dict:
        """
        Evaluate a classifier on the dataset.
        
        Args:
            classifier: Classifier object with classify() method
            dataset: List of samples with 'sequence' and 'label'
            model_name: Name of the model for reporting
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        predictions = []
        ground_truth = []
        inference_times = []
        
        print(f"\nEvaluating {model_name}...")
        print("=" * 60)
        
        for sample in dataset:
            sequence = sample['sequence']
            true_label = sample['label']
            
            # Time the inference
            start_time = time.time()
            
            if model_name == "PDA":
                pred_enum, _ = classifier.classify(sequence)
                pred_label = pred_enum.value
            else:
                pred_label_key = classifier.classify(sequence)
                pred_label = self.label_mapping.get(pred_label_key, "Unknown")
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Map true label
            true_label_mapped = self.label_mapping.get(true_label, true_label)
            
            predictions.append(pred_label)
            ground_truth.append(true_label_mapped)
        
        # Calculate metrics
        metrics = self.calculate_metrics(ground_truth, predictions)
        metrics['avg_inference_time_ms'] = np.mean(inference_times) * 1000
        metrics['std_inference_time_ms'] = np.std(inference_times) * 1000
        
        return metrics
    
    def calculate_metrics(self, 
                         y_true: List[str], 
                         y_pred: List[str]) -> Dict:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary of metrics
        """
        # Get unique labels
        labels = sorted(set(y_true + y_pred))
        
        # Initialize confusion matrix
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        for true, pred in zip(y_true, y_pred):
            confusion_matrix[true][pred] += 1
        
        # Per-class metrics
        class_metrics = {}
        
        for label in labels:
            tp = confusion_matrix[label][label]
            fp = sum(confusion_matrix[other][label] 
                    for other in labels if other != label)
            fn = sum(confusion_matrix[label][other] 
                    for other in labels if other != label)
            tn = sum(confusion_matrix[other1][other2]
                    for other1 in labels if other1 != label
                    for other2 in labels if other2 != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': tp + fn
            }
        
        # Overall metrics
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        
        # Macro-averaged metrics
        macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in class_metrics.values()])
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'class_metrics': class_metrics,
            'confusion_matrix': dict(confusion_matrix),
            'num_samples': len(y_true)
        }
    
    def print_results(self, metrics: Dict, model_name: str):
        """Print formatted evaluation results"""
        print(f"\n{'='*60}")
        print(f"Results for {model_name}")
        print(f"{'='*60}\n")
        
        print(f"Overall Metrics:")
        print(f"  Accuracy:         {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
        print(f"  Macro F1-Score:   {metrics['macro_f1']:.4f}")
        print(f"  Avg Inference:    {metrics.get('avg_inference_time_ms', 0):.3f} ms")
        print(f"  Total Samples:    {metrics['num_samples']}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        
        for class_name, class_metric in sorted(metrics['class_metrics'].items()):
            print(f"{class_name:<30} "
                  f"{class_metric['precision']:<12.4f} "
                  f"{class_metric['recall']:<12.4f} "
                  f"{class_metric['f1_score']:<12.4f} "
                  f"{class_metric['support']:<10}")
        
        print(f"\nConfusion Matrix:")
        self.print_confusion_matrix(metrics['confusion_matrix'])
    
    def print_confusion_matrix(self, cm: Dict):
        """Print formatted confusion matrix"""
        labels = sorted(set(list(cm.keys()) + 
                           [k for v in cm.values() for k in v.keys()]))
        
        # Truncate labels for display
        display_labels = [l.replace("AV Block", "").replace("(", "").replace(")", "").strip()[:15] 
                         for l in labels]
        
        # Header
        print(f"\n{'Predicted →':<20}", end="")
        for label in display_labels:
            print(f"{label:<15}", end="")
        print()
        print(f"{'Actual ↓':<20}" + "-" * (15 * len(labels)))
        
        # Rows
        for i, true_label in enumerate(labels):
            print(f"{display_labels[i]:<20}", end="")
            for pred_label in labels:
                count = cm.get(true_label, {}).get(pred_label, 0)
                print(f"{count:<15}", end="")
            print()
    
    def compare_models(self, results: Dict[str, Dict]):
        """
        Compare multiple models and print comparison table.
        
        Args:
            results: Dictionary mapping model names to their metrics
        """
        print(f"\n{'='*80}")
        print("Model Comparison")
        print(f"{'='*80}\n")
        
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        
        for model_name, metrics in sorted(results.items()):
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['macro_precision']:<12.4f} "
                  f"{metrics['macro_recall']:<12.4f} "
                  f"{metrics['macro_f1']:<12.4f}")
        
        print("\n" + "="*80)


def run_experiments(dataset_path: str):
    """
    Run complete experimental evaluation.
    
    Args:
        dataset_path: Path to the dataset JSON file
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded dataset: {len(dataset)} samples")
    
    # Initialize models
    pda_classifier = PDAClassifier(verbose=False)
    rule_based = BaselineClassifier(method="rule_based")
    fsa_baseline = BaselineClassifier(method="fsa")
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Evaluate PDA
    pda_results = evaluator.evaluate_model(pda_classifier, dataset, "PDA")
    evaluator.print_results(pda_results, "Pushdown Automaton (PDA)")
    
    # Evaluate rule-based baseline
    rule_results = evaluator.evaluate_model(rule_based, dataset, "Rule-Based")
    evaluator.print_results(rule_results, "Rule-Based Baseline")
    
    # Evaluate FSA baseline
    fsa_results = evaluator.evaluate_model(fsa_baseline, dataset, "FSA")
    evaluator.print_results(fsa_results, "Finite State Automaton (FSA)")
    
    # Compare all models
    all_results = {
        "PDA (Ours)": pda_results,
        "Rule-Based": rule_results,
        "FSA": fsa_results
    }
    evaluator.compare_models(all_results)
    
    # Save results
    results_file = dataset_path.replace('data/', 'results/').replace('.json', '_results.json')
    with open(results_file, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for model, metrics in all_results.items():
            serializable_results[model] = {
                'accuracy': float(metrics['accuracy']),
                'macro_precision': float(metrics['macro_precision']),
                'macro_recall': float(metrics['macro_recall']),
                'macro_f1': float(metrics['macro_f1']),
                'avg_inference_time_ms': float(metrics.get('avg_inference_time_ms', 0)),
                'class_metrics': {k: {
                    'precision': float(v['precision']),
                    'recall': float(v['recall']),
                    'f1_score': float(v['f1_score']),
                    'support': int(v['support'])
                } for k, v in metrics['class_metrics'].items()}
            }
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    # Run experiments
    dataset_path = "/home/claude/av_block_pda_project/data/synthetic_ecg_dataset.json"
    results = run_experiments(dataset_path)
