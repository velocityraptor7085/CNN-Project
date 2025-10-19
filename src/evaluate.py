"""
Evaluation Module for CNN Image Classification

This module provides comprehensive evaluation metrics and analysis including:
1. Accuracy, Precision, Recall, F1-Score
2. Confusion Matrix
3. Per-class performance analysis
4. Error analysis

Mathematical Concepts:
- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP) - "How many predicted positives are actually positive?"
- Recall = TP / (TP + FN) - "How many actual positives did we find?"
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean

Where:
TP = True Positives (correctly predicted positive)
TN = True Negatives (correctly predicted negative)
FP = False Positives (incorrectly predicted positive)
FN = False Negatives (incorrectly predicted negative)
"""

import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_recall_fscore_support
)
from tqdm import tqdm
import json
import os


class Evaluator:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Provides:
    - Overall accuracy
    - Per-class metrics (precision, recall, F1)
    - Confusion matrix
    - Detailed classification report
    """
    
    def __init__(self, model, test_loader, device='cuda', class_names=None):
        """
        Initialize the evaluator.
        
        Args:
            model (nn.Module): Trained CNN model
            test_loader (DataLoader): Test data loader
            device (str): Device to use ('cuda' or 'cpu')
            class_names (list): List of class names for reporting
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)
        
        if class_names is None:
            # Default CIFAR-10 class names
            self.class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
    
    def evaluate(self):
        """
        Perform comprehensive evaluation on test set.
        
        Mathematical Process:
        1. Forward pass all test samples
        2. Collect predictions and ground truth labels
        3. Calculate metrics:
           - Accuracy = Correct / Total
           - Precision_c = TP_c / (TP_c + FP_c) for each class c
           - Recall_c = TP_c / (TP_c + FN_c) for each class c
           - F1_c = 2 * (Precision_c * Recall_c) / (Precision_c + Recall_c)
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Set model to evaluation mode
        # - Disables dropout
        # - Uses running statistics for batch normalization
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("Evaluating model on test set...")
        
        # No gradient computation needed for evaluation
        # Saves memory and speeds up computation
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                # Move data to device
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Get probabilities using softmax
                # Softmax Formula: P(class_i) = exp(z_i) / Σ(exp(z_j))
                # Converts raw logits to probabilities that sum to 1
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get predicted classes (argmax of logits)
                # Mathematical: predicted_class = argmax(outputs)
                _, predicted = outputs.max(1)
                
                # Store predictions and labels
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert lists to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # ====================================================================
        # CALCULATE METRICS
        # ====================================================================
        
        # Overall Accuracy
        # Mathematical Formula: Accuracy = (# correct predictions) / (# total predictions)
        # Range: [0, 1] or [0%, 100%]
        # Higher is better
        accuracy = 100.0 * np.mean(all_predictions == all_labels)
        
        # Confusion Matrix
        # A matrix where:
        # - Rows represent actual classes
        # - Columns represent predicted classes
        # - Element [i,j] = number of class i samples predicted as class j
        # 
        # Perfect prediction → diagonal matrix (all off-diagonal elements are 0)
        # 
        # Example for 3 classes:
        #              Predicted
        #              0    1    2
        # Actual  0  [100   5    3]  ← 100 correct, 5 confused with class 1, 3 with class 2
        #         1  [  2  95    8]
        #         2  [  1   4   97]
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Per-class metrics
        # precision, recall, f1, support = arrays of length num_classes
        # support = number of samples in each class
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, 
            all_predictions, 
            average=None,  # Return per-class metrics
            zero_division=0
        )
        
        # Macro-averaged metrics (unweighted average across classes)
        # Mathematical Formula: macro_avg = (1/n) * Σ(metric_i)
        # Treats all classes equally regardless of class imbalance
        # Good when all classes are equally important
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Weighted-averaged metrics (weighted by support)
        # Mathematical Formula: weighted_avg = Σ(metric_i * support_i) / Σ(support_i)
        # Accounts for class imbalance
        # Good when some classes are more common than others
        weighted_precision = np.sum(precision * support) / np.sum(support)
        weighted_recall = np.sum(recall * support) / np.sum(support)
        weighted_f1 = np.sum(f1 * support) / np.sum(support)
        
        # ====================================================================
        # COMPILE RESULTS
        # ====================================================================
        
        results = {
            'overall': {
                'accuracy': accuracy,
                'macro_precision': macro_precision * 100,
                'macro_recall': macro_recall * 100,
                'macro_f1': macro_f1 * 100,
                'weighted_precision': weighted_precision * 100,
                'weighted_recall': weighted_recall * 100,
                'weighted_f1': weighted_f1 * 100,
            },
            'per_class': {},
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probabilities.tolist()
        }
        
        # Per-class results
        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': precision[i] * 100,
                'recall': recall[i] * 100,
                'f1_score': f1[i] * 100,
                'support': int(support[i])
            }
        
        return results
    
    def print_results(self, results):
        """
        Print evaluation results in a readable format.
        
        Args:
            results (dict): Results from evaluate() method
        """
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  Accuracy: {results['overall']['accuracy']:.2f}%")
        print(f"  Macro Precision: {results['overall']['macro_precision']:.2f}%")
        print(f"  Macro Recall: {results['overall']['macro_recall']:.2f}%")
        print(f"  Macro F1-Score: {results['overall']['macro_f1']:.2f}%")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        for class_name, metrics in results['per_class'].items():
            print(f"{class_name:<12} "
                  f"{metrics['precision']:>10.2f}%  "
                  f"{metrics['recall']:>10.2f}%  "
                  f"{metrics['f1_score']:>10.2f}%  "
                  f"{metrics['support']:>8}")
        
        print("="*70)
        
        # Interpretation guide
        print("\nMetric Interpretations:")
        print("  • Accuracy: Overall correctness of predictions")
        print("  • Precision: Of all predicted positives, how many are actually positive?")
        print("  • Recall: Of all actual positives, how many did we find?")
        print("  • F1-Score: Harmonic mean of precision and recall")
        print("  • Support: Number of samples in each class")
        print("\nFormulas:")
        print("  • Precision = TP / (TP + FP)")
        print("  • Recall = TP / (TP + FN)")
        print("  • F1 = 2 * (Precision * Recall) / (Precision + Recall)")
        print("  • Accuracy = (TP + TN) / Total")
    
    def analyze_errors(self, results, num_examples=10):
        """
        Analyze misclassified examples to understand model errors.
        
        Error Analysis helps understand:
        - Which classes are confused with each other
        - What types of mistakes the model makes
        - How to improve the model
        
        Args:
            results (dict): Results from evaluate() method
            num_examples (int): Number of error examples to analyze
        
        Returns:
            dict: Error analysis results
        """
        predictions = np.array(results['predictions'])
        labels = np.array(results['labels'])
        probabilities = np.array(results['probabilities'])
        
        # Find misclassified samples
        misclassified_indices = np.where(predictions != labels)[0]
        
        print(f"\nError Analysis:")
        print(f"Total test samples: {len(labels)}")
        print(f"Misclassified samples: {len(misclassified_indices)}")
        print(f"Error rate: {len(misclassified_indices)/len(labels)*100:.2f}%")
        
        # Analyze common confusions
        confusion_counts = {}
        for idx in misclassified_indices:
            true_class = self.class_names[labels[idx]]
            pred_class = self.class_names[predictions[idx]]
            key = f"{true_class} → {pred_class}"
            confusion_counts[key] = confusion_counts.get(key, 0) + 1
        
        # Sort by frequency
        sorted_confusions = sorted(
            confusion_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("\nMost Common Confusions:")
        for confusion, count in sorted_confusions[:10]:
            print(f"  {confusion}: {count} times")
        
        # Analyze confidence of errors
        error_confidences = []
        for idx in misclassified_indices:
            pred_class = predictions[idx]
            confidence = probabilities[idx][pred_class]
            error_confidences.append(confidence)
        
        print(f"\nError Confidence Statistics:")
        print(f"  Mean confidence: {np.mean(error_confidences):.4f}")
        print(f"  Median confidence: {np.median(error_confidences):.4f}")
        print(f"  Min confidence: {np.min(error_confidences):.4f}")
        print(f"  Max confidence: {np.max(error_confidences):.4f}")
        
        print("\nInterpretation:")
        print("  • High confidence errors: Model is confidently wrong (harder to fix)")
        print("  • Low confidence errors: Model is uncertain (easier to fix)")
        
        return {
            'misclassified_count': len(misclassified_indices),
            'common_confusions': sorted_confusions[:10],
            'error_confidence_stats': {
                'mean': float(np.mean(error_confidences)),
                'median': float(np.median(error_confidences)),
                'min': float(np.min(error_confidences)),
                'max': float(np.max(error_confidences))
            }
        }
    
    def save_results(self, results, save_dir='./results/metrics'):
        """
        Save evaluation results to JSON file.
        
        Args:
            results (dict): Results from evaluate() method
            save_dir (str): Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save full results
        results_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {results_path}")
        
        # Save confusion matrix separately (easier to read)
        conf_matrix_path = os.path.join(save_dir, 'confusion_matrix.txt')
        with open(conf_matrix_path, 'w') as f:
            f.write("Confusion Matrix\n")
            f.write("================\n\n")
            f.write("Rows: Actual classes\n")
            f.write("Columns: Predicted classes\n\n")
            
            # Header
            f.write("       ")
            for name in self.class_names:
                f.write(f"{name[:8]:>8} ")
            f.write("\n")
            
            # Matrix
            conf_matrix = np.array(results['confusion_matrix'])
            for i, name in enumerate(self.class_names):
                f.write(f"{name[:6]:>6} ")
                for j in range(len(self.class_names)):
                    f.write(f"{conf_matrix[i,j]:>8} ")
                f.write("\n")
        
        print(f"Confusion matrix saved to: {conf_matrix_path}")


# Example usage
if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("See main.py for complete evaluation pipeline.")
