#!/usr/bin/env python3
"""
ModernBERT Model Evaluator - Detailed model evaluation and analysis
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data_generation.restoration_tester import restore_identifiers

from .modern_bert_trainer import TokenClassificationDataset


class ModelEvaluator:
    """Detailed model evaluation and analysis"""

    def __init__(self, model_path: str, test_data_path: str):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model
            test_data_path: Path to test data (JSONL format)
        """
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)

        # Load model and tokenizer
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

        # Load test data
        print(f"Loading test data from {test_data_path}")
        self.test_samples = self._load_test_data()

        print(f"Loaded {len(self.test_samples)} test samples")

    def _load_test_data(self) -> List[Dict]:
        """Load test data from JSONL file"""
        samples = []
        with open(self.test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    samples.append(sample)
        return samples

    def evaluate_token_classification(self, max_length: int = 128) -> Dict:
        """
        Evaluate token classification performance.

        Args:
            max_length: Maximum sequence length

        Returns:
            Detailed evaluation results
        """
        print("Evaluating token classification performance...")

        # Create dataset
        dataset = TokenClassificationDataset(
            self.test_samples, self.tokenizer, max_length
        )

        all_predictions = []
        all_labels = []
        sample_results = []

        # Evaluate each sample
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                # Get model predictions
                input_ids = sample["input_ids"].unsqueeze(0)
                attention_mask = sample["attention_mask"].unsqueeze(0)
                labels = sample["labels"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0)

                # Filter out padding tokens (-100 labels)
                valid_mask = labels != -100
                valid_predictions = predictions[valid_mask].cpu().numpy()
                valid_labels = labels[valid_mask].cpu().numpy()

                all_predictions.extend(valid_predictions)
                all_labels.extend(valid_labels)

                # Store sample result
                sample_results.append(
                    {
                        "sample_idx": i,
                        "predictions": valid_predictions.tolist(),
                        "labels": valid_labels.tolist(),
                        "original_text": self.test_samples[i].get("original_text", ""),
                    }
                )

        # Calculate overall metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average="binary", pos_label=1
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(all_labels, all_predictions, average=None)
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Classification report
        class_report = classification_report(
            all_labels,
            all_predictions,
            target_names=["Non-Identifier", "Data-Identifier"],
            output_dict=True,
        )

        results = {
            "token_classification": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support[1])
                if isinstance(support, np.ndarray)
                else int(support),
                "per_class_metrics": {
                    "precision": precision_per_class.tolist(),
                    "recall": recall_per_class.tolist(),
                    "f1": f1_per_class.tolist(),
                    "support": support_per_class.tolist(),
                },
                "confusion_matrix": cm.tolist(),
                "classification_report": class_report,
            },
            "sample_results": sample_results,
        }

        return results

    def evaluate_identifier_restoration(self) -> Dict:
        """
        Evaluate identifier restoration performance.

        Returns:
            Identifier restoration results
        """
        print("Evaluating identifier restoration performance...")

        restoration_results = []
        total_exact_matches = 0
        total_precision = 0
        total_recall = 0

        with torch.no_grad():
            for i, test_sample in enumerate(self.test_samples):
                # Get tokens and expected identifiers
                tokens = test_sample.get("tokens", [])
                expected_identifiers = test_sample.get("expected_identifiers", [])

                if not tokens:
                    continue

                # Get model predictions
                input_ids = torch.tensor(
                    test_sample["input_ids"], dtype=torch.long
                ).unsqueeze(0)
                outputs = self.model(input_ids=input_ids)
                predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0)

                # Truncate predictions to match token length
                predictions = predictions[: len(tokens)].cpu().numpy().tolist()

                # Restore identifiers from predictions
                restored_identifiers = restore_identifiers(tokens, predictions)

                # Calculate sample metrics
                restored_set = set(restored_identifiers)
                expected_set = set(expected_identifiers)
                intersection = restored_set & expected_set

                precision = (
                    len(intersection) / len(restored_set) if restored_set else 0.0
                )
                recall = (
                    len(intersection) / len(expected_set)
                    if expected_set
                    else (1.0 if not restored_set else 0.0)
                )
                exact_match = restored_set == expected_set

                total_exact_matches += exact_match
                total_precision += precision
                total_recall += recall

                restoration_results.append(
                    {
                        "sample_idx": i,
                        "expected": expected_identifiers,
                        "restored": restored_identifiers,
                        "exact_match": exact_match,
                        "precision": precision,
                        "recall": recall,
                        "original_text": test_sample.get("original_text", "")[:100]
                        + "...",
                    }
                )

        # Overall restoration metrics
        num_samples = len(restoration_results)
        if num_samples > 0:
            avg_precision = total_precision / num_samples
            avg_recall = total_recall / num_samples
            exact_match_rate = total_exact_matches / num_samples
            f1_score = (
                2 * avg_precision * avg_recall / (avg_precision + avg_recall)
                if (avg_precision + avg_recall) > 0
                else 0.0
            )
        else:
            avg_precision = avg_recall = exact_match_rate = f1_score = 0.0

        return {
            "identifier_restoration": {
                "exact_match_rate": exact_match_rate,
                "average_precision": avg_precision,
                "average_recall": avg_recall,
                "f1_score": f1_score,
                "total_samples": num_samples,
            },
            "sample_results": restoration_results,
        }

    def analyze_errors(self, results: Dict) -> Dict:
        """
        Analyze common error patterns.

        Args:
            results: Results from evaluate_token_classification

        Returns:
            Error analysis results
        """
        print("Analyzing error patterns...")

        sample_results = results["sample_results"]

        # Error pattern analysis
        false_positives = []  # Predicted as identifier but actually not
        false_negatives = []  # Should be identifier but predicted as not

        for sample in sample_results:
            predictions = sample["predictions"]
            labels = sample["labels"]

            for i, (pred, label) in enumerate(zip(predictions, labels)):
                if pred == 1 and label == 0:  # False positive
                    false_positives.append(
                        {
                            "sample_idx": sample["sample_idx"],
                            "token_idx": i,
                            "text": sample["original_text"],
                        }
                    )
                elif pred == 0 and label == 1:  # False negative
                    false_negatives.append(
                        {
                            "sample_idx": sample["sample_idx"],
                            "token_idx": i,
                            "text": sample["original_text"],
                        }
                    )

        return {
            "error_analysis": {
                "false_positives": len(false_positives),
                "false_negatives": len(false_negatives),
                "false_positive_examples": false_positives[:10],  # First 10 examples
                "false_negative_examples": false_negatives[:10],  # First 10 examples
            }
        }

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")

    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 50)

        # Token classification metrics
        token_metrics = results["token_classification"]
        print("\n📊 Token Classification Performance:")
        print(f"  Accuracy:  {token_metrics['accuracy']:.4f}")
        print(f"  Precision: {token_metrics['precision']:.4f}")
        print(f"  Recall:    {token_metrics['recall']:.4f}")
        print(f"  F1 Score:  {token_metrics['f1']:.4f}")

        # Per-class breakdown
        print("\n📈 Per-Class Metrics:")
        per_class = token_metrics["per_class_metrics"]
        classes = ["Non-Identifier", "Data-Identifier"]
        for i, class_name in enumerate(classes):
            print(f"  {class_name}:")
            print(f"    Precision: {per_class['precision'][i]:.4f}")
            print(f"    Recall:    {per_class['recall'][i]:.4f}")
            print(f"    F1 Score:  {per_class['f1'][i]:.4f}")
            print(f"    Support:   {per_class['support'][i]}")

        # Identifier restoration metrics
        if "identifier_restoration" in results:
            resto_metrics = results["identifier_restoration"]
            print("\n🔍 Identifier Restoration Performance:")
            print(f"  Exact Match Rate: {resto_metrics['exact_match_rate']:.4f}")
            print(f"  Average Precision: {resto_metrics['average_precision']:.4f}")
            print(f"  Average Recall:    {resto_metrics['average_recall']:.4f}")
            print(f"  F1 Score:         {resto_metrics['f1_score']:.4f}")

        # Error analysis
        if "error_analysis" in results:
            error_metrics = results["error_analysis"]
            print("\n❌ Error Analysis:")
            print(f"  False Positives: {error_metrics['false_positives']}")
            print(f"  False Negatives: {error_metrics['false_negatives']}")

        print("\n" + "=" * 50)

    def run_full_evaluation(
        self, max_length: int = 128, output_path: str = None
    ) -> Dict:
        """
        Run complete model evaluation.

        Args:
            max_length: Maximum sequence length
            output_path: Path to save results (optional)

        Returns:
            Complete evaluation results
        """
        print("Starting comprehensive model evaluation...")

        # Token classification evaluation
        token_results = self.evaluate_token_classification(max_length)

        # Identifier restoration evaluation
        restoration_results = self.evaluate_identifier_restoration()

        # Error analysis
        error_results = self.analyze_errors(token_results)

        # Combine results
        complete_results = {
            **token_results,
            **restoration_results,
            **error_results,
            "model_info": {
                "model_path": str(self.model_path),
                "test_data_path": str(self.test_data_path),
                "num_test_samples": len(self.test_samples),
                "max_length": max_length,
            },
        }

        # Print summary
        self.print_summary(complete_results)

        # Save results if requested
        if output_path:
            self.save_results(complete_results, output_path)

        return complete_results
