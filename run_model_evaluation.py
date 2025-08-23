#!/usr/bin/env python3
"""
Model Evaluation Runner

Usage:
    python run_model_evaluation.py --model models/modernbert_data_identifier --test-data training_data/bert_samples.jsonl
    python run_model_evaluation.py --model models/modernbert_data_identifier --test-data test_data.jsonl --output evaluation_results.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path
# sys.path.append(str(Path(__file__).parent / "src"))
from src.training.model_evaluator import ModelEvaluator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ModernBERT Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_model_evaluation.py --model models/modernbert_data_identifier --test-data training_data/bert_samples.jsonl
  python run_model_evaluation.py --model models/best_model --test-data test_data.jsonl --output results.json
  python run_model_evaluation.py --model ./trained_model --test-data validation.jsonl --max-length 64
        """,
    )

    parser.add_argument(
        "--model", "-m", required=True, help="Path to trained model directory"
    )

    parser.add_argument(
        "--test-data", "-d", required=True, help="Path to test data file (JSONL format)"
    )

    parser.add_argument(
        "--output", "-o", help="Path to save evaluation results (JSON format)"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for evaluation (default: 128)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress detailed output"
    )

    args = parser.parse_args()

    try:
        # Validate paths
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model path does not exist: {model_path}")
            return 1

        test_data_path = Path(args.test_data)
        if not test_data_path.exists():
            print(f"Error: Test data file does not exist: {test_data_path}")
            return 1

        # Initialize evaluator
        if not args.quiet:
            print("=== ModernBERT Model Evaluation ===")
            print(f"Model: {args.model}")
            print(f"Test data: {args.test_data}")
            print(f"Max length: {args.max_length}")
            if args.output:
                print(f"Output file: {args.output}")
            print()

        evaluator = ModelEvaluator(
            model_path=str(model_path), test_data_path=str(test_data_path)
        )

        # Run evaluation
        results = evaluator.run_full_evaluation(
            max_length=args.max_length, output_path=args.output
        )

        if not args.quiet:
            print("\nEvaluation completed successfully!")

            if args.output:
                print(f"Detailed results saved to: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback

        if not args.quiet:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
