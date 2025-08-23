#!/usr/bin/env python3
"""
ModernBERT Trainer for Data Identifier Classification
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from .training_config_loader import ModernBERTTrainingConfig


class TokenClassificationDataset(Dataset):
    """Dataset for token classification"""

    def __init__(
        self, samples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 128
    ):
        """
        Initialize dataset.

        Args:
            samples: List of training samples
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get input_ids and labels
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Truncate or pad to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Pad sequences
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend(
                [-100] * padding_length
            )  # -100 is ignored in loss computation

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ModernBERTTrainer:
    """ModernBERT trainer for data identifier classification"""

    def __init__(self, config: ModernBERTTrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = self._setup_logging()

        # Set random seeds for reproducibility
        set_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.logger.info(f"Random seed set to {config.seed}")

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(__name__)
        return logger

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load and split training data.

        Returns:
            Tuple of (train_samples, val_samples)
        """
        self.logger.info(
            f"Loading training data from {self.config.data.train_data_path}"
        )

        samples = []
        with open(self.config.data.train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    samples.append(sample)

        self.logger.info(f"Loaded {len(samples)} samples")

        # Split into train and validation
        if self.config.data.validation_split > 0:
            train_samples, val_samples = train_test_split(
                samples,
                test_size=self.config.data.validation_split,
                random_state=self.config.data.shuffle_seed,
                stratify=[
                    1 if sample.get("expected_identifiers", []) else 0
                    for sample in samples
                ],
            )
            self.logger.info(
                f"Split data: {len(train_samples)} train, {len(val_samples)} validation"
            )
        else:
            train_samples = samples
            val_samples = []
            self.logger.info(
                f"Using all {len(train_samples)} samples for training (no validation)"
            )

        return train_samples, val_samples

    def initialize_model(self):
        """Initialize tokenizer and model"""
        self.logger.info(f"Loading tokenizer and model: {self.config.model.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_name)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model.model_name,
            num_labels=self.config.model.num_labels,
            hidden_dropout_prob=self.config.model.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.model.attention_probs_dropout_prob,
        )

        # Resize token embeddings if necessary
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.logger.info(
            f"Model loaded with {self.model.num_parameters():,} parameters"
        )

    def create_datasets(self, train_samples: List[Dict], val_samples: List[Dict]):
        """Create PyTorch datasets"""
        self.logger.info("Creating datasets")

        self.train_dataset = TokenClassificationDataset(
            train_samples, self.tokenizer, self.config.data.max_length
        )

        if val_samples:
            self.val_dataset = TokenClassificationDataset(
                val_samples, self.tokenizer, self.config.data.max_length
            )
        else:
            self.val_dataset = None

        self.logger.info(
            f"Created datasets: train={len(self.train_dataset)}, val={len(self.val_dataset) if self.val_dataset else 0}"
        )

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred

        # Get predictions (argmax over last dimension)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (padding token) from labels
        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:  # Ignore padding tokens
                    true_predictions.append(pred_id)
                    true_labels.append(label_id)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average="binary", pos_label=1
        )
        accuracy = accuracy_score(true_labels, true_predictions)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def create_trainer(self):
        """Create HuggingFace Trainer"""
        self.logger.info("Setting up trainer")

        # Create output directory
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.data.batch_size,
            per_device_eval_batch_size=self.config.data.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            max_grad_norm=self.config.training.max_grad_norm,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            # Evaluation and saving
            evaluation_strategy=self.config.evaluation.eval_strategy,
            eval_steps=self.config.evaluation.eval_steps
            if self.config.evaluation.eval_strategy == "steps"
            else None,
            save_strategy=self.config.evaluation.save_strategy,
            save_steps=self.config.evaluation.save_steps
            if self.config.evaluation.save_strategy == "steps"
            else None,
            save_total_limit=self.config.evaluation.save_total_limit,
            load_best_model_at_end=self.config.evaluation.load_best_model_at_end,
            metric_for_best_model=self.config.evaluation.metric_for_best_model,
            greater_is_better=self.config.evaluation.greater_is_better,
            # Logging
            logging_dir=self.config.logging.logging_dir,
            logging_steps=self.config.logging.logging_steps,
            report_to=self.config.logging.report_to,
            # Hardware optimization
            fp16=self.config.hardware.mixed_precision and torch.cuda.is_available(),
            dataloader_num_workers=self.config.hardware.dataloader_num_workers,
            dataloader_pin_memory=self.config.hardware.dataloader_pin_memory,
            gradient_checkpointing=self.config.advanced.gradient_checkpointing,
            # Hub integration
            push_to_hub=self.config.advanced.push_to_hub,
            hub_model_id=self.config.advanced.hub_model_id,
            # Other settings
            seed=self.config.seed,
            remove_unused_columns=False,
        )

        # Set up callbacks
        callbacks = []
        if self.config.training.early_stopping and self.val_dataset is not None:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=self.config.training.early_stopping_patience,
                early_stopping_threshold=self.config.training.early_stopping_threshold,
            )
            callbacks.append(early_stopping_callback)

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics if self.val_dataset else None,
            callbacks=callbacks,
        )

        self.logger.info("Trainer created successfully")

    def train(self):
        """Run training"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call create_trainer() first.")

        self.logger.info("Starting training")

        # Check for existing checkpoints
        last_checkpoint = get_last_checkpoint(self.config.training.output_dir)
        if last_checkpoint:
            self.logger.info(f"Resuming training from {last_checkpoint}")

        # Train model
        train_result = self.trainer.train(resume_from_checkpoint=last_checkpoint)

        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.training.output_dir)

        # Log training results
        self.logger.info("Training completed")
        self.logger.info(f"Training metrics: {train_result.metrics}")

        return train_result

    def evaluate(self):
        """Run evaluation on validation set"""
        if self.val_dataset is None:
            self.logger.warning("No validation dataset available")
            return None

        self.logger.info("Running evaluation")
        eval_result = self.trainer.evaluate()

        self.logger.info(f"Evaluation metrics: {eval_result}")
        return eval_result

    def run_full_training(self):
        """Run complete training pipeline"""
        try:
            # Load data
            train_samples, val_samples = self.load_data()

            # Initialize model
            self.initialize_model()

            # Create datasets
            self.create_datasets(train_samples, val_samples)

            # Create trainer
            self.create_trainer()

            # Train model
            train_result = self.train()

            # Final evaluation
            if self.val_dataset is not None:
                eval_result = self.evaluate()
            else:
                eval_result = None

            self.logger.info("Training pipeline completed successfully")

            return {
                "train_result": train_result,
                "eval_result": eval_result,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
