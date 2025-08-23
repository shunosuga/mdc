"""
BERT Training Data Generation Module

This module provides tools for generating training data for BERT-based
data identifier detection using token-level classification.
"""

from .bert_training_data_generator import BERTTrainingDataGenerator
from .restoration_tester import (
    RestorationTester,
    extend_to_word_boundary,
    restore_identifiers,
)

__all__ = [
    "BERTTrainingDataGenerator",
    "RestorationTester",
    "restore_identifiers",
    "extend_to_word_boundary",
]
