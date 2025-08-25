"""
Text Processing Strategies for BERT Training

This module provides interfaces and implementations for text cropping
and negative sampling strategies used in BERT training data generation.
"""

from .negative_samplers import (
    BalancedNegativeSampler,
    ContextualNegativeSampler,
    NegativeSampler,
    RandomNegativeSampler,
    create_negative_sampler,
)

__all__ = [
    "NegativeSampler",
    "RandomNegativeSampler",
    "ContextualNegativeSampler",
    "BalancedNegativeSampler",
    "create_negative_sampler",
]
