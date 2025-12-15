"""Attention module package."""

from .pipeline import (
    AttentionConfig,
    StatArbModel,
    StatArbRunner,
    StatArbTrainer,
)
from .cache import ResidualCache
from .data import FeatureRepository, FeatureGroup, FactorBatch, FactorResult
from .writer import FactorResultWriter

__all__ = [
    "AttentionConfig",
    "StatArbModel",
    "StatArbRunner",
    "StatArbTrainer",
    "ResidualCache",
    "FeatureRepository",
    "FeatureGroup",
    "FactorBatch",
    "FactorResult",
    "FactorResultWriter",
]
