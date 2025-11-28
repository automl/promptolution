"""Module for LLM predictors."""

from promptolution.predictors.first_occurrence_predictor import FirstOccurrencePredictor
from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor

__all__ = [
    "FirstOccurrencePredictor",
    "MarkerBasedPredictor",
]
