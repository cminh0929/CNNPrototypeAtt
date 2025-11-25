"""
TSML Module - Time Series Model Evaluation and Benchmarking.

This module provides tools for evaluating CNNProto models and comparing
them against state-of-the-art time series classification methods.
"""

from tsml.cnnproto_classifier import CNNProtoClassifier
from tsml.evaluator import evaluate_with_tsml, evaluate_without_tsml
from tsml.comparison_table import create_comparison_table, print_summary

__all__ = [
    'CNNProtoClassifier',
    'evaluate_with_tsml',
    'evaluate_without_tsml',
    'create_comparison_table',
    'print_summary',
]
