"""
Fuzzy MCDM Module

Implements advanced Fuzzy Multi-Criteria Decision Making methods:
- Pythagorean Fuzzy TOPSIS
- q-Rung Orthopair Fuzzy VIKOR

Used to evaluate and rank supply chain resilience strategies
against AI-generated Black Swan scenarios.
"""

from .pythagorean_fuzzy_topsis import PythagoreanFuzzyTOPSIS
from .q_rung_orthopair_vikor import QRungOrthopairVIKOR
from .strategy_evaluator import StrategyEvaluator

__all__ = [
    'PythagoreanFuzzyTOPSIS',
    'QRungOrthopairVIKOR',
    'StrategyEvaluator'
]
