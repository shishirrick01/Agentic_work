"""
NLP Translator Module

Translates qualitative scenario descriptions into quantitative parameters
for supply chain disruption modeling.
"""

from .translator import ScenarioTranslator
from .parameter_extractor import ParameterExtractor

__all__ = ['ScenarioTranslator', 'ParameterExtractor']
