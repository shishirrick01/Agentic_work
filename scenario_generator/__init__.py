"""
Scenario Generator Module

Generates unprecedented supply chain disruption scenarios using LLMs.
Combines macro-economic data, climate reports, and geopolitical events
to create multi-node disruption scenarios.
"""

from .generator import ScenarioGenerator
from .prompt_templates import ScenarioPromptTemplates

__all__ = ['ScenarioGenerator', 'ScenarioPromptTemplates']
