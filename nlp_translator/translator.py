"""
Scenario Translator

Converts qualitative Black Swan scenario descriptions into quantitative
parameters suitable for supply chain modeling and MCDM evaluation.
"""

import re
from typing import Dict, List, Tuple, Optional
from loguru import logger


class ScenarioTranslator:
    """
    Translates text-based scenarios into quantitative disruption parameters.
    
    Extracts metrics like:
    - Lead time increases (percentage)
    - Cost increases (percentage)
    - Capacity reductions (percentage)
    - Demand shocks (percentage)
    - Reliability scores (0-1)
    """
    
    # Keyword mappings to quantitative impacts
    DISRUPTION_IMPACTS = {
        # Cyber attacks
        "cyberattack": {"lead_time": 0.35, "cost": 0.20, "capacity": 0.40},
        "ransomware": {"lead_time": 0.50, "cost": 0.30, "capacity": 0.60},
        "data breach": {"lead_time": 0.15, "cost": 0.25, "capacity": 0.10},
        
        # Natural disasters
        "earthquake": {"lead_time": 0.60, "cost": 0.40, "capacity": 0.70},
        "flood": {"lead_time": 0.45, "cost": 0.35, "capacity": 0.50},
        "hurricane": {"lead_time": 0.50, "cost": 0.40, "capacity": 0.55},
        "wildfire": {"lead_time": 0.40, "cost": 0.30, "capacity": 0.45},
        
        # Geopolitical
        "sanction": {"lead_time": 0.25, "cost": 0.35, "capacity": 0.30},
        "trade war": {"lead_time": 0.30, "cost": 0.40, "capacity": 0.25},
        "embargo": {"lead_time": 0.50, "cost": 0.45, "capacity": 0.60},
        "political instability": {"lead_time": 0.35, "cost": 0.30, "capacity": 0.35},
        
        # Infrastructure
        "port closure": {"lead_time": 0.55, "cost": 0.40, "capacity": 0.50},
        "infrastructure failure": {"lead_time": 0.45, "cost": 0.35, "capacity": 0.55},
        "power outage": {"lead_time": 0.30, "cost": 0.20, "capacity": 0.65},
        
        # Health
        "pandemic": {"lead_time": 0.40, "cost": 0.35, "capacity": 0.45},
        "quarantine": {"lead_time": 0.35, "cost": 0.25, "capacity": 0.40},
        
        # Resource
        "shortage": {"lead_time": 0.40, "cost": 0.50, "capacity": 0.35},
        "scarcity": {"lead_time": 0.35, "cost": 0.45, "capacity": 0.30},
        
        # Labor
        "strike": {"lead_time": 0.45, "cost": 0.30, "capacity": 0.55},
        "labor shortage": {"lead_time": 0.30, "cost": 0.25, "capacity": 0.35},
    }
    
    SEVERITY_MULTIPLIERS = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5,
        "extreme": 2.0
    }
    
    DURATION_FACTORS = {
        "short_term": (1, 7),      # days
        "medium_term": (8, 30),
        "long_term": (31, 90),
        "extended": (91, 365)
    }
    
    def __init__(self, use_ml_enhancement: bool = False):
        """
        Initialize the translator.
        
        Args:
            use_ml_enhancement: Whether to use ML models for better extraction
        """
        self.use_ml_enhancement = use_ml_enhancement
        logger.info("ScenarioTranslator initialized")
    
    def translate_scenario(self, scenario: Dict) -> Dict:
        """
        Translate a single scenario into quantitative parameters.
        
        Args:
            scenario: Scenario dictionary with description and metadata
            
        Returns:
            Dictionary with quantitative disruption parameters
        """
        description = scenario.get('description', '').lower()
        title = scenario.get('title', '').lower()
        severity = scenario.get('severity', 'Medium').lower()
        duration_days = scenario.get('estimated_duration_days', 30)
        disruption_types = scenario.get('disruption_types', [])
        
        # Base impact extraction
        base_impacts = self._extract_base_impacts(description, title, disruption_types)
        
        # Apply severity multiplier
        severity_mult = self.SEVERITY_MULTIPLIERS.get(severity, 1.0)
        
        # Calculate final impacts
        lead_time_increase = min(0.95, base_impacts['lead_time'] * severity_mult)
        cost_increase = min(0.95, base_impacts['cost'] * severity_mult)
        capacity_reduction = min(0.95, base_impacts['capacity'] * severity_mult)
        
        # Duration-based adjustments
        duration_factor = self._get_duration_factor(duration_days)
        
        # Calculate derived metrics
        reliability_score = max(0.05, 1.0 - (capacity_reduction * 0.8))
        recovery_time = duration_days * (1 + lead_time_increase * 0.5)
        total_cost_impact = cost_increase * (1 + duration_factor * 0.2)
        
        # Create quantitative parameter set
        quantitative_params = {
            'scenario_id': scenario.get('id'),
            
            # Primary impacts (percentages as decimals)
            'lead_time_increase_pct': round(lead_time_increase, 4),
            'cost_increase_pct': round(cost_increase, 4),
            'capacity_reduction_pct': round(capacity_reduction, 4),
            
            # Derived metrics
            'reliability_score': round(reliability_score, 4),
            'recovery_time_days': round(recovery_time, 2),
            'total_cost_impact': round(total_cost_impact, 4),
            
            # Temporal factors
            'duration_days': duration_days,
            'duration_category': self._categorize_duration(duration_days),
            'duration_factor': round(duration_factor, 4),
            
            # Severity and complexity
            'severity_level': severity,
            'severity_multiplier': severity_mult,
            'num_affected_nodes': len(scenario.get('affected_nodes', [])),
            'complexity_score': self._calculate_complexity(scenario),
            
            # Original metadata
            'affected_nodes': scenario.get('affected_nodes', []),
            'disruption_types': disruption_types,
            'root_causes': scenario.get('root_causes', [])
        }
        
        logger.debug(f"Translated scenario {quantitative_params['scenario_id']}")
        return quantitative_params
    
    def translate_scenarios_batch(self, scenarios: List[Dict]) -> List[Dict]:
        """
        Translate multiple scenarios in batch.
        
        Args:
            scenarios: List of scenario dictionaries
            
        Returns:
            List of quantitative parameter dictionaries
        """
        logger.info(f"Translating {len(scenarios)} scenarios...")
        translated = []
        
        for scenario in scenarios:
            try:
                params = self.translate_scenario(scenario)
                translated.append(params)
            except Exception as e:
                logger.error(f"Failed to translate scenario {scenario.get('id')}: {e}")
                continue
        
        logger.success(f"Successfully translated {len(translated)}/{len(scenarios)} scenarios")
        return translated
    
    def _extract_base_impacts(
        self,
        description: str,
        title: str,
        disruption_types: List[str]
    ) -> Dict[str, float]:
        """Extract base impact values from text."""
        combined_text = f"{title} {description}"
        
        # Initialize with defaults
        impacts = {'lead_time': 0.2, 'cost': 0.15, 'capacity': 0.2}
        
        # Check disruption types first (most reliable)
        for dtype in disruption_types:
            dtype_lower = dtype.lower()
            for keyword, impact in self.DISRUPTION_IMPACTS.items():
                if keyword in dtype_lower:
                    impacts = self._merge_impacts(impacts, impact)
                    break
        
        # Also scan description for keywords
        for keyword, impact in self.DISRUPTION_IMPACTS.items():
            if keyword in combined_text:
                impacts = self._merge_impacts(impacts, impact, weight=0.5)
        
        return impacts
    
    def _merge_impacts(
        self,
        current: Dict[str, float],
        new: Dict[str, float],
        weight: float = 1.0
    ) -> Dict[str, float]:
        """Merge impact dictionaries with weighting."""
        merged = {}
        for key in current.keys():
            merged[key] = current[key] + (new.get(key, 0) * weight)
        return merged
    
    def _get_duration_factor(self, duration_days: int) -> float:
        """Calculate duration-based adjustment factor."""
        if duration_days <= 7:
            return 0.5
        elif duration_days <= 30:
            return 1.0
        elif duration_days <= 90:
            return 1.5
        else:
            return 2.0
    
    def _categorize_duration(self, duration_days: int) -> str:
        """Categorize duration into standard buckets."""
        if duration_days <= 7:
            return "short_term"
        elif duration_days <= 30:
            return "medium_term"
        elif duration_days <= 90:
            return "long_term"
        else:
            return "extended"
    
    def _calculate_complexity(self, scenario: Dict) -> float:
        """Calculate scenario complexity score (0-1)."""
        num_nodes = len(scenario.get('affected_nodes', []))
        num_types = len(scenario.get('disruption_types', []))
        num_causes = len(scenario.get('root_causes', []))
        
        # Normalize and combine
        node_score = min(1.0, num_nodes / 5.0)
        type_score = min(1.0, num_types / 3.0)
        cause_score = min(1.0, num_causes / 3.0)
        
        complexity = (node_score * 0.4 + type_score * 0.4 + cause_score * 0.2)
        return round(complexity, 4)
    
    def export_to_csv(self, quantitative_scenarios: List[Dict], filepath: str):
        """Export quantitative scenarios to CSV."""
        import csv
        
        if not quantitative_scenarios:
            logger.warning("No scenarios to export")
            return
        
        fieldnames = list(quantitative_scenarios[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(quantitative_scenarios)
        
        logger.info(f"Exported {len(quantitative_scenarios)} scenarios to {filepath}")
