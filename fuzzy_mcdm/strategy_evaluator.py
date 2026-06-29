"""
Strategy Evaluator

Integrates scenario generation, translation, and MCDM evaluation
to rank supply chain resilience strategies.
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


class StrategyEvaluator:
    """
    End-to-end evaluator for supply chain resilience strategies.
    
    Combines:
    1. AI-generated Black Swan scenarios
    2. NLP-based quantitative translation
    3. Fuzzy MCDM evaluation (TOPSIS or VIKOR)
    """
    
    DEFAULT_STRATEGIES = [
        "Multi-sourcing",
        "Safety Stock Holding",
        "Nearshoring",
        "Supplier Diversification",
        "Vertical Integration",
        "Digital Twin Implementation",
        "Agile Manufacturing",
        "Strategic Partnerships"
    ]
    
    DEFAULT_CRITERIA = [
        "Cost Efficiency",
        "Response Time",
        "Flexibility",
        "Risk Mitigation",
        "Implementation Complexity",
        "Scalability"
    ]
    
    def __init__(
        self,
        mcdm_method: str = "pythagorean_topsis",
        criteria_weights: Optional[List[float]] = None
    ):
        """
        Initialize the strategy evaluator.
        
        Args:
            mcdm_method: 'pythagorean_topsis' or 'q_rung_vikor'
            criteria_weights: Optional weights for evaluation criteria
        """
        self.mcdm_method = mcdm_method
        self.criteria_weights = criteria_weights
        
        # Initialize MCDM evaluator
        if mcdm_method == "pythagorean_topsis":
            from fuzzy_mcdm import PythagoreanFuzzyTOPSIS
            self.mcdm_evaluator = PythagoreanFuzzyTOPSIS(criteria_weights)
        elif mcdm_method == "q_rung_vikor":
            from fuzzy_mcdm import QRungOrthopairVIKOR
            self.mcdm_evaluator = QRungOrthopairVIKOR(q=3, criteria_weights=criteria_weights)
        else:
            raise ValueError(f"Unknown MCDM method: {mcdm_method}")
        
        logger.info(f"StrategyEvaluator initialized with {mcdm_method}")
    
    def evaluate_strategies(
        self,
        quantitative_scenarios: List[Dict],
        strategies: Optional[List[str]] = None,
        criteria: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Evaluate resilience strategies against generated scenarios.
        
        Args:
            quantitative_scenarios: List of translated scenario parameters
            strategies: List of strategy names to evaluate
            criteria: List of evaluation criteria
            
        Returns:
            Ranked list of strategies with performance scores
        """
        if strategies is None:
            strategies = self.DEFAULT_STRATEGIES
        
        if criteria is None:
            criteria = self.DEFAULT_CRITERIA
        
        n_strategies = len(strategies)
        n_scenarios = len(quantitative_scenarios)
        n_criteria = len(criteria)
        
        logger.info(f"Evaluating {n_strategies} strategies across {n_scenarios} scenarios")
        
        # Build decision matrix
        # Shape: (strategies, scenarios, criteria, 2) for fuzzy values
        decision_matrix = np.zeros((n_strategies, n_scenarios, n_criteria, 2))
        
        for s_idx in range(n_strategies):
            for scen_idx in range(n_scenarios):
                scenario = quantitative_scenarios[scen_idx]
                
                # Calculate strategy performance for this scenario
                performance = self._calculate_strategy_performance(
                    strategies[s_idx],
                    scenario,
                    criteria
                )
                
                # Convert to Pythagorean fuzzy values
                pf_values = self._crisp_to_fuzzy(performance)
                decision_matrix[s_idx, scen_idx, :, :] = pf_values
        
        # Define criteria types (benefit or cost)
        criteria_types = self._get_criteria_types(criteria)
        
        # Run MCDM evaluation
        results = self.mcdm_evaluator.evaluate(
            decision_matrix=decision_matrix,
            criteria_types=criteria_types,
            strategies=strategies
        )
        
        logger.success("Strategy evaluation complete")
        return results
    
    def _calculate_strategy_performance(
        self,
        strategy: str,
        scenario: Dict,
        criteria: List[str]
    ) -> np.ndarray:
        """
        Calculate how well a strategy performs against a scenario.
        
        This is a simplified model. In practice, you would use domain-specific
        models or simulations to determine strategy effectiveness.
        """
        n_criteria = len(criteria)
        performance = np.zeros(n_criteria)
        
        # Extract scenario characteristics
        lead_time_impact = scenario.get('lead_time_increase_pct', 0.3)
        cost_impact = scenario.get('cost_increase_pct', 0.2)
        capacity_impact = scenario.get('capacity_reduction_pct', 0.3)
        complexity = scenario.get('complexity_score', 0.5)
        severity = scenario.get('severity_multiplier', 1.0)
        
        # Strategy effectiveness matrices (simplified expert knowledge)
        # Values represent how much each strategy mitigates each type of impact
        strategy_profiles = {
            "Multi-sourcing": {
                "lead_time": 0.7, "cost": 0.6, "flexibility": 0.8, "risk": 0.7
            },
            "Safety Stock Holding": {
                "lead_time": 0.8, "cost": 0.4, "flexibility": 0.5, "risk": 0.6
            },
            "Nearshoring": {
                "lead_time": 0.7, "cost": 0.5, "flexibility": 0.7, "risk": 0.8
            },
            "Supplier Diversification": {
                "lead_time": 0.6, "cost": 0.6, "flexibility": 0.7, "risk": 0.8
            },
            "Vertical Integration": {
                "lead_time": 0.5, "cost": 0.3, "flexibility": 0.4, "risk": 0.7
            },
            "Digital Twin Implementation": {
                "lead_time": 0.6, "cost": 0.5, "flexibility": 0.8, "risk": 0.6
            },
            "Agile Manufacturing": {
                "lead_time": 0.7, "cost": 0.5, "flexibility": 0.9, "risk": 0.6
            },
            "Strategic Partnerships": {
                "lead_time": 0.6, "cost": 0.7, "flexibility": 0.6, "risk": 0.7
            }
        }
        
        profile = strategy_profiles.get(strategy, {
            "lead_time": 0.5, "cost": 0.5, "flexibility": 0.5, "risk": 0.5
        })
        
        # Map to criteria
        for c_idx, criterion in enumerate(criteria):
            if criterion == "Cost Efficiency":
                # Higher is better - how well does strategy control costs?
                base_effectiveness = profile.get("cost", 0.5)
                performance[c_idx] = base_effectiveness * (1 - cost_impact * 0.5)
            
            elif criterion == "Response Time":
                # Higher is better - how quickly can strategy respond?
                base_effectiveness = profile.get("lead_time", 0.5)
                performance[c_idx] = base_effectiveness * (1 - lead_time_impact * 0.3)
            
            elif criterion == "Flexibility":
                # Higher is better
                base_effectiveness = profile.get("flexibility", 0.5)
                performance[c_idx] = base_effectiveness * (1 - complexity * 0.2)
            
            elif criterion == "Risk Mitigation":
                # Higher is better
                base_effectiveness = profile.get("risk", 0.5)
                performance[c_idx] = base_effectiveness * (1 - severity * 0.1)
            
            elif criterion == "Implementation Complexity":
                # Lower is better (will be marked as cost criterion)
                # Some strategies are inherently more complex
                complexity_map = {
                    "Multi-sourcing": 0.4,
                    "Safety Stock Holding": 0.3,
                    "Nearshoring": 0.7,
                    "Supplier Diversification": 0.5,
                    "Vertical Integration": 0.8,
                    "Digital Twin Implementation": 0.6,
                    "Agile Manufacturing": 0.5,
                    "Strategic Partnerships": 0.4
                }
                performance[c_idx] = complexity_map.get(strategy, 0.5)
            
            elif criterion == "Scalability":
                # Higher is better
                scalability_map = {
                    "Multi-sourcing": 0.7,
                    "Safety Stock Holding": 0.5,
                    "Nearshoring": 0.6,
                    "Supplier Diversification": 0.7,
                    "Vertical Integration": 0.4,
                    "Digital Twin Implementation": 0.8,
                    "Agile Manufacturing": 0.7,
                    "Strategic Partnerships": 0.6
                }
                performance[c_idx] = scalability_map.get(strategy, 0.5)
            
            else:
                performance[c_idx] = 0.5  # Default neutral value
        
        return performance
    
    def _crisp_to_fuzzy(self, crisp_values: np.ndarray) -> np.ndarray:
        """Convert crisp performance values to Pythagorean fuzzy numbers."""
        n_criteria = len(crisp_values)
        pf_values = np.zeros((n_criteria, 2))
        
        for i in range(n_criteria):
            val = crisp_values[i]
            # Ensure value is in [0, 1]
            val = max(0, min(1, val))
            
            # Convert to Pythagorean fuzzy number
            # membership = value, non-membership = 1 - value
            # with some uncertainty factor
            uncertainty = 0.1
            
            mu = val * (1 - uncertainty)
            nu = (1 - val) * (1 - uncertainty)
            
            # Ensure Pythagorean condition
            sum_sq = mu**2 + nu**2
            if sum_sq > 1:
                scale = np.sqrt(1 / sum_sq)
                mu *= scale
                nu *= scale
            
            pf_values[i, 0] = mu
            pf_values[i, 1] = nu
        
        return pf_values
    
    def _get_criteria_types(self, criteria: List[str]) -> List[str]:
        """Determine if each criterion is benefit or cost type."""
        types = []
        for criterion in criteria:
            if criterion in ["Implementation Complexity"]:
                types.append("cost")  # Lower is better
            else:
                types.append("benefit")  # Higher is better
        return types
    
    def generate_report(
        self,
        results: List[Dict],
        output_path: Optional[str] = None
    ) -> str:
        """Generate a human-readable report of strategy rankings."""
        report_lines = [
            "=" * 60,
            "SUPPLY CHAIN RESILIENCE STRATEGY EVALUATION REPORT",
            "=" * 60,
            "",
            f"MCDM Method: {self.mcdm_method}",
            f"Number of Strategies Evaluated: {len(results)}",
            "",
            "-" * 60,
            "STRATEGY RANKINGS",
            "-" * 60,
            ""
        ]
        
        for result in results:
            rank = result['rank']
            strategy = result['strategy']
            score = result['score']
            
            if 'S' in result:
                report_lines.append(
                    f"{rank}. {strategy:30s} Score: {score:.4f} (S={result.get('S', 0):.4f}, "
                    f"R={result.get('R', 0):.4f})"
                )
            else:
                report_lines.append(
                    f"{rank}. {strategy:30s} Score: {score:.4f}"
                )
        
        report_lines.extend([
            "",
            "-" * 60,
            "RECOMMENDATIONS",
            "-" * 60,
            "",
            f"Top Strategy: {results[0]['strategy']}",
            "",
            "The top-ranked strategy demonstrates the best overall performance",
            "across all generated Black Swan scenarios, balancing cost efficiency,",
            "response time, flexibility, and risk mitigation capabilities.",
            "",
            "=" * 60
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        print(report)
        return report
