"""
Pythagorean Fuzzy TOPSIS Implementation

TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
using Pythagorean Fuzzy Sets for handling uncertainty in decision making.
"""

import numpy as np
from typing import List, Dict, Tuple
from loguru import logger


class PythagoreanFuzzyTOPSIS:
    """
    Pythagorean Fuzzy TOPSIS method for multi-criteria decision making.
    
    Pythagorean fuzzy sets extend intuitionistic fuzzy sets by allowing
    the sum of squares of membership and non-membership degrees to be ≤ 1,
    providing more flexibility in modeling uncertainty.
    """
    
    def __init__(self, criteria_weights: List[float] = None):
        """
        Initialize the Pythagorean Fuzzy TOPSIS evaluator.
        
        Args:
            criteria_weights: Weights for each criterion (will be normalized)
        """
        self.criteria_weights = np.array(criteria_weights) if criteria_weights else None
        logger.info("PythagoreanFuzzyTOPSIS initialized")
    
    def evaluate(
        self,
        decision_matrix: np.ndarray,
        criteria_types: List[str],
        strategies: List[str]
    ) -> List[Dict]:
        """
        Evaluate strategies using Pythagorean Fuzzy TOPSIS.
        
        Args:
            decision_matrix: 3D array of shape (strategies, scenarios, criteria)
                           containing Pythagorean fuzzy values (membership, non-membership)
            criteria_types: List of 'benefit' or 'cost' for each criterion
            strategies: List of strategy names
            
        Returns:
            Ranked list of strategies with scores and rankings
        """
        n_strategies, n_scenarios, n_criteria, _ = decision_matrix.shape
        
        # Normalize weights if not provided
        if self.criteria_weights is None:
            self.criteria_weights = np.ones(n_criteria) / n_criteria
        
        logger.info(f"Evaluating {n_strategies} strategies across {n_scenarios} scenarios")
        
        # Calculate Pythagorean fuzzy scores for each strategy-scenario pair
        scores = []
        for s_idx in range(n_strategies):
            strategy_scores = []
            for scen_idx in range(n_scenarios):
                # Extract Pythagorean fuzzy values for this strategy-scenario combination
                pf_values = decision_matrix[s_idx, scen_idx, :, :]
                
                # Calculate score for this scenario
                scenario_score = self._calculate_pf_score(pf_values, criteria_types)
                strategy_scores.append(scenario_score)
            
            # Average score across all scenarios
            avg_score = np.mean(strategy_scores)
            scores.append({
                'strategy': strategies[s_idx],
                'score': avg_score,
                'scenario_scores': strategy_scores
            })
        
        # Rank strategies by score (higher is better)
        scores.sort(key=lambda x: x['score'], reverse=True)
        for i, item in enumerate(scores):
            item['rank'] = i + 1
        
        logger.success(f"TOPSIS evaluation complete. Best strategy: {scores[0]['strategy']}")
        return scores
    
    def _calculate_pf_score(
        self,
        pf_values: np.ndarray,
        criteria_types: List[str]
    ) -> float:
        """
        Calculate Pythagorean fuzzy score for a single alternative.
        
        Args:
            pf_values: Array of shape (criteria, 2) with [membership, non-membership]
            criteria_types: List of 'benefit' or 'cost'
            
        Returns:
            Crisp score value
        """
        n_criteria = len(criteria_types)
        
        # Calculate Pythagorean fuzzy positive and negative ideal solutions
        pfs_positive = []
        pfs_negative = []
        
        for c_idx in range(n_criteria):
            membership = pf_values[c_idx, 0]
            non_membership = pf_values[c_idx, 1]
            
            if criteria_types[c_idx] == 'benefit':
                # For benefit criteria: max membership, min non-membership
                pfs_positive.append([membership, non_membership])
                pfs_negative.append([non_membership, membership])
            else:
                # For cost criteria: min membership, max non-membership
                pfs_positive.append([non_membership, membership])
                pfs_negative.append([membership, non_membership])
        
        pfs_positive = np.array(pfs_positive)
        pfs_negative = np.array(pfs_negative)
        
        # Calculate distances to ideal solutions
        dist_positive = self._pf_distance(pf_values, pfs_positive)
        dist_negative = self._pf_distance(pf_values, pfs_negative)
        
        # Calculate relative closeness to ideal solution
        if dist_positive + dist_negative == 0:
            return 0.5
        
        closeness = dist_negative / (dist_positive + dist_negative)
        return closeness
    
    def _pf_distance(self, pf1: np.ndarray, pf2: np.ndarray) -> float:
        """
        Calculate distance between two Pythagorean fuzzy sets.
        
        Uses the normalized Euclidean distance for Pythagorean fuzzy sets.
        """
        n = len(pf1)
        sum_sq = 0.0
        
        for i in range(n):
            # Distance formula for Pythagorean fuzzy sets
            diff_mem = pf1[i, 0] - pf2[i, 0]
            diff_nonmem = pf1[i, 1] - pf2[i, 1]
            diff_indet = np.sqrt(1 - pf1[i, 0]**2 - pf1[i, 1]**2) - \
                        np.sqrt(1 - pf2[i, 0]**2 - pf2[i, 1]**2)
            
            sum_sq += diff_mem**2 + diff_nonmem**2 + diff_indet**2
        
        return np.sqrt(sum_sq / (2 * n))
    
    def convert_to_pythagorean_fuzzy(
        self,
        crisp_values: np.ndarray,
        uncertainty_factor: float = 0.1
    ) -> np.ndarray:
        """
        Convert crisp values to Pythagorean fuzzy numbers.
        
        Args:
            crisp_values: Array of crisp numerical values
            uncertainty_factor: Factor to introduce uncertainty (0-1)
            
        Returns:
            Array of Pythagorean fuzzy values [membership, non-membership]
        """
        # Normalize crisp values to [0, 1]
        min_val = np.min(crisp_values)
        max_val = np.max(crisp_values)
        
        if max_val - min_val == 0:
            normalized = np.ones_like(crisp_values) * 0.5
        else:
            normalized = (crisp_values - min_val) / (max_val - min_val)
        
        # Convert to Pythagorean fuzzy numbers
        # membership = normalized value with some uncertainty
        # non-membership = 1 - membership with some uncertainty
        pf_shape = crisp_values.shape + (2,)
        pf_values = np.zeros(pf_shape)
        
        pf_values[..., 0] = normalized * (1 - uncertainty_factor)
        pf_values[..., 1] = (1 - normalized) * (1 - uncertainty_factor)
        
        # Ensure Pythagorean condition: mu^2 + nu^2 <= 1
        sum_sq = pf_values[..., 0]**2 + pf_values[..., 1]**2
        mask = sum_sq > 1
        if np.any(mask):
            scale_factor = np.sqrt(1 / sum_sq[mask])
            pf_values[mask, 0] *= scale_factor
            pf_values[mask, 1] *= scale_factor
        
        return pf_values
