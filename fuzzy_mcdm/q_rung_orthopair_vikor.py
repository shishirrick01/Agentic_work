"""
q-Rung Orthopair Fuzzy VIKOR Implementation

VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje)
using q-Rung Orthopair Fuzzy Sets for advanced uncertainty modeling.
"""

import numpy as np
from typing import List, Dict, Tuple
from loguru import logger


class QRungOrthopairVIKOR:
    """
    q-Rung Orthopair Fuzzy VIKOR method for multi-criteria decision making.
    
    q-Rung orthopair fuzzy sets generalize both intuitionistic and Pythagorean
    fuzzy sets by allowing mu^q + nu^q <= 1, where q >= 1 is a parameter
    controlling the level of uncertainty that can be modeled.
    """
    
    def __init__(self, q: int = 3, criteria_weights: List[float] = None):
        """
        Initialize the q-Rung Orthopair Fuzzy VIKOR evaluator.
        
        Args:
            q: The rung parameter (q >= 1). Higher q allows more uncertainty.
            criteria_weights: Weights for each criterion (will be normalized)
        """
        if q < 1:
            raise ValueError("Parameter q must be >= 1")
        self.q = q
        self.criteria_weights = np.array(criteria_weights) if criteria_weights else None
        logger.info(f"QRungOrthopairVIKOR initialized with q={q}")
    
    def evaluate(
        self,
        decision_matrix: np.ndarray,
        criteria_types: List[str],
        strategies: List[str],
        v: float = 0.5
    ) -> List[Dict]:
        """
        Evaluate strategies using q-Rung Orthopair Fuzzy VIKOR.
        
        Args:
            decision_matrix: 4D array of shape (strategies, scenarios, criteria, 2)
                           containing q-ROF values [membership, non-membership]
            criteria_types: List of 'benefit' or 'cost' for each criterion
            strategies: List of strategy names
            v: Weight for the strategy of maximum group utility (0 <= v <= 1)
            
        Returns:
            Ranked list of strategies with Q, S, and R values
        """
        n_strategies, n_scenarios, n_criteria, _ = decision_matrix.shape
        
        # Normalize weights if not provided
        if self.criteria_weights is None:
            self.criteria_weights = np.ones(n_criteria) / n_criteria
        
        logger.info(f"Evaluating {n_strategies} strategies with q-ROF VIKOR (q={self.q})")
        
        all_results = []
        
        # Process each scenario
        for scen_idx in range(n_scenarios):
            scenario_matrix = decision_matrix[:, scen_idx, :, :]
            
            # Calculate score function for q-ROF numbers
            scores = self._calculate_qrof_scores(scenario_matrix)
            
            # Find ideal solutions
            best_values = []
            worst_values = []
            
            for c_idx in range(n_criteria):
                col_values = scores[:, c_idx]
                if criteria_types[c_idx] == 'benefit':
                    best_values.append(np.max(col_values))
                    worst_values.append(np.min(col_values))
                else:
                    best_values.append(np.min(col_values))
                    worst_values.append(np.max(col_values))
            
            best_values = np.array(best_values)
            worst_values = np.array(worst_values)
            
            # Calculate S (group utility) and R (individual regret)
            S_values = []
            R_values = []
            
            for s_idx in range(n_strategies):
                s_val = 0.0
                r_val = 0.0
                
                for c_idx in range(n_criteria):
                    # Normalized distance from ideal
                    if worst_values[c_idx] - best_values[c_idx] == 0:
                        norm_dist = 0
                    else:
                        norm_dist = abs(best_values[c_idx] - scores[s_idx, c_idx]) / \
                                   abs(worst_values[c_idx] - best_values[c_idx])
                    
                    weighted_dist = self.criteria_weights[c_idx] * norm_dist
                    s_val += weighted_dist
                    r_val = max(r_val, weighted_dist)
                
                S_values.append(s_val)
                R_values.append(r_val)
            
            S_values = np.array(S_values)
            R_values = np.array(R_values)
            
            # Calculate Q (compromise ranking)
            S_min, S_max = np.min(S_values), np.max(S_values)
            R_min, R_max = np.min(R_values), np.max(R_values)
            
            Q_values = []
            for i in range(n_strategies):
                s_term = v * (S_values[i] - S_min) / (S_max - S_min) if S_max != S_min else 0
                r_term = (1 - v) * (R_values[i] - R_min) / (R_max - R_min) if R_max != R_min else 0
                Q_values.append(s_term + r_term)
            
            Q_values = np.array(Q_values)
            
            # Store results for this scenario
            for s_idx in range(n_strategies):
                result = {
                    'strategy': strategies[s_idx],
                    'scenario_idx': scen_idx,
                    'S': S_values[s_idx],
                    'R': R_values[s_idx],
                    'Q': Q_values[s_idx]
                }
                all_results.append(result)
        
        # Aggregate across scenarios
        aggregated = {}
        for result in all_results:
            key = result['strategy']
            if key not in aggregated:
                aggregated[key] = {'S': [], 'R': [], 'Q': []}
            aggregated[key]['S'].append(result['S'])
            aggregated[key]['R'].append(result['R'])
            aggregated[key]['Q'].append(result['Q'])
        
        final_scores = []
        for strategy, values in aggregated.items():
            avg_Q = np.mean(values['Q'])
            avg_S = np.mean(values['S'])
            avg_R = np.mean(values['R'])
            
            final_scores.append({
                'strategy': strategy,
                'score': avg_Q,  # Lower Q is better in VIKOR
                'S': avg_S,
                'R': avg_R,
                'Q': avg_Q
            })
        
        # Rank strategies (lower Q is better)
        final_scores.sort(key=lambda x: x['score'])
        for i, item in enumerate(final_scores):
            item['rank'] = i + 1
        
        logger.success(f"VIKOR evaluation complete. Best strategy: {final_scores[0]['strategy']}")
        return final_scores
    
    def _calculate_qrof_scores(self, qrof_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate score function for q-ROF numbers.
        
        Score function: S(mu, nu) = mu^q - nu^q
        
        Args:
            qrof_matrix: Array of shape (alternatives, criteria, 2)
            
        Returns:
            Score matrix of shape (alternatives, criteria)
        """
        membership = qrof_matrix[..., 0]
        non_membership = qrof_matrix[..., 1]
        
        scores = membership**self.q - non_membership**self.q
        
        # Normalize to [0, 1]
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score > 0:
            scores = (scores - min_score) / (max_score - min_score)
        
        return scores
    
    def convert_to_qrof(
        self,
        crisp_values: np.ndarray,
        uncertainty_factor: float = 0.1
    ) -> np.ndarray:
        """
        Convert crisp values to q-rung orthopair fuzzy numbers.
        
        Args:
            crisp_values: Array of crisp numerical values
            uncertainty_factor: Factor to introduce uncertainty (0-1)
            
        Returns:
            Array of q-ROF values [membership, non-membership]
        """
        # Normalize crisp values to [0, 1]
        min_val = np.min(crisp_values)
        max_val = np.max(crisp_values)
        
        if max_val - min_val == 0:
            normalized = np.ones_like(crisp_values) * 0.5
        else:
            normalized = (crisp_values - min_val) / (max_val - min_val)
        
        # Convert to q-ROF numbers
        qrof_shape = crisp_values.shape + (2,)
        qrof_values = np.zeros(qrof_shape)
        
        qrof_values[..., 0] = np.power(normalized, 1/self.q) * (1 - uncertainty_factor)
        qrof_values[..., 1] = np.power(1 - normalized, 1/self.q) * (1 - uncertainty_factor)
        
        # Ensure q-ROF condition: mu^q + nu^q <= 1
        sum_q = qrof_values[..., 0]**self.q + qrof_values[..., 1]**self.q
        mask = sum_q > 1
        if np.any(mask):
            scale_factor = np.power(1 / sum_q[mask], 1/self.q)
            qrof_values[mask, 0] *= scale_factor
            qrof_values[mask, 1] *= scale_factor
        
        return qrof_values
