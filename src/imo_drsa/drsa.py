import itertools
import operator
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DRSA:
    """
    Dominance-Based Rough Set Approach (DRSA) for multicriteria sorting and rule induction.


    :param pareto_set: Data matrix of shape (N, n_features).
    :param decision_attribute: Decision array of length N, encoded as integers 1 to m.
    :param criteria_full: Tuple of all criterion indices.
    :param criteria_reduct: Tuple of current reduct criteria indices.
    :param N: Number of objects.
    :param n_features: Number of criteria/features.
    :param m: Number of decision classes.
    """

    def __init__(self, pareto_set:np.ndarray=None,
                 decision_attribute:np.ndarray=None,
                 criteria:Tuple=None):
        """
        :param pareto_set: NumPy array with shape (N, n_var), each row is an object, columns are criteria evaluated on that object
        :param decision_attribute: NumPy array of length N, integer‐encoded decision classes (1, ..., m)
        :param criteria: list of column indices in pareto_set
        """
        if pareto_set is not None and decision_attribute is not None and criteria is not None:
            self.fit(pareto_set, decision_attribute, criteria)


    def fit(self, pareto_set:np.ndarray,
            decision_attribute:np.ndarray,
            criteria:Tuple) -> None:
        """
        :param pareto_set: NumPy array with shape (N, n_var), each row is an object, columns are criteria evaluated on that object
        :param decision_attribute: NumPy array of length N, integer‐encoded decision classes (1, ..., m)
        :param criteria: list of column indices in pareto_set
        """
        assert pareto_set is not None, "Pareto set must not be empty."
        assert criteria is not None, "Criteria must not be provided."

        self.pareto_set = pareto_set
        self.decision_attribute = decision_attribute

        self.criteria_full = criteria

        self.N, self.n_features = pareto_set.shape
        self.m = int(decision_attribute.max())

    # ---------------------------------------------------------------------------------------------------------- #
    # Dominance‐cone computations
    # ---------------------------------------------------------------------------------------------------------- #

    def positive_cone(self, criteria: Tuple) -> np.ndarray:
        """
        Boolean mask [y, x] True if object y P-dominates x.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}

        :return: the P-dominating set of x
        """
        mask = np.ones((self.N, self.N), dtype=bool)

        for idx in criteria:
            vals = self.pareto_set[:, idx]
            mask &= vals[:, None] >= vals[None, :]

        return mask

    def negative_cone(self, criteria: Tuple) -> np.ndarray:
        """
        Boolean mask [y, x] True if object x P-dominates y.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}

        :return: the P-dominated set of x
        """
        mask = np.ones((self.N, self.N), dtype=bool)

        for idx in criteria:
            vals = self.pareto_set[:, idx]
            mask &= vals[:, None] <= vals[None, :]

        return mask

    # ---------------------------------------------------------------------------------------------------------- #
    # Rough approximations
    # ---------------------------------------------------------------------------------------------------------- #

    def lower_approx_up(self, criteria: Tuple, threshold: int) -> np.ndarray:
        """
        Lower approximation of upward union for decision >= threshold.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param threshold:int of class

        :return: np.ndarray containing the lower approximation of upward union
        """
        cone = self.positive_cone(criteria)

        return np.all(~cone | (self.decision_attribute[:, None] >= threshold), axis=0)

    def upper_approx_up(self, criteria: Tuple, threshold: int) -> np.ndarray:
        """
        Upper approximation of upward union for decision >= threshold.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param threshold:int of class

        :return: np.ndarray containing the upper approximation of upward union
         """
        cone = self.negative_cone(criteria)

        return np.any(cone & (self.decision_attribute[:, None] >= threshold), axis=0)

    def lower_approx_down(self, criteria: Tuple, threshold: int) -> np.ndarray:
        """
        Lower approximation of downward union for decision <= threshold.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param threshold:int of class

        :return: np.ndarray containing the lower approximation of downward union
        """
        cone = self.negative_cone(criteria)

        return np.all(~cone | (self.decision_attribute[:, None] <= threshold), axis=0)

    def upper_approx_down(self, criteria: Tuple, threshold: int) -> np.ndarray:
        """
        Upper approximation of downward union for decision <= threshold.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param threshold:int of class

        :return: np.ndarray containing the upper approximation of downward union
        """
        cone = self.positive_cone(criteria)

        return np.any(cone & (self.decision_attribute[:, None] <= threshold), axis=0)

    # ---------------------------------------------------------------------------------------------------------- #
    # Quality of approximation gamma_P(Cl)
    # ---------------------------------------------------------------------------------------------------------- #

    def quality(self, criteria: Tuple) -> float:
        """
        Compute the quality of approximation gamma for given criteria.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        """
        consistent_mask = np.ones(self.N, dtype=bool)

        for t in range(2, self.m + 1):
            lower = self.lower_approx_up(criteria, t)
            upper = self.upper_approx_up(criteria, t)

            boundary = upper & ~lower
            consistent_mask &= ~boundary

        return float(consistent_mask.sum()) / self.N

    # ---------------------------------------------------------------------------------------------------------- #
    # Finding reducts (brute‐force; not good for large n, use heuristic)
    # ---------------------------------------------------------------------------------------------------------- #
    def find_reducts(self) -> List[Tuple]:
        """
        Return minimal subsets of criteria preserving full quality.

        :return: list of reducts.
        """
        full_quality = self.quality(self.criteria_full)

        reducts = []

        for r in range(1, len(self.criteria_full) + 1):

            for subset in combinations(self.criteria_full, r):

                if self.quality(subset) == full_quality:

                    if not any(set(red).issubset(subset) for red in reducts):
                        reducts.append(subset)

            if reducts:
                break

        return reducts

    # ---------------------------------------------------------------------------------------------------------- #
    # Decision-rule induction
    # ---------------------------------------------------------------------------------------------------------- #

    def make_rule_description(self, profile: Dict,
                              conclusion: str,
                              support: float,
                              confidence: float,
                              kind: str,
                              direction: str) -> str:
        """
        Build human-readable rule description.

        :param profile: dict with column indices of the compared objectives and variables
        :param conclusion: conclusion of the decision
        :param support: support of the decision
        :param confidence: confidence of the decision
        :param kind: type of rule
        :param direction: direction of the rule

        :return: rule description
        """
        conds = []

        for idx, val in profile.items():
            op = ">=" if direction == 'up' else "<="
            conds.append(f"f_{idx + 1} {op} {val}")

        premise = ' AND '.join(conds)

        return (f"[{kind.upper()}] IF {premise} THEN {conclusion} (support={support:.2f}, confidence={confidence:.2f})")

    def induce_decision_rules(self, criteria: Tuple = None,
                              direction: str = 'up',
                              threshold: int = 2) -> List:
        """
        Induce certain and possible decision rules for Cl>=threshold or Cl<=threshold.
        direction: 'up' or 'down'.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param direction: str direction of union, either 'up' or 'down'.
        :param threshold:int of class

        :return: list of induced decision rules of form (profile, concl, support, confidence, kind, direction, desc)
        """
        crit = criteria or self.criteria_full

        if direction == 'up':
            lower = self.lower_approx_up(crit, threshold)
            upper = self.upper_approx_up(crit, threshold)
            comp = operator.ge
            conf_fn = lambda mask: (self.decision_attribute[mask] >= threshold).mean()
            concl = f"d >= {threshold}"

        elif direction == 'down':
            lower = self.lower_approx_down(crit, threshold)
            upper = self.upper_approx_down(crit, threshold)
            comp = operator.le
            conf_fn = lambda mask: (self.decision_attribute[mask] <= threshold).mean()
            concl = f"d <= {threshold}"

        else:
            raise ValueError("direction must be either 'up' or 'down'")

        rules = []
        for kind, indices in [('certain', np.where(lower)[0]), ('possible', np.where(upper & ~lower)[0])]:

            for idx in indices:
                profile = {i: self.pareto_set[idx, i] for i in crit}
                mask = np.ones(self.N, dtype=bool)

                for i, val in profile.items():
                    mask &= comp(self.pareto_set[:, i], val)

                support = mask.mean()
                confidence = conf_fn(mask)

                desc = self.make_rule_description(profile, concl, support, confidence, kind, direction)

                rules.append((profile, concl, support, confidence, kind, direction, desc))

        return rules

    # ---------------------------------------------------------------------------------------------------------- #
    # Association-rule mining
    # ---------------------------------------------------------------------------------------------------------- #

    def find_single_rule(self, f_i: np.ndarray,
                         f_j: np.ndarray,
                         min_support: float = 0.1,
                         min_confidence: float = 0.8) -> Dict:
        """
        Find the strongest single association rule for two objectives f_i, f_j.

        :param f_i: first objective
        :param f_j: second objective
        :param min_support: minimum support of the decision
        :param min_confidence: minimum confidence of the decision

        :return: The strongest single association rule for two objectives f_i, f_j.
        """
        best = None

        ti = np.unique(f_i)
        tj = np.unique(f_j)

        for sym_x, op_x in ((">=", np.greater_equal), ("<=", np.less_equal)):
            masks_x = {t: op_x(f_i, t) for t in ti}

            for sym_y, op_y in ((">=", np.greater_equal), ("<=", np.less_equal)):
                for t_x, mask_x in masks_x.items():
                    n_prem = mask_x.sum()

                    if n_prem == 0:
                        continue

                    for t_y in tj:
                        mask_y = op_y(f_j, t_y)
                        both = mask_x & mask_y
                        support = both.mean()

                        if support < min_support:
                            continue

                        confidence = both.sum() / n_prem
                        if confidence < min_confidence:
                            continue

                        score = (confidence, support)
                        rule = {'if': f"x {sym_x} {t_x:.4g}", 'then': f"y {sym_y} {t_y:.4g}",
                                'support': support, 'confidence': confidence, 'score': score}

                        if best is None or score > best['score']:
                            best = rule

        return best

    def find_association_rules(self, pareto_set: np.ndarray,
                               criteria: Tuple = None,
                               min_support: float = 0.1,
                               min_confidence: float = 0.8,
                               bidirectional: bool = True) -> Dict:
        """
        Induce association rules among feature pairs.
        TODO: This isn't really DRSA's responsibility, maybe move directly to IMO_DRSA?

        :param pareto_set: feature pairs
        :param criteria: criteria for association rules
        :param min_support: minimum support of the decision
        :param min_confidence: minimum confidence of the decision
        :param bidirectional: bidirectional association rules

        :return: the mapping (i,j) -> rule dict or None.
        """
        rules = {}

        for i, j in combinations(criteria, 2):
            r_ij = self.find_single_rule(pareto_set[:, i], pareto_set[:, j], min_support, min_confidence)

            if bidirectional:
                r_ji = self.find_single_rule(pareto_set[:, j], pareto_set[:, i], min_support, min_confidence)

                if r_ij and r_ji:
                    chosen = r_ij if r_ij['confidence'] >= r_ji['confidence'] else r_ji

                else:
                    chosen = r_ij or r_ji

                if chosen:
                    # ensure proper x,y labels
                    if chosen is r_ji:
                        chosen['if'] = chosen['if'].replace('x', f'f_{j}(x)')
                        chosen['then'] = chosen['then'].replace('y', f'f_{i}(x)')
                        rules[(j, i)] = chosen

                    else:
                        chosen['if'] = chosen['if'].replace('x', f'f_{i}(x)')
                        chosen['then'] = chosen['then'].replace('y', f'f_{j}(x)')
                        rules[(i, j)] = chosen

                else:
                    rules[(i, j)] = None

            else:
                rules[(i, j)] = r_ij

        return rules

    # ---------------------------------------------------------------------------------------------------------- #
    # Write the rules as strings
    # ---------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def explain_rules(rules: List, verbose: bool = True) -> List:
        """
        Convert decision or association rules to human-readable strings.

        :param rules: decision or association rules
        :param verbose: bool print the explanation if True, not if False

        :return: list of strings describing the rules
        """
        explanations = []

        for rule in rules:
            if len(rule) == 7:  # decision rule
                desc = rule[6]
                explanations.append(desc)

                if verbose:
                    print(desc)

            elif isinstance(rule, dict):  # association rule
                desc = f"if {rule['if']} then {rule['then']}  "
                desc += f"(support={rule['support']:.2f}, confidence={rule['confidence']:.2f})"
                explanations.append(desc)

                if verbose:
                    print(desc)

            else:
                raise ValueError(f"Unknown rule format: {rule!r}")

        return explanations
