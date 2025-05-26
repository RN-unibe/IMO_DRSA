from itertools import combinations, product
import operator
from typing import Dict, List, Tuple

import numpy as np



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

    def __init__(self, pareto_set: np.ndarray = None, criteria: Tuple = None, decision_attribute: np.ndarray = None):
        """
        :param pareto_set: NumPy array with shape (N, n_var), each row is an object, columns are criteria evaluated on that object
        :param decision_attribute: NumPy array of length N, integer‐encoded decision classes (1, ..., m)
        :param criteria: list of column indices in pareto_set
        """
        if pareto_set is not None and decision_attribute is not None and criteria is not None:
            self.fit(pareto_set, criteria, decision_attribute)

    def fit(self, pareto_set: np.ndarray, criteria: Tuple, decision_attribute: np.ndarray = None):
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

        self.N = pareto_set.shape[0]
        self.m = 0 if decision_attribute is None else (decision_attribute.max())

        return self

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

    def core(self) -> Tuple:
        """Compute core criteria as intersecti
        on of all reducts."""
        reducts = self.find_reducts()
        if not reducts:
            return ()

        core_set = set(reducts[0])

        for red in reducts[1:]:
            core_set &= set(red)

        return tuple(sorted(core_set))

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
                              threshold: int = 2,
                              minimal: bool = True,
                              robust: bool = True) -> List[Tuple]:
        """
        Induce certain and possible decision rules for Cl>=threshold or Cl<=threshold.
        direction: 'up' or 'down'.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param direction: str direction of union, either 'up' or 'down'.
        :param threshold:int of class

        :return: list of induced decision rules of form (profile, concl, support, confidence, kind, direction, desc)
        """
        crit = criteria or self.criteria_full
        if direction not in ('up', 'down'):
            raise ValueError("direction must be 'up' or 'down'")

        # Select appropriate approximations
        if direction == 'up':
            lower = self.lower_approx_up(crit, threshold)
            upper = self.upper_approx_up(crit, threshold)
            comp = operator.ge
            conf_fn = lambda mask: (self.decision_attribute[mask] >= threshold).mean()
            concl = f"d >= {threshold}"
        else:
            lower = self.lower_approx_down(crit, threshold)
            upper = self.upper_approx_down(crit, threshold)
            comp = operator.le
            conf_fn = lambda mask: (self.decision_attribute[mask] <= threshold).mean()
            concl = f"d <= {threshold}"

        seen = set()
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

                if desc not in seen:
                    seen.add(desc)
                    rules.append((profile, concl, support, confidence, kind, direction, desc))

        # Filter robust: at least one base
        if robust:
            rules = [r for r in rules if self.is_robust(r, direction)]

        # Filter minimal: no weaker rule subsumes it
        if minimal:
            rules = rules.copy()
            minimal_rules = []

            for r in rules:
                if not any(self.subsumes(r2, r, direction) for r2 in rules if r2 != r):
                    minimal_rules.append(r)

            rules = minimal_rules

        return rules

    @staticmethod
    def subsumes(r1, r2, direction='up'):
        p1, _, _, _, _, _, _ = r1
        p2, _, _, _, _, _, _ = r2
        # r1 subsumes r2 if p1 is weaker (i.e., thresholds for >= lower, <= higher)
        for i in p1:
            if i not in p2:
                return False
            if direction == 'up' and p1[i] < p2[i]:
                return False
            if direction == 'down' and p1[i] > p2[i]:
                return False

        return True

    def is_robust(self, rule, direction='up'):
        profile, _, _, _, kind, _, _ = rule
        mask = np.ones(self.N, dtype=bool)
        cmp_op = operator.ge if direction == 'up' else operator.le

        for i, val in profile.items():
            mask &= cmp_op(self.pareto_set[:, i], val)

        # Base: exact match
        base_mask = np.ones(self.N, dtype=bool)
        for i, val in profile.items():
            base_mask &= self.pareto_set[:, i] == val

        return bool(np.any(base_mask & mask))

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
            desc = rule[-1]
            explanations.append(desc)

            if verbose:
                print(desc)

        return explanations

    # ---------------------------------------------------------------------------------------------------------- #
    # Association-rule mining
    # ---------------------------------------------------------------------------------------------------------- #

    def find_association_rules(self,
                               criteria: Tuple[int, ...] = None,
                               min_support: float = 0.1,
                               min_confidence: float = 0.8,
                               max_antecedent: int = 2,
                               max_consequent: int = 2) -> List[Tuple]:
        """
        Induce association rules with multi-feature antecedents and consequents.
        WARNING: This is horrible for many criteria!

        :param criteria: features to consider
        :param min_support: minimum support threshold
        :param min_confidence: minimum confidence threshold
        :param max_antecedent: max number of conditions in the IF part
        :param max_consequent: max number of parts in the THEN part
        :return: list of tuples
                 (antecedent, consequent, support, confidence, kind, relation, description)
        """
        crit = criteria or self.criteria_full
        rules = []
        seen = set()
        X = self.pareto_set  # shape (n_samples, n_features)

        # Precompute unique thresholds per feature
        thresholds = {f: np.unique(X[:, f]) for f in crit}

        # Loop over all possible antecedent feature sets
        for lhs_size in range(1, max_antecedent + 1):
            for lhs_feats in combinations(crit, lhs_size):
                # For each possible assignment of (feature, op, threshold) on the left hand side (LHS)
                lhs_conditions = []

                for f in lhs_feats:
                    lhs_conditions.append([
                        (f, t, op_sym, op_fn)
                        for t in thresholds[f]
                        for op_sym, op_fn in ((">=", np.greater_equal), ("<=", np.less_equal))
                    ])
                # Cartesian product of each feature’s possible conds
                for lhs_assignment in product(*lhs_conditions):
                    # build LHS mask
                    mask_lhs = np.ones(X.shape[0], dtype=bool)
                    for f, t, _, fn in lhs_assignment:
                        mask_lhs &= fn(X[:, f], t)

                    support_lhs = mask_lhs.mean()
                    if support_lhs < min_support:
                        continue

                    # Now right hand side (RHS): pick disjoint feature sets
                    remaining = set(crit) - set(lhs_feats)
                    for rhs_size in range(1, max_consequent + 1):
                        for rhs_feats in combinations(remaining, rhs_size):
                            rhs_conditions = []

                            for f in rhs_feats:
                                rhs_conditions.append([
                                    (f, t, op_sym, op_fn)
                                    for t in thresholds[f]
                                    for op_sym, op_fn in ((">=", np.greater_equal), ("<=", np.less_equal))
                                ])

                            for rhs_assignment in product(*rhs_conditions):
                                # build RHS mask
                                mask_rhs = np.ones(X.shape[0], dtype=bool)
                                for f, t, _, fn in rhs_assignment:
                                    mask_rhs &= fn(X[:, f], t)

                                support_both = np.mean(mask_lhs & mask_rhs)
                                if support_both < min_support:
                                    continue

                                confidence = support_both / support_lhs
                                if confidence < min_confidence:
                                    continue

                                # format rule
                                antecedent = " AND ".join(f"f_{f + 1} {sym} {t}"
                                                          for f, t, sym, _ in lhs_assignment)
                                consequent = " AND ".join(f"f_{f + 1} {sym} {t}"
                                                          for f, t, sym, _ in rhs_assignment)
                                desc = (f"[ASSOC] IF {antecedent} "
                                        f"THEN {consequent} "
                                        f"(support={support_both:.2f}, confidence={confidence:.2f})")
                                if desc in seen:
                                    continue
                                seen.add(desc)
                                relation = f"{','.join(str(f) for f in lhs_feats)}->" \
                                           f"{','.join(str(f) for f in rhs_feats)}"
                                rules.append((lhs_assignment,
                                              rhs_assignment,
                                              support_both,
                                              confidence,
                                              'assoc',
                                              relation,
                                              desc))
        return rules
