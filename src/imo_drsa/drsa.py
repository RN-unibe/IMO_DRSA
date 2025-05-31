import operator
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple
from itertools import combinations

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


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

    def __init__(self, pareto_front=None,
                 pareto_set: np.ndarray = None,
                 criteria: Tuple = None,
                 decision_attribute: np.ndarray = None,
                 direction="down"):
        """
        :param pareto_front: NumPy array with shape (N, n_var), each row is an object
        :param pareto_set: NumPy array with shape (N, n_var), each row is an object, columns are criteria evaluated on that object
        :param criteria: Tuple of column indices in pareto_set
        :param decision_attribute: NumPy array of length N, integer‐encoded decision classes (1, ..., m)
        :param direction: str direction of the union
        """
        self.fit(pareto_front=pareto_front,
                 pareto_set=pareto_set,
                 criteria=criteria,
                 decision_attribute=decision_attribute,
                 direction=direction)

    def fit(self, pareto_front=None,
            pareto_set: np.ndarray = None,
            criteria: Tuple = None,
            decision_attribute: np.ndarray = None,
            direction="down"):
        """
        :param pareto_front: NumPy array with shape (N, n_var), each row is an object
        :param pareto_set: NumPy array with shape (N, n_var), each row is an object, columns are criteria evaluated on that object
        :param criteria: Tuple of column indices in pareto_set
        :param decision_attribute: NumPy array of length N, integer‐encoded decision classes (1, ..., m)
        :param direction: str direction of the union
        :return self
        """

        self.pareto_front = pareto_front
        self.pareto_set = pareto_set
        self.decision_attribute = decision_attribute

        self.criteria_full = criteria

        self.N = 0 if pareto_set is None else pareto_set.shape[0]
        self.m = 0 if decision_attribute is None else (decision_attribute.max())

        self.direction = direction

        return self

    # ---------------------------------------------------------------------------------------------------------- #
    # Dominance‐cone computations
    # ---------------------------------------------------------------------------------------------------------- #

    def positive_cone(self, criteria: Tuple) -> np.ndarray:
        """
        Boolean mask [y, x] True if object y P-dominates x.

        :param criteria: Tuple of column indices in T to use as P subset of F = {f1,...,fn}
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

        :param criteria: Tuple of column indices in T to use as P subset of F = {f1,...,fn}
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

        :param criteria: Tuple of column indices in T to use as P subset of F = {f1,...,fn}
        :param threshold: int index of class
        :return: np.ndarray containing the lower approximation of upward union
        """
        cone = self.positive_cone(criteria)

        return np.all(~cone | (self.decision_attribute[:, None] >= threshold), axis=0)

    def upper_approx_up(self, criteria: Tuple, threshold: int) -> np.ndarray:
        """
        Upper approximation of upward union for decision >= threshold.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param threshold: int index of class
        :return: np.ndarray containing the upper approximation of upward union
         """
        cone = self.negative_cone(criteria)

        return np.any(cone & (self.decision_attribute[:, None] >= threshold), axis=0)

    def lower_approx_down(self, criteria: Tuple, threshold: int) -> np.ndarray:
        """
        Lower approximation of downward union for decision <= threshold.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param threshold: int index of class
        :return: np.ndarray containing the lower approximation of downward union
        """
        cone = self.negative_cone(criteria)

        return np.all(~cone | (self.decision_attribute[:, None] <= threshold), axis=0)

    def upper_approx_down(self, criteria: Tuple, threshold: int) -> np.ndarray:
        """
        Upper approximation of downward union for decision <= threshold.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param threshold: int index of class
        :return: np.ndarray containing the upper approximation of downward union
        """
        cone = self.positive_cone(criteria)

        return np.any(cone & (self.decision_attribute[:, None] <= threshold), axis=0)

    # ---------------------------------------------------------------------------------------------------------- #
    # Quality of approximation gamma_P(Cl)
    # ---------------------------------------------------------------------------------------------------------- #

    def quality(self, criteria: Tuple) -> float:
        """
        Compute the quality of approximation (gamma) for given criteria.

        :param criteria: Tuple of column indices in T to use as P subset of F = {f1,...,fn}
        """
        consistent_mask = np.ones(self.N, dtype=bool)

        for t in range(2, self.m + 1):
            lower = self.lower_approx_up(criteria, t)
            upper = self.upper_approx_up(criteria, t)

            boundary = upper & ~lower
            consistent_mask &= ~boundary

        return float(consistent_mask.sum()) / self.N

    # ---------------------------------------------------------------------------------------------------------- #
    # Finding reducts (brute‐force, not good for large n,)
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


    def core(self, reducts:List[Tuple]=None) -> Tuple:
        """
        Compute core criteria as intersection of all reducts.
        :param reducts: List of reducts as Tuples
        """
        reducts = reducts or self.find_reducts()

        if not reducts:
            return ()

        core_set = set(reducts[0])

        for red in reducts[1:]:
            core_set &= set(red)

        return tuple(sorted(core_set))

    # ---------------------------------------------------------------------------------------------------------- #
    # Decision-rule induction
    # ---------------------------------------------------------------------------------------------------------- #

    def make_rule_description(self, profile: Dict, conclusion: str, support: float, confidence: float, kind: str) -> str:
        """
        Build human-readable rule description.

        :param profile: dict with column indices of the compared objectives and variables
        :param conclusion: str conclusion of the decision
        :param support: float support of the decision
        :param confidence: float confidence of the decision
        :param kind: str type of rule
        :return: rule description
        """
        conds = []

        for idx, val in profile.items():
            op = ">=" if self.direction == 'up' else "<="
            conds.append(f"f_{idx + 1}(x) {op} {val}")

        premise = ' AND '.join(conds)

        return (f"[{kind.upper()}] IF {premise} THEN {conclusion} (support={support:.2f}, confidence={confidence:.2f})")

    def induce_decision_rules(self, criteria: Tuple = None,
                              threshold: int = 2,
                              minimal: bool = True) -> List[Tuple]:
        """
        Induce certain and possible decision rules for Cl>=threshold or Cl<=threshold.
        direction: 'up' or 'down'.

        :param criteria: list of column indices in T to use as P subset of F = {f1,...,fn}
        :param threshold: int index of class
        :param minimal: bool True if rules should be minimal
        :return: list of induced decision rules of form (profile, concl, support, confidence, kind, direction, desc)
        """
        criteria = criteria or self.criteria_full


        # Select appropriate approximations
        if self.direction == 'up':
            lower = self.lower_approx_up(criteria, threshold)
            upper = self.upper_approx_up(criteria, threshold)
            comp = operator.ge
            conf_fn = lambda mask: (self.decision_attribute[mask] >= threshold).mean()
        else:
            lower = self.lower_approx_down(criteria, threshold)
            upper = self.upper_approx_down(criteria, threshold)
            comp = operator.le
            conf_fn = lambda mask: (self.decision_attribute[mask] <= threshold).mean()

        concl = "x is 'good'"

        seen = set()
        rules = []
        for kind, indices in [('certain', np.where(lower)[0]), ('possible', np.where(upper & ~lower)[0])]:
            for idx in indices:
                profile = {i: self.pareto_set[idx, i] for i in criteria}
                mask = np.ones(self.N, dtype=bool)

                for i, val in profile.items():
                    mask &= comp(self.pareto_set[:, i], val)

                support = mask.mean()
                confidence = conf_fn(mask)

                desc = self.make_rule_description(profile, concl, support, confidence, kind)

                if desc not in seen:
                    seen.add(desc)
                    rule = (profile, concl, support, confidence, kind, self.direction, desc)
                    if self.is_robust(rule):
                        rules.append(rule)

        if minimal:
            minimal_rules = []

            for r in rules:
                if not any(self.subsumes(r2, r) for r2 in rules if r2 != r):
                    minimal_rules.append(r)

            rules = minimal_rules

        return rules

    def subsumes(self,
                 r1:Tuple,
                 r2:Tuple) -> bool:
        """
        Check if a rule is subsumed by another rule, i.e., if they have the same result,
        but one has a weaker premise.

        :param r1: Tuple rule to check
        :param r2: Tuple rule which migh subsume r1
        :return: True if premise of r1 is weaker than premise of r2, but has the same outcome, False otherwise.
        """
        p1 = r1[0]
        p2 = r2[0]
        # r1 subsumes r2 if p1 is weaker (i.e., thresholds for >= lower, <= higher)
        for i in p1:
            if i not in p2:
                return False
            if self.direction == 'down' and p1[i] < p2[i]:
                return False
            if self.direction == 'up' and p1[i] > p2[i]:
                return False

        return True

    def is_robust(self, rule):
        profile, _, _, _, kind, _, _ = rule
        mask = np.ones(self.N, dtype=bool)
        cmp_op = operator.ge if self.direction == 'up' else operator.le

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

        idx = 0
        for rule in rules:
            desc = rule[-1]
            explanations.append(f'[{idx}] {desc}')


            if verbose:
                print(f'[{idx}] {desc}')
            idx += 1

        return explanations

    # ---------------------------------------------------------------------------------------------------------- #
    # Association-rule mining
    # ---------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def find_association_rules(pareto_set:np.ndarray,
                               criteria:Tuple,
                               min_support: float = 0.1,
                               min_confidence: float = 0.8,
                               use_fp: bool = True) -> List[Tuple]:
        """
        Mine association rules among objectives (criteria) in the Pareto set.
        Only criterion bins are used—no decision classes involved.

        :param pareto_set: NumPy array with shape (N, n_var), each row is an object, columns are criteria evaluated on that object
        :param criteria: Tuple of column indices in pareto_set
        :param min_support: minimum support threshold
        :param min_confidence: minimum confidence threshold
        :param use_fp: if True, use fpgrowth; otherwise use apriori
        :return: list of (antecedents, consequents, support, confidence, description)
        """
        assert pareto_set is not None, "Pareto set is None"
        assert criteria is not None, "Criteria is None"

        bin_edges = {i: np.percentile(pareto_set[:, i], [25, 50, 75]) for i in criteria}

        transactions = []
        for x in pareto_set:
            items = []

            for i in criteria:
                val = x[i]
                bins = bin_edges[i]

                if val <= bins[0]:
                    label = f"f_{i + 1}<=Q1"
                elif val <= bins[1]:
                    label = f"f_{i + 1}<=Q2"
                elif val <= bins[2]:
                    label = f"f_{i + 1}<=Q3"
                else:
                    label = f"f_{i + 1}<=Q4"

                items.append(label)

            transactions.append(items)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        if use_fp:
            freq = fpgrowth(df, min_support=min_support, use_colnames=True)
        else:
            freq = apriori(df, min_support=min_support, use_colnames=True)

        assoc_rules = []
        if freq is not None and len(freq) > 0 :
            raw_rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)

            for _, row in raw_rules.iterrows():
                ant = row['antecedents']
                con = row['consequents']
                sup = row['support']
                conf = row['confidence']

                desc = DRSA.make_association_rule_description(ant, con, sup, conf)
                assoc_rules.append((ant, con, sup, conf, desc))

        return assoc_rules

    @staticmethod
    def make_association_rule_description(antecedents: frozenset,
                                            consequents: frozenset,
                                            support: float,
                                            confidence: float) -> str:
        """
        Human-readable DRSA-format description of an objective-only association rule.
        """
        conditions = [item.replace('<=Q', '<=').replace('>=Q', '>=') for item in sorted(antecedents)]
        conclusions = [item.replace('<=Q', '<=').replace('>=Q', '>=') for item in sorted(consequents)]
        premise = ' AND '.join(conditions)
        conclusions = ' AND '.join(conclusions)

        return f"[ASSOC] IF {premise} THEN {conclusions} (support={support:.2f}, confidence={confidence:.2f})"

    @staticmethod
    def summarize_association_rules(assoc_rules: List[Tuple], top_k: int = -1) -> Tuple:
        """
        Summarize only monotonic objective-objective rules in human-friendly terms.

        :param assoc_rules: List of association rules as Tuples
        :return: summaries (set of (text, total_sup, conf) tuples) and summaries_str
        """
        simple_rules = []
        for ant, con, sup, conf, _ in assoc_rules:
            if len(ant) == 1 and len(con) == 1:
                a, = ant
                c, = con
                pa = a.split("<=Q")
                pc = c.split("<=Q")
                if len(pa) != 2 or len(pc) != 2:
                    continue

                idx_a = int(pa[0].split("_")[1])
                q_a = int(pa[1])
                idx_c = int(pc[0].split("_")[1])
                q_c = int(pc[1])

                dir_a = "higher" if q_a >= 3 else "lower"
                dir_c = "higher" if q_c >= 3 else "lower"

                text = f"If objective {idx_a} is {dir_a}, objective {idx_c} tends to be {dir_c}"
                simple_rules.append((text, sup, conf))

        simple_rules.sort(key=lambda x: (-x[2], x[0]))

        top_rules = simple_rules if top_k < 0 else simple_rules[:top_k]

        merged: Dict[str, Tuple[float, float]] = {}

        for text, sup, conf in top_rules:
            if text in merged:
                total_sup, best_conf = merged[text]
                merged[text] = (total_sup + sup, max(best_conf, conf))
            else:
                merged[text] = (sup, conf)

        summaries = {(text, total_sup, conf) for text, (total_sup, conf) in merged.items()}

        lines = []
        for text, (total_sup, conf) in merged.items():
            lines.append(f"{text} (support={total_sup:.2f}, confidence={conf:.2f})")
        summaries_str = "\n".join(lines)

        return summaries, summaries_str

