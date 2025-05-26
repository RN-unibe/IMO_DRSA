import abc
import inspect
import random

import numpy as np
from contourpy.util.data import simple
from matplotlib.pyplot import inferno
from pymoo.indicators.hv import Hypervolume, HV

from .drsa import DRSA


# ---------------------------------------------------------------------------------------------------------- #
# Base DM template
# ---------------------------------------------------------------------------------------------------------- #
class BaseDM:


    @abc.abstractmethod
    def classify(self, T, association_rules):
        """

        :param T:
        :param association_rules:
        :return: classification of the values in T with either 2 (good) or 1 (other)
        """
        pass

    @abc.abstractmethod
    def select(self, rules):
        pass

    @abc.abstractmethod
    def is_satisfied(self, X, T, rules) -> bool:
        pass

    @abc.abstractmethod
    def set_decision_attribute(self, decision_attribute):
        pass

    def simple_score(self, rules, k=5, alpha=0.7):
        scored = [(alpha * r[3] + (1 - alpha) * r[2], r) for r in rules if r[4] == 'certain']
        scored.sort(reverse=True, key=lambda x: x[0])

        return [r for (_, r) in scored[:k]]

    def select_pareto(self, rules):
        certain = [r for r in rules if r[4] == 'certain']
        pareto = []

        for r in certain:
            s1, c1 = r[2], r[3]
            dominated = any((s2 >= s1 and c2 >= c1) and (s2 > s1 or c2 > c1)
                            for (_, _, s2, c2, _, _, _) in certain)

            if not dominated:
                pareto.append(r)

        return pareto


# ---------------------------------------------------------------------------------------------------------- #
# Interactive DM
# ---------------------------------------------------------------------------------------------------------- #
class InteractiveDM(BaseDM):

    def classify(self, T: np.ndarray, association_rules) -> np.ndarray:
        # Step 2: Show association rules as context
        print("\nAssociation Rules:")
        DRSA.explain_rules(association_rules, verbose=True)

        # Step 4: Present sample to DM
        print("\nCurrent Pareto Sample (objective values):")
        for idx, obj in enumerate(T):
            print(f"[{idx}] {obj}")

        # Ask which are “good”
        selection = input("\nSelect indices of 'good' solutions (comma-separated): ")
        good_idxs = {int(i) for i in selection.split(",") if i.strip().isdigit()}

        # Build labels: 2=good, 1=other
        labels = np.ones(len(T), dtype=int)
        for i in good_idxs:
            if 0 <= i < len(T):
                labels[i] = 2

        # Wire up decision attribute for DRSA
        self.set_decision_attribute(labels)
        return labels

    def select(self, rules):
        # Step 6: Present induced decision rules
        print("\nInduced Decision Rules:")
        descriptions = DRSA.explain_rules(rules, verbose=False)
        for idx, desc in enumerate(descriptions):
            print(f"[{idx}] {desc}")

        # Step 7: Ask DM to pick rules to enforce
        selection = input("\nSelect rule indices to enforce (comma-separated): ")
        chosen = []
        for token in selection.split(","):
            if token.strip().isdigit():
                i = int(token)
                if 0 <= i < len(rules):
                    chosen.append(rules[i])

        return chosen

    def is_satisfied(self, X, T: np.ndarray, rules) -> bool:
        # Step 3 & 9: After applying constraints and regenerating, ask if any solution is final
        print("\nNew Pareto Sample:")
        for idx, obj in enumerate(T):
            print(f"[{idx}] {obj}")

        selection = input("\nEnter index of a satisfying solution (or press Enter to continue): ")
        if selection.strip().isdigit():
            sol = int(selection)
            if 0 <= sol < len(T):
                print(f"\nSolution {sol} selected. Ending now.")
                return True

        print("\nContinuing to next iteration.")
        return False


# ---------------------------------------------------------------------------------------------------------- #
# Automated DM
# ---------------------------------------------------------------------------------------------------------- #
class AutomatedDM(BaseDM):
    """
    Automated Decision Maker for IMO-DRSA.
    Automatically classifies, selects rules, and decides stopping without human input.
    Primarily used for Test Problems, or if the optimal procedure is already known.
    """
    def __init__(self, max_rounds: int = 3, vol_eps: float = 1e-3, score: str = 'simple'):
        self.hv_indicator = None
        self.max_rounds = max_rounds
        self.vol_eps = vol_eps
        self.score = score  # 'pareto' or 'simple'
        self.prev_rules = None
        self.prev_hv = None
        self.round = 0

    def classify(self, T: np.ndarray, association_rules) -> np.ndarray:
        """
        Classify Pareto sample into 'good' (2) vs 'other' (1) via median-split.
        Chooses the objective whose split yields the most balanced classes.

        :param T: array of shape (n_samples, n_objectives)
        :param association_rules: not used in this strategy
        :return: array of labels (1 or 2)
        """
        medians = np.median(T, axis=0)
        n_points, n_objs = T.shape
        best_balance = n_points + 1
        best_labels = np.ones(n_points, dtype=int)

        for i in range(n_objs):
            labels = np.where(T[:, i] <= medians[i], 2, 1)
            balance = abs((labels == 2).sum() - (labels == 1).sum())
            if balance < best_balance:
                best_balance = balance
                best_labels = labels

        # compute and record current hypervolume (float)
        ref_point = np.max(T, axis=0) * (1 + 0.05) # margin = 0.05
        self.hv_indicator = HV(ref_point)

        hv_value = self.hv_indicator(T)
        self.prev_hv = hv_value

        return best_labels

    def select(self, rules):
        """
        Selects a subset of induced DRSA rules based on scoring method.

        :param rules: list of decision rules, each as a tuple
        :return: chosen subset of rules
        """
        if self.score == 'simple':
            return self.simple_score(rules)

        return self.select_pareto(rules)

    def is_satisfied(self, X, T: np.ndarray, rules) -> bool:
        """
        Determine whether the engine should stop.

        Stopping conditions:
        1) Only one or zero solutions remain.
        2) Hypervolume change < vol_eps since last iteration.
        3) Selected rule-set unchanged.
        4) Maximum rounds reached.

        :param X: current Pareto front (decision space), unused here
        :param T: current Pareto set (objective space)
        :param rules: selected DRSA rules
        :return: True to stop, False to continue
        """

        # 1) no or single solution
        if T.shape[0] <= 1:
            return True

        # 2) Hypervolume convergence
        hv = self.hv_indicator(T)
        if self.prev_hv is not None and abs(hv - self.prev_hv) < self.vol_eps:
            return True
        self.prev_hv = hv

        # 3) stable rule set
        if self.prev_rules is not None and rules == self.prev_rules:
            return True
        self.prev_rules = list(rules)

        # 4) fallback to max rounds
        self.round += 1
        return self.round >= self.max_rounds


# ---------------------------------------------------------------------------------------------------------- #
# Dummy DM (for unit tests)
# ---------------------------------------------------------------------------------------------------------- #
class DummyDM(BaseDM):
    def __init__(self):
        self.round = 0
        self.score = 'pareto'

    def classify(self, T, association_rules):
        """
        :param T: numpy array of shape (n_samples, n_objectives)
        :param association_rules: ignored in this simple strategy
        :return: numpy array of 1's (other) and 2's (good)
        """
        # Median-split on each objective, choose the split with the most balanced classes
        n, m = T.shape
        best_labels = np.ones(n, dtype=int)

        return best_labels

    def select(self, rules):
        if self.score == 'simple':
            return self.simple_score(rules)

        elif self.score == 'pareto':
            return self.select_pareto(rules)



    def is_satisfied(self, X, T, rules) -> bool:
        if self.round == 3:
            return True

        self.round += 1

        return False
