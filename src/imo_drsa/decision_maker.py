import inspect
import random

import numpy as np
from contourpy.util.data import simple
from matplotlib.pyplot import inferno

from .drsa import DRSA


# ---------------------------------------------------------------------------------------------------------- #
# Base DM template
# ---------------------------------------------------------------------------------------------------------- #
class BaseDM:
    def __init__(self):
        self.decision_attribute = None

    def classify(self, T, association_rules):
        """

        :param T:
        :param association_rules:
        :return: classification of the values in T with either 2 (good) or 1 (other)
        """
        pass

    def select(self, rules):
        return rules

    def is_satisfied(self, X, T, rules) -> bool:
        return False

    def set_decision_attribute(self, decision_attribute):
        self.decision_attribute = decision_attribute

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
    def __init__(self):
        BaseDM.__init__(self)

    def select(self, rules):
        DRSA.explain_rules(rules, verbose=True)

        return rules

    def is_satisfied(self, X, T, rules) -> bool:
        answer = input("Are you satisfied by this Rule set? (y, n)")

        if answer == "y":
            print("Great! Ending now.")
            return True

        else:
            print("Sorry to hear that, let's try again.")
            return False


# ---------------------------------------------------------------------------------------------------------- #
# Automated DM
# ---------------------------------------------------------------------------------------------------------- #
class AutomatedDM(BaseDM):
    def __init__(self):
        BaseDM.__init__(self)


    def classify(self, T, association_rules):
        """

        :param T:
        :param association_rules:
        :return: classification of the values in T with either 2 (good) or 1 (other)
        """
        n = len(T)
        half = int((n*0.5))
        d1 = [1 for _ in range(0, half)]
        d2 = [2 for _ in range(half+1, n)]

        return np.concatenate([d1, d2])

    def select(self, rules):
        """
        For now, just only select the 'certain' rules.

        :param rules:
        :return:
        """
        chosen = []

        for rule in rules:
            if rule[4] == 'certain':
                chosen.append(rule)

        return chosen


# ---------------------------------------------------------------------------------------------------------- #
# Dummy DM (for unit tests)
# ---------------------------------------------------------------------------------------------------------- #
class DummyDM(BaseDM):
    def __init__(self):
        super().__init__()
        self.round = 0
        self.score = 'pareto'

    def classify(self, T, association_rules):
        """
        :param T: numpy array of shape (n_samples, n_objectives)
        :param association_rules: ignored in this simple strategy
        :return: numpy array of 1's (other) and 2's (good)
        """
        # Median-split on each objective, choose the split with the most balanced classes
        T = np.asarray(T)
        medians = np.median(T, axis=0)
        n, m = T.shape
        best_balance = n + 1
        best_labels = np.ones(n, dtype=int)

        for i in range(m):
            # smaller or equal values are "good" (2), others are "other" (1)
            labels = np.where(T[:, i] <= medians[i], 2, 1)
            balance = abs(np.sum(labels == 2) - np.sum(labels == 1))

            if balance < best_balance:
                best_balance = balance
                best_labels = labels

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
