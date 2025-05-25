import random

import numpy as np

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
        BaseDM.__init__(self)
        self.round = 0

    def classify(self, T, association_rules):
        """

        :param T:
        :param association_rules:
        :return: classification of the values in T with either 2 (good) or 1 (other)
        """

        random.seed(42)
        n = len(T)

        return np.array([random.choice([1, 2]) for _ in range(n)])


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


    def is_satisfied(self, X, T, rules) -> bool:
        if self.round == 1 :
            return True

        self.round = 1

        return False