
class BaseDM:
    def __init__(self, d):
        self.d = d

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




class StandardDM(BaseDM):
    def __init__(self, d):
        BaseDM.__init__(self, d)

    def choose_decision_rules(self, rules):
        chosen = []

        for rule in rules:
            if rule[3] == 'certain':
                chosen.append(rule)

        return chosen


class DummyDM(BaseDM):
    def __init__(self, d):
        BaseDM.__init__(self, d)



