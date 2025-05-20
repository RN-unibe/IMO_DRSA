
class BaseDM:
    def __init__(self, d):
        self.d = d


    def choose_decision_rules(self, rules):
        pass

    def get_decision_atribute(self):
        return self.d


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



