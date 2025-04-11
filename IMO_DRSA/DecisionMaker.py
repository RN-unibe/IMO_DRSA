


class BaseDM:
    def __init__(self):
        pass


    def check_solutions(self, rules):
        pass

    def select_rules(self, pareto_front, pareto_set):
        return None


class StandardDM(BaseDM):
    def __init__(self):
        super().__init__()


class DummyDM(BaseDM):
    def __init__(self):
        super().__init__()