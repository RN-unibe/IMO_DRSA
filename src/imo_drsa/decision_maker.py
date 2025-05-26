import abc
import numpy as np

from pymoo.indicators.hv import HV

from .drsa import DRSA


# ---------------------------------------------------------------------------------------------------------- #
# Base DM template
# ---------------------------------------------------------------------------------------------------------- #
class BaseDM:
    """
    BaseDM provides the abstract interface for a decision maker (DM) used in an
    iterative preference elicitation loop. Subclasses must implement classify,
    select, and is_satisfied methods to drive the loop.
    """

    @abc.abstractmethod
    def classify(self, T, assoc_rules_summary):
        """
        Classify Pareto sample into 'good' (2) vs 'other' (1) via median-split.
        Chooses the objective whose split yields the most balanced classes.

        :param T: array of shape (n_samples, n_objectives)
        :param assoc_rules_summary: str the summary of the association rules
        :return: array of labels (1 or 2) for each sample in T
        """
        pass

    @abc.abstractmethod
    def select(self, rules):
        """
        Selects a subset of induced DRSA rules based on scoring method.

        :param rules: list of decision rules, each as a tuple
        :return: chosen subset of rules
        """
        pass

    @abc.abstractmethod
    def select_reduct(self, reducts, core):
        pass

    @abc.abstractmethod
    def is_satisfied(self, X, T, rules) -> bool:
        """
        Determine whether the engine should stop the preference elicitation loop.

        :param X: current Pareto front in decision space (unused here)
        :param T: current Pareto set in objective space
        :param rules: selected DRSA rules
        :return: True to stop, False to continue
        """
        pass

    def simple_score(self, rules, k=5, alpha=0.7):
        """
        Score and select top-k rules based on a weighted combination of rule support
        and confidence for 'certain' rules.

        :param rules: list of DRSA rules, each tuple contains (conditions, conclusions, support, confidence, type, ...)
        :param k: number of rules to select
        :param alpha: weight for confidence vs support (0 <= alpha <= 1)
        :return: top-k scored rules
        """
        scored = [(alpha * r[3] + (1 - alpha) * r[2], r) for r in rules if r[4] == 'certain']
        scored.sort(reverse=True, key=lambda x: x[0])

        return [r for (_, r) in scored[:k]]

    def select_pareto(self, rules):
        """
        Select Pareto-optimal rules based on support and confidence for 'certain' rules.

        :param rules: list of DRSA rules
        :return: Pareto-optimal subset of rules
        """
        certain = [r for r in rules if r[4] == 'certain']
        pareto = []
        for r in certain:
            s1, c1 = r[2], r[3]
            dominated = any((s2 >= s1 and c2 >= c1) and (s2 > s1 or c2 > c1) for (_, _, s2, c2, _, _, _) in certain)

            if not dominated:
                pareto.append(r)

        return pareto

    def is_interactive(self):
        return False

# ---------------------------------------------------------------------------------------------------------- #
# Interactive DM
# ---------------------------------------------------------------------------------------------------------- #
class InteractiveDM(BaseDM):
    """
    An interactive Decision Maker that guides a human-in-the-loop
    preference elicitation process over Pareto-optimal samples using association
    and decision rules.
    """

    def classify(self, T: np.ndarray, assoc_rules_summary:str) -> np.ndarray:
        """
        Prompt the user to classify Pareto-optimal samples into 'good' or 'other'.
        Displays association rules and current Pareto sample for context.

        :param T: objective values of Pareto-optimal samples (n_samples, n_objectives)
        :param assoc_rules_summary: association rules for context
        :return: array of labels (2 for 'good', 1 for 'other')
        """
        print("\nAssociation Rules:")
        print(assoc_rules_summary)

        print("\nCurrent Pareto Sample (objective values):")
        for idx, obj in enumerate(T):
            print(f"[{idx}] {obj}")

        selection = input("\nSelect indices of 'good' solutions (comma-separated): ")
        good_idxs = {int(i) for i in selection.split(",") if i.strip().isdigit()}

        labels = np.ones(len(T), dtype=int)
        for i in good_idxs:
            if 0 <= i < len(T):
                labels[i] = 2
        return labels

    def select(self, rules):
        """
        Prompt the user to select which induced DRSA rules to enforce next iteration.

        :param rules: list of induced decision rules
        :return: subset of rules selected by the user
        """
        print("\nInduced Decision Rules:")
        descriptions = DRSA.explain_rules(rules, verbose=False)
        for idx, desc in enumerate(descriptions):
            print(f"[{idx}] {desc}")

        selection = input("\nSelect rule indices to enforce (comma-separated): ")
        chosen = []
        for token in selection.split(","):
            if token.strip().isdigit():
                i = int(token)
                if 0 <= i < len(rules):
                    chosen.append(rules[i])
        return chosen

    def select_reduct(self, reducts, core):
        if len(reducts) > 1:
            print("Available Reducts:")
            for idx, red in enumerate(reducts):
                print(f"[{idx}] {red}")

            print(f"Core criteria (must keep): {core}")

            selected_idx = input("Select reduct by index (default 0): ").strip()
            selected_idx = int(selected_idx) if selected_idx.isdigit() else 0
            selected_reduct = reducts[selected_idx]
        else:
            selected_reduct = reducts[0]

        return selected_reduct

    def is_satisfied(self, X, T: np.ndarray, rules) -> bool:
        """
        Prompt the user to indicate if any solution from the new Pareto sample is satisfactory.

        :param X: decision space points (unused)
        :param T: objective values of new Pareto sample
        :param rules: enforced DRSA rules (unused)
        :return: True if the user selects a solution to end, False otherwise
        """
        print("\nNew Pareto Sample:")
        for idx, obj in enumerate(T):
            print(f"[{idx}] {obj}")

        selection = input("\nAre you satisfied with this selection? (y, n): ")
        if selection.strip().lower() == 'y':
            return True

        print("\nContinuing to next iteration.")
        return False

    def is_interactive(self):
        return True

# ---------------------------------------------------------------------------------------------------------- #
# Automated DM
# ---------------------------------------------------------------------------------------------------------- #
class AutomatedDM(BaseDM):
    """
    Automated Decision Maker for IMO-DRSA.
    Automatically classifies, selects rules, and decides stopping without human input.
    Primarily used for test problems or known optimal procedures.
    """

    def __init__(self, max_rounds: int = 3, vol_eps: float = 1e-3, score: str = 'simple'):
        """
        Initialize the automated decision maker.

        :param max_rounds: maximum number of iterations before stopping
        :param vol_eps: hypervolume convergence threshold
        :param score: scoring method ('simple' or 'pareto')
        """
        self.hv_indicator = None
        self.max_rounds = max_rounds
        self.vol_eps = vol_eps
        self.score = score  # 'pareto' or 'simple'
        self.prev_rules = None
        self.prev_hv = None
        self.round = 0

    def classify(self, T: np.ndarray, assoc_rules_summary) -> np.ndarray:
        """
        Automatically classify samples via median-split on objectives for balanced labels,
        compute initial hypervolume indicator.

        :param T: objective values of Pareto-optimal samples
        :param assoc_rules_summary: association rules (unused)
        :return: labels array where 2 indicates better than median
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

        ref_point = np.max(T, axis=0) * (1 + 0.05)  # margin = 0.05
        self.hv_indicator = HV(ref_point)

        hv_value = self.hv_indicator(T)
        self.prev_hv = hv_value

        return best_labels

    def select(self, rules):
        """
        Automatically select rules based on configured scoring strategy.

        :param rules: list of induced decision rules
        :return: selected subset of rules
        """
        if self.score == 'simple':
            return self.simple_score(rules)
        return self.select_pareto(rules)

    def select_reduct(self, reducts, core):
        return min(reducts, key=len)

    def is_satisfied(self, X, T: np.ndarray, rules) -> bool:
        """
        Determine stopping condition based on:
        1) number of solutions <= 1
        2) hypervolume change < vol_eps
        3) unchanged rule set
        4) reaching max_rounds

        :param X: decision space points (unused)
        :param T: objective values of Pareto-optimal samples
        :param rules: selected DRSA rules
        :return: True if stopping condition met, False otherwise
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
    """
    Dummy Decision Maker for unit tests.
    Provides trivial classification and selection logic without interaction.
    """

    def __init__(self):
        """
        Initialize the dummy decision maker.
        """
        self.round = 0
        self.score = 'pareto'

    def classify(self, T, assoc_rules_summary):
        """
        Dummy classification: always returns label 1 for all samples.

        :param T: objective values (unused)
        :param assoc_rules_summary: (unused)
        :return: label array of ones
        """
        n, m = T.shape
        return np.ones(n, dtype=int)

    def select(self, rules):
        """
        Dummy selection: choose rules based on configured scoring strategy.

        :param rules: list of DRSA rules
        :return: selected rules
        """
        if self.score == 'simple':
            return self.simple_score(rules)
        elif self.score == 'pareto':
            return self.select_pareto(rules)

    def select_reduct(self, reducts, core):
        return reducts[0]

    def is_satisfied(self, X, T, rules) -> bool:
        """
        Dummy stopping: stops after 3 rounds.

        :param X: (unused)
        :param T: (unused)
        :param rules: (unused)
        :return: True if 3 rounds completed, False otherwise
        """
        if self.round == 3:
            return True
        self.round += 1
        return False
