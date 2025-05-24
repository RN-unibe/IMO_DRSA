import numpy as np

from typing import Callable, List

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from src.imo_drsa.decision_maker import BaseDM
from src.imo_drsa.drsa import DRSA
from src.imo_drsa.problem import DRSABaseProblem


class IMO_DRSA():
    """
    Interactive Multi-objective Optimization using Dominance-based Rough Set Approach (IMO-DRSA).

    :param universe: Initial decision variable matrix of shape (n_solutions, n_variables).
    :param objectives: List of objective functions mapping a solution vector to a float.
    """

    def __init__(self, **kwargs):
        """
        Initialise the DRSA and NSGA2 Classes.

        :param kwargs: any and all NSGA2 parameters
        """
        self.drsa = DRSA()
        self.algorithm = NSGA2(**kwargs)

    def fit(self, universe: np.ndarray = None, objectives: List[Callable] = None):
        """
        Initialize the IMO-DRSA solver.

        :param universe: Initial population bounds.
        :param objectives: Objective functions.
        """
        self.universe = universe
        self.objectives = objectives

    def solve(self, decision_maker: BaseDM, visualise: bool = False, max_iter: int = 5) -> bool:
        """
        Run the interactive optimization loop.

        :param decision_maker: Interface for classification and feedback.
        :param visualise: Whether to plot the Pareto front each iteration.
        :param max_iter: Maximum interactive iterations.
        :return: True if session finishes successfully; False otherwise.
        """
        assert isinstance(self.universe, np.ndarray), "Universe must be a numpy array."
        assert self.objectives is not None, "Objectives must be provided."

        constraints = []
        rules = []
        P_idx = tuple(range(len(self.objectives)))

        iteration: int = 0
        while iteration < max_iter:
            # Compute Pareto front under current constraints
            pareto_front, pareto_set = self.pareto_front(self.universe, self.objectives, constraints)

            if visualise:
                self._visualise(pareto_front, pareto_set)

            if pareto_front.size == 0:
                print("Infeasible constraints: please revise.")
                return False

            # Ask DM if current solutions are satisfactory
            if decision_maker.is_satisfied(pareto_front, pareto_set, rules):
                return True

            # Induce association rules from current table
            association_rules = self.drsa.find_association_rules(pareto_set)

            # Classify with DM feedback
            decisions = decision_maker.classify(pareto_set, association_rules)

            # Find a reduct and induce decision rules
            self.drsa.fit(pareto_set, decisions, P_idx)
            reduct = self.drsa.find_reducts()[0]

            self.objectives = [self.objectives[i] for i in reduct]

            rules = self.drsa.induce_decision_rules(reduct)

            # DM selects preferred rules
            selected = decision_maker.select(rules)

            # Generate new constraints from selected rules
            new_constraints = self.generate_constraints(selected)
            constraints.extend(new_constraints)

            iteration += 1

        return False

    def _visualise(self, pareto_front, pareto_set) -> None:
        pass

    def pareto_front(self, U, P, constraints: List = None, pop_size=100, n_gen=200):
        """
        Compute Pareto-optimal set using NSGA2 algorithm.

        :param universe: Bounds of initial population.
        :param objectives: Objective functions.
        :param constraints: Inequality constraints g(x) <= 0.
        :param pop_size: Population size for NSGA2.
        :param n_gen: Number of generations.

        :return: Tuple of decision variables (pareto_front) and objective values of Pareto front (pareto_set).
        """

        xl = np.min(U, axis=0)
        xu = np.max(U, axis=0)

        delta = xu - xl
        delta[delta == 0] = 1e-6  # prevent identical bounds
        margin = 0.05
        xl = xl - margin * delta
        xu = xu + margin * delta

        problem = DRSABaseProblem(P=P,
                                  n_var=U.shape[1],
                                  n_obj=len(P),
                                  xl=xl, xu=xu,
                                  constr=constraints)

        res = minimize(problem, self.algorithm, termination=('n_gen', n_gen), verbose=True)

        pareto_front, pareto_set = res.X, res.F

        return pareto_front, pareto_set

    def generate_constraints(self, selected_rules) -> List:
        """
        Translate selected decision rules into inequality constraints g(x) <= 0.

        :param selected_rules: List of decision rules (profile, conclusion, support, confidence, kind, direction, description).
        :return: Constraints as functions mapping x to a float (<=0).
        """
        constraints = []

        for profile, _, _, _, _, direction, _ in selected_rules:
            for idx, threshold in profile.items():
                if direction == 'up':
                    # f_i(x) >= threshold  ->  threshold - f_i(x) <= 0
                    constraints.append(lambda x, fi=idx, th=threshold: th - self.objectives[fi](x))

                else:
                    # f_i(x) <= threshold  ->  f_i(x) - threshold <= 0
                    constraints.append(lambda x, fi=idx, th=threshold: self.objectives[fi](x) - th)

        return constraints
