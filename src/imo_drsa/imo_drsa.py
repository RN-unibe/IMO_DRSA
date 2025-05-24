import numpy as np

from typing import Callable, List

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.optimize import minimize


from src.imo_drsa.decision_maker import BaseDM
from src.imo_drsa.drsa import DRSA
from src.imo_drsa.problem_wrapper import ElementwiseProblemWrapper, ProblemWrapper


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


    def fit(self, problem:Problem,
            universe: np.ndarray,
            objectives: List[Callable]) -> None:
        """
        Fit the IMO-DRSA solver.

        :param problem: Problem to be optimised.
        :param universe: Initial population bounds.
        :param objectives: Objective functions.
        """
        self.problem = ProblemWrapper(base_problem=problem)
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
        assert self.universe is not None, "Universe must not be empty."
        assert self.objectives is not None, "Objectives must be provided."

        rules = []
        P_idx = tuple(range(len(self.objectives)))

        iteration: int = 0
        while iteration < max_iter:
            # Compute Pareto front under current constraints
            pareto_front, pareto_set = self.pareto_front()

            if pareto_front.size == 0:
                print("Infeasible constraints: please revise.")
                return False

            if visualise:
                self._visualise(pareto_front, pareto_set)

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
            self.problem.extend_constraints(new_constraints)

            iteration += 1

        return False


    def _visualise(self, pareto_front, pareto_set) -> None:
        pass


    def pareto_front(self, n_gen=200) -> (np.ndarray, np.ndarray):
        """
        Compute Pareto-optimal set using NSGA2 algorithm.

        :param universe: Bounds of initial population.
        :param objectives: Objective functions.
        :param constraints: Inequality constraints g(x) <= 0.
        :param pop_size: Population size for NSGA2.
        :param n_gen: Number of generations.

        :return: Tuple of decision variables (pareto_front) and objective values of Pareto front (pareto_set).
        """

        res = minimize(self.problem, self.algorithm, termination=('n_gen', n_gen), verbose=True)

        pareto_front, pareto_set = res.X, res.F

        return pareto_front, pareto_set


    def generate_constraints(self, selected_rules) -> List[Callable]:
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
