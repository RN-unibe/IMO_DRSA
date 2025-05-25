import numpy as np

from typing import Callable, List

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.plotting import plot



from src.imo_drsa.decision_maker import BaseDM
from src.imo_drsa.drsa import DRSA
from src.imo_drsa.problem_extender import ProblemExtender


class IMO_DRSAEngine():
    """
    Interactive Multi-objective Optimization using Dominance-based Rough Set Approach (IMO-DRSA).

    :param universe: Initial decision variable matrix of shape (n_solutions, n_variables).
    :param objectives: List of objective functions mapping a solution vector to a float.
    """

    def __init__(self, algorithm=NSGA2, **kwargs):
        """
        Initialise the DRSA and NSGA2 Classes.

        :param algorithm: Algorithm to use. Must be pymoo compatible
        :param kwargs: any and all algorithm parameters
        """
        self.drsa = DRSA()
        self.algorithm = algorithm(**kwargs)
        self.wrapper = ProblemExtender()


    def fit(self, problem:Problem, objectives:List[Callable]=None):
        """
        Fit the IMO-DRSA solver.

        :param problem: Problem to be optimised.
        :param universe: Initial population bounds.
        """
        self.problem = self.wrapper.enable_dynamic_constraints(problem=problem)
        self.objectives = objectives

        return self



    def solve(self, decision_maker: BaseDM, visualise: bool = False, max_iter: int = 5) -> bool:
        """
        Run the interactive optimization loop.

        :param decision_maker: Interface for classification and feedback.
        :param visualise: Whether to plot the Pareto front each iteration.
        :param max_iter: Maximum interactive iterations.
        :return: True if session finishes successfully; False otherwise.
        """


        P_idx = [i for i in range(0, len(self.objectives))]
        decision_attribute = None

        iteration: int = 0
        while iteration < max_iter:
            # Compute Pareto front under current constraints
            pareto_front, pareto_set = self.get_pareto_front()


            if pareto_front.size == 0:
                print("Infeasible constraints: please revise.")
                return False

            if visualise:
                self.visualise()

            self.drsa.fit(pareto_set=pareto_set, criteria=P_idx, decision_attribute=decision_attribute)

            # Induce association rules from current table
            association_rules = self.drsa.find_association_rules(pareto_set, criteria=P_idx)

            # Classify with DM feedback
            decisions = decision_maker.classify(pareto_set, association_rules)

            # Find a reduct and induce decision rules
            self.drsa.fit(pareto_set, P_idx, decisions)
            P_idx = self.drsa.find_reducts()[0]

            rules = self.drsa.induce_decision_rules(P_idx)

            # DM selects preferred rules
            selected = decision_maker.select(rules)

            # Generate new constraints from selected rules
            new_constraints = self.generate_constraints(selected)
            self.problem.add_constraints(new_constraints)

            if visualise:
                self.visualise()
            # Ask DM if current solutions are satisfactory
            if decision_maker.is_satisfied(pareto_front, pareto_set, rules):
                return True



            iteration += 1

        return False


    def visualise(self) -> None:
        plot(self.problem.pareto_front())



    def get_pareto_front(self, n_gen=200) -> (np.ndarray, np.ndarray):
        """
        Compute Pareto-optimal set using NSGA2 algorithm.

        :param universe: Bounds of initial population.
        :param objectives: Objective functions.
        :param constraints: Inequality constraints g(x) <= 0.
        :param pop_size: Population size for NSGA2.
        :param n_gen: Number of generations.

        :return: Tuple of decision variables (pareto_front) and objective values of Pareto front (pareto_set).
        """
        algorithm = NSGA2(pop_size=100)

        res = minimize(self.problem, algorithm, termination=('n_gen', n_gen), verbose=False)


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
                    constraints.append(lambda x, i=idx, th=threshold: th - self.objectives[i](x))

                else:
                    # f_i(x) <= threshold  ->  f_i(x) - threshold <= 0
                    constraints.append(lambda x, i=idx, th=threshold: self.objectives[i](x) - th)

        return constraints
