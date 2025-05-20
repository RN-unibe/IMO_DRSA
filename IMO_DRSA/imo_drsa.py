import numpy as np
import pandas as pd
from matplotlib.pyplot import xcorr

from pymoo.algorithms.moo.nsga2 import NSGA2


from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from IMO_DRSA.problem import DRSAProblem
from IMO_DRSA.drsa import DRSA



class IMO_DRSA():

    def __init__(self,
                 U=None,
                 F=None):
        """
        Interactive Multi-objective Optimization using Dominance-based Rough Set Approach (IMO-DRSA).

        :param U: the universe of objects
        :param F: the set of all objective functions
        :param DM: the decision maker (DM)
        :param max_iter:
        """

        self.U = U
        self.F = F
        self.P = F
        self.I = [i for i in range(0, len(F)-1)]


    def solve(self, dm, visualise:bool=False, max_iter:int=5) -> bool:
        """

        :return: True, if the process finished successfully. False, otherwise.
        """
        drsa = DRSA()
        constraints = []
        X = self.U

        n_iter = 0
        while n_iter < max_iter:
            # Calc pareto 1
            # get association rules 2
            # Ask DM 3, 4
            # DRSA 5
            # Show DM 6, 7
            # make constraints 8

            X, T = self.pareto_front(X, constraints)


            association_rules = self.get_association_rules(T)

            d = dm.classify(T, association_rules) # the decision attribute

            drsa.fit(T, d, self.I)
            reduct = drsa.find_reducts()[0]

            self.set_P(reduct)

            rules = drsa.induce_rules(reduct, union_type='up', t=2)

            rules = dm.select(rules)

            new_constraints = self.generate_constraints(rules)

            constraints.extend(new_constraints)

            if dm.is_satisfied():
                return True

        return False



    def _visualise(self, pareto_front, pareto_set) -> None:
        pass

    def get_association_rules(self, T):
        pass

    def pareto_front(self, X, constraints=None, pop_size=100, n_gen=200):
        """
        Compute the Pareto front using NSGA2.

        :param X: the current universe of objects
        :param constraints: List of inequality constraint functions g_i(x) <= 0. Defaults to None.
        :type constraints: Optional[List[Callable[[np.ndarray], float]]]
        :param pop_size: Population size for NSGA2. Defaults to 100.
        :type pop_size: int
        :param n_gen: Number of generations to run. Defaults to 200.
        :type n_gen: int
        :return: Tuple containing:
                 - X (np.ndarray): Decision variable matrix, shape (n_solutions, n_variables).
                 - F (np.ndarray): Objective value matrix, shape (n_solutions, n_objectives).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if constraints is None:
            constraints = []

        xl = np.array([b[0] for b in X], dtype=float)
        xu = np.array([b[1] for b in X], dtype=float)

        problem = DRSAProblem(P=self.P, constr=constraints, n_var=len(self.P), n_obj=len(X), n_ieq_constr=len(constraints), xl=xl, xu=xu)

        algorithm = NSGA2(pop_size=pop_size)

        res = minimize(problem, algorithm, termination=('n_gen', n_gen), verbose=False)

        return res.X, res.F

    def generate_constraints(self, rules):
        pass

    def set_P(self, reduct):
        mask = np.isin(self.P, reduct)
        self.P = self.P[mask]

