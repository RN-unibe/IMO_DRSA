from typing import List, Callable

import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2


from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from IMO_DRSA.decision_maker import BaseDM
from IMO_DRSA.problem import DRSAProblem
from IMO_DRSA.drsa import DRSA



class IMO_DRSA():

    def __init__(self,
                 U:np.ndarray,
                 F:List[Callable[[np.ndarray], float]]):
        """
        Interactive Multi-objective Optimization using Dominance-based Rough Set Approach (IMO-DRSA).

        :param U: the universe of objects, numpy.ndarray of shape (N, n_var)
        :param F: the set of all objective functions, List of functions, i.e., callables
        :param DM: the decision maker (DM)
        :param max_iter:
        """
        
        assert isinstance(U, np.ndarray)
        assert U is not None

        
        assert isinstance(F, List)
        assert F is not None

        self.U = U
        self.F = F

        self.drsa = DRSA()


    def solve(self, dm:BaseDM, visualise:bool=False, max_iter:int=5) -> bool:
        """

        :return: True, if the process finished successfully. False, otherwise.
        """

        constraints = []
        rules = []
        P = self.F
        I = [i for i in range(0, len(P))]

        n_iter = 0
        while n_iter < max_iter:
            # Calc pareto 1
            # get association rules 2
            # Ask DM 3, 4
            # DRSA 5
            # Show DM 6, 7
            # make constraints 8

            X, new_T = self.pareto_front(self.U, P, constraints)



            if len(X) == 0:
                print("Infeasible constraint set. Asking DM to revise.")

                # ask dm to revise

                return False

            if dm.is_satisfied(X, new_T, rules): #Not sure if I need X here, but for now keep it
                return True

            T = new_T

            association_rules = self.get_association_rules(T)

            d = dm.classify(T, association_rules) # the decision attribute, must be either 1 (other) or 2 (good)

            self.drsa.fit(T, d, I)
            reduct = self.drsa.find_reducts()[0] #For now, just choose the first available reduct

            P = P[reduct]
            I = [i for i in range(0, len(P))]

            rules = self.drsa.induce_rules(reduct, union_type='up', t=2)

            rules = dm.select(rules)

            new_constraints = self.generate_constraints(rules)

            constraints.extend(new_constraints)



            n_iter += 1

        return False



    def _visualise(self, pareto_front, pareto_set) -> None:
        pass

    def get_association_rules(self, T):
        pass

    def pareto_front(self, U, P, constraints:List[Callable[[np.ndarray], float]]=None, pop_size=100, n_gen=200):
        """
        Compute the Pareto front using NSGA2.

        :param U: the current universe of objects
        :param P: the current set of criteria
        :param constraints: List of inequality constraint functions g_i(x) <= 0. Defaults to None.
        :param pop_size: Population size for NSGA2. Defaults to 100.
        :param n_gen: Number of generations to run. Defaults to 200.
        :return: Tuple containing:
                 - X (np.ndarray): Decision variable matrix, shape (n_solutions, n_variables).
                 - T (np.ndarray): Objective value matrix, shape (n_solutions, n_objectives).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """


        xl = np.min(U, axis=0)
        xu = np.max(U, axis=0)

        delta = xu - xl
        delta[delta == 0] = 1e-6  # prevent identical bounds
        margin = 0.05
        xl = xl - margin * delta
        xu = xu + margin * delta

        if not constraints:

            problem = DRSAProblem(P=P,
                                  n_var=U.shape[1],
                                  n_obj=len(P),
                                  xl=xl, xu=xu)

        else :
            problem = DRSAProblem(P=P,
                                  n_var=U.shape[1],
                                  n_obj=len(P),
                                  n_ieq_constr=len(constraints),
                                  xl=xl, xu=xu,
                                  constr=constraints)

        algorithm = NSGA2(pop_size=pop_size)

        res = minimize(problem, algorithm, termination=('n_gen', n_gen), verbose=True)

        X, T = res.X, res.F

        return X, T



    def generate_constraints(self, rules):
        
        return []



