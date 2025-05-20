import numpy as np
import pandas as pd
from matplotlib.pyplot import xcorr
from numpy.f2py.auxfuncs import throw_error
from pymoo.core.algorithm import Algorithm

from sklearn.base import BaseEstimator

from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from itertools import chain, combinations


import types
import json
from datetime import datetime
from collections import defaultdict

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from IMO_DRSA.problem import DRSAProblem



class IMO_DRSA_Controller():

    def __init__(self,
                 U=None,
                 F=None,
                 d=None,
                 DM=None,
                 max_iter: int = 200):
        """
        Interactive Multi-objective Optimization using Dominance-based Rough Set Approach (IMO-DRSA).

        :param U: the universe of objects
        :param F: the set of all objective functions
        :param d: the decision attribute
        :param DM: the decision maker (DM)
        :param max_iter:
        """

        self.DM_ = DM
        self.U_ = U
        self.F_ = F
        self.d_ = d
        self.max_iter_ = max_iter

        self.algorithm_ = NSGA2(pop_size=100) #TODO: params
        self.drsa_ = DRSA(U, F, d, DM, pareto_front=None, pareto_set=None)





    def solve(self, n_gen:int=200, visualise:bool=False) -> bool:
        """
        :param n_gen:

        :return: True, if the process finished successfully. False, otherwise.
        """

        if not self.F_:
            raise Exception("X is undefined")

        if not self.d_:
            raise Exception("y is undefined")

        pareto_front, pareto_set = None, None

        n_iter:int = 0
        while n_iter < self.max_iter_:
            pareto_front, pareto_set

            sorting = self.DM_.check_solutions(pareto_front, pareto_set)



        return True






    def _visualise(self, pareto_front, pareto_set) -> None:
        pass

    def incorporate_rules(self, accepted_rules):
        self.rules_ = accepted_rules # Is this enough?







if __name__ == "__main__":

    # TODO
    drsa_rob = DRSAProblem(n_var=2, n_obj=2, n_ieq_constr=2, xl=np.array([-2, -2]), xu=np.array([2, 2]))
    ref_points = np.array([[0.5, 0.2], [0.1, 0.6]])
    imo_drsa = IMO_DRSA_Controller(ref_points=ref_points, pop_size=100)

    res = minimize(drsa_rob,
                   imo_drsa,
                   ("n_gen", 100),
                   verbose=False,
                   seed=1)

    plot = Scatter()
    plot.add(res.F, edgecolor="red", facecolor="none")
    plot.show()