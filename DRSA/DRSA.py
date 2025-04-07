import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem



class IMO_DRSA(BaseEstimator):
    """
    """

    def __init__(self, problem, algorithm):
        """
        :param problem: The problem defined in problem.py
        :param algorithm: for now, NSGA2
        """
        self.problem = problem
        self.algorithm = algorithm


    def fit(self, X, y=None):
        """
        X and y together = The Sorting Examples

        :param X: Condition attributes
        :param y: Decision attributes (bad, medium, good)
        :return:
        """
        self.is_fitted_ = True
        return self

    def predict(self, X):

        pareto_front, pareto_set = self._pareto()
        pass


    def calculation(self):
        pass

    def dialogue(self):
        pass

    def _pareto(self, n_gen:int=200) -> [np.ndarray, np.ndarray]:
        #problem = None

        #algorithm = NSGA2(ref_points=ref_points, pop_size=pop_size)

        termination = get_termination("n_gen", n_gen)

        results = minimize(self.problem, self.algorithm, termination, seed=1, verbose=True)

        pareto_front = results.F  # Objective values
        pareto_set = results.X  # Corresponding decision variables

        return pareto_front, pareto_set

    