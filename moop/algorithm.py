import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem

class IMO_DRSA(BaseEstimator):
    def __init__(self) :
        self.drsa_ = None # DRSA(...)
        pass



    def predict(self, X):
        self.calculation()
        self.dialogue()

        pass


    def calculation(self):
        pass

    def dialogue(self):
        """
        In the dialogue stage, it requires the DM to sort a sample of solutions into two classes: “good” and “others”.
        Finally, it gives a recommendation for the choice.

        :return:
        """
        pass



class DRSA(BaseEstimator):
    """
    Dominance-based Rough Set Approach (moop) "is a methodology of multiple criteria decision analysis aiming at obtaining
    a representation of the DM’s preferences in terms of easily understandable “if ..., then ...” decision rules,
    on the basis of some exemplary decisions (past decisions or simulated decisions) given by the DM." (Greco et al., 2008)
    """

    def __init__(self, problem, algorithm, ref_point_):
        """
        :param problem: The problem defined in problem.py
        :param algorithm: for now, NSGA2
        :param ref_point_: the reference point given by the DM for the dialogue stage
        """
        self.problem_ = problem
        self.algorithm_ = algorithm
        self.ref_point_ = ref_point_


    def fit(self, X, y=None):
        """
        X and y together = The Sorting Examples

        "The criteria and the class assignment considered within moop
        correspond to the condition attributes and the decision attribute, respectively,
        in the classical Rough Set Approach (Pawlak, 1991). For example, in multiple
        criteria sorting of cars, an example of decision is an assignment of a particular
        car evaluated on such criteria as maximum speed, acceleration, price and
        fuel consumption to one of three classes of overall quality: “bad”, “medium”,
        “good”." (Greco et al., 2008)

        :param X: Condition attributes (Matrix, where X_ij = f_i(x_j))
        :param y: Decision attributes d : U -> {1, ...,m}
        :return:
        """
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X):

        pareto_front, pareto_set = self._pareto()
        pass




    def certainty(self, X=None) -> float :
        """
        "The difference between the upper and lower approximation constitutes
        the boundary region of the rough set, whose elements cannot be characterized
        with certainty as belonging or not to X (by using the available information).
        The information about objects from the boundary region is, therefore, inconsistent.
        The cardinality of the boundary region states, moreover, the extent
        to which it is possible to express X in terms of certainty, on the basis of the
        available information. In fact, these objects have the same description, but are
        assigned to different classes, such as patients having the same symptoms (the
        same description), but different pathologies (different classes). For this reason,
        this cardinality may be used as a measure of inconsistency of the information
        about X." (p. 125 bzw. 141)

        :param X:
        :return:
        """
        if not X:
            X = self.X_

        #Do a thing with the cardinality of X, I guess
        pass

    def _pareto(self, n_gen:int=200) -> [np.ndarray, np.ndarray]:
        #problem = None

        #algorithm = NSGA2(ref_points=ref_points, pop_size=pop_size)

        termination = get_termination("n_gen", n_gen)

        results = minimize(self.problem_, self.algorithm_, termination, seed=1, verbose=True)

        pareto_front = results.F  # Objective values
        pareto_set = results.X  # Corresponding decision variables

        return pareto_front, pareto_set
