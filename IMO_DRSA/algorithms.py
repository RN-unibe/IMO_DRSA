import numpy as np
import pandas as pd
from matplotlib.pyplot import xcorr
from numpy.f2py.auxfuncs import throw_error

from sklearn.base import BaseEstimator

from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling


class IMO_DRSA():

    def __init__(self, X=None, y=None,
                 problem:ElementwiseProblem=None,
                 algorithm:GeneticAlgorithm=None,
                 max_iter:int=200):
        """
        Interactive Multiobjective Optimization using Dominance-based Rough Set Approach (IMO-DRSA).

        :param X:
        :param y:
        :param problem: The problem defined in problems.py
        :param ref_point_: the reference point given by the DM for the dialogue stage
        :param max_iter:
        """
        self.X_ = X
        self.y_ = y
        self.max_iter_ = max_iter

        self.problem_ = problem
        self.algorithm_ = algorithm#NSGA2(pop_size=pop_size, n_offsprings=n_offsprings, sampling=sampling, crossover=crossover, mutation=mutation, eliminate_duplicates=eliminate_duplicates)

        self.drsa_ = None # DRSA(...)
        self.pareto_front_ = None
        self.pareto_set_ = None


    def solve(self, X=None, y=None, n_gen:int=200) -> bool:
        """

        :param X:
        :param y:
        :param n_gen:

        :return: True, if the process finished successfully. False, otherwise.
        """

        if X is not None:
            self.X_ = X

        if y is not None:
            self.y_ = y

        if not self.X_:
            raise Exception("X is undefined")

        if not self.y_:
            raise Exception("y is undefined")

        n_iter:int = 0
        pareto_front, pareto_set = None, None

        while n_iter < self.max_iter_:
        # LOOP:
            # CALCULATION:
            #1. Generate Pareto Front
            pareto_front, pareto_set = self._pareto(n_gen=n_gen)

            #DIALOGUE:
            #2. Present the sample to the DM, possibly together with association rules showing relationships between
            #   attainable values of objective functions in the Pareto optimal set.

            #3. If the DM is satisfied with one solution from the sample, then this is the most preferred solution and the
            #   procedure stops. Otherwise continue.

            #4. Ask the DM to indicate a subset of “good” solutions in the sample.

            #CALCULATION:
            #5. DRSA

            #DIALOGUE:
            #6. Present the obtained set of rules to the DM.
            #7. Ask the DM to select the decision rules most adequate to their preferences.

            # CALCULATION:
            #8. Adjoin the constraints coming from the rules selected in Step 7 to the set of constraints imposed on the
            #   Pareto optimal set, in order to focus on a part interesting from the point of view of DM’s preferences.

            #9. Repeat
            n_iter += 1

        self.pareto_front_ = pareto_front
        self.pareto_set_ = pareto_set

        return True



    def _pareto(self, n_gen:int=200) -> [np.ndarray, np.ndarray]:

        termination = get_termination("n_gen", n_gen)

        results = minimize(self.problem_, self.algorithm_, termination, seed=1, verbose=True)

        pareto_front = results.F  # Objective values
        pareto_set = results.X  # Corresponding decision variables

        return pareto_front, pareto_set


    def visualise(self):
        pass


class DRSA(BaseEstimator):
    """

    """

    def __init__(self, pareto_front):
        """
        Dominance-based Rough Set Approach (DRSA) "is a methodology of multiple criteria decision analysis aiming at obtaining
        a representation of the DM’s preferences in terms of easily understandable “if ..., then ...” decision rules,
        on the basis of some exemplary decisions (past decisions or simulated decisions) given by the DM." (Greco et al., 2008)

        """



    def fit(self, X, y=None):
        """
        X and y together = The Sorting Examples

        "The criteria and the class assignment considered within IMO_DRSA
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


        #Do a thing with the cardinality of X, I guess
        pass


