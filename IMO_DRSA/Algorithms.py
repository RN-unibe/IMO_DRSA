import numpy as np
from numpy import ndarray

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

from itertools import chain, combinations

from DecisionMaker import BaseDM

class DRSA():
    """

    """

    def __init__(self, U, F, d,
                 DM:BaseDM,
                 pareto_front=None,
                 pareto_set=None):
        """
        Dominance-based Rough Set Approach (DRSA) "is a methodology of multiple criteria decision analysis aiming at obtaining
        a representation of the DM’s preferences in terms of easily understandable “if ..., then ...” decision rules,
        on the basis of some exemplary decisions (past decisions or simulated decisions) given by the DM." (Greco et al., 2008)

        """
        self.U_ = U
        self.F_ = F
        self.d_ = d
        self.DM_ = DM
        # I = [i for i in range(F.size)]
        self.P_ = None

        self.pareto_front_ = pareto_front
        self.pareto_set_ = pareto_set
        self.rules_ = None


    def gen_rules(self) -> ndarray:
        """
        X and y together = The Sorting Examples

        "The criteria and the class assignment considered within IMO_DRSA
        correspond to the condition attributes and the decision attribute, respectively,
        in the classical Rough Set Approach (Pawlak, 1991). For example, in multiple
        criteria sorting of cars, an example of decision is an assignment of a particular
        car evaluated on such criteria as maximum speed, acceleration, price and
        fuel consumption to one of three classes of overall quality: “bad”, “medium”,
        “good”." (Greco et al., 2008)

        :param F: Condition attributes (Matrix, where X_ij = f_i(x_j))
        :param d: Decision attributes d : U -> {1, ...,m}
        :return:
        """


        return self.rules_




    def _gen_sorting(self):
        pass

    def _gen_upward_union(self):
        pass

    def _gen_downward_union(self):
        pass

    def _gen_dominating_set(self, x):
        pass

    def _gen_dominated_set(self, x):
        pass

    def _gen_p_lower_approx(self, cl_t):
        # P_
        # assert inclusion property
        # assert complementarity property
        pass


    def _gen_p_upper_approx(self, cl_t):
        # P-
        # assert inclusion property
        # assert complementarity property
        pass
    
    def _gen_p_boundaries(self, cl_t):
        # Bn_p
        pass

    def _gen_p_consistent_set(self, bn_p, cl_t_upper, cl_t_lower):
        # Cn_p
        pass

    def _gen_reduct(self, cl):
        pass

    def _gen_core(self, cl):
        pass


    def _is_robust(self, rule) -> bool:
        return False

    def _relative_support(self):
        pass

    def confidence_ration(self):
        pass

    def approx_quality(self, cn_p):
        # gamma_p(Cl)
        pass


    def set_is_minimal(self) -> bool:
        if not self._set_is_complete(self.rules_):
            return False

        flat = self.rules_.flatten()
        subsets = chain.from_iterable(combinations(flat, r) for r in range(len(flat) + 1))

        for subset in subsets:
            if self._set_is_complete(subset):
                # get some way to see the "better" rule subset
                return False

        return True

    def _set_is_complete(self, rules: ndarray) -> bool:

        for x in self.U_:
            if not any(rule(x) for rule in rules):
                print(f"No rule matched x = {x}")
                return False

        return True


    def set_pareto_front(self, pareto_front):
        self.pareto_front_ = pareto_front

    def set_pareto_set(self, pareto_set):
        self.pareto_set_ = pareto_set





class IMO_DRSA():

    def __init__(self, U=None, F=None, d=None, DM=None,
                 problem:ElementwiseProblem=None,
                 algorithm:GeneticAlgorithm=None,
                 max_iter:int=200):
        """
        Interactive Multi-objective Optimization using Dominance-based Rough Set Approach (IMO-DRSA).

        :param U: the universe of objects
        :param F: the set of all objective functions
        :param d: the decision attribute
        :param problem: The problem defined in Problems.py
        :param ref_point_: the reference point given by the DM for the dialogue stage
        :param max_iter:
        """
        self.DM_ = DM
        self.U_ = U
        self.F_ = F
        self.d_ = d
        self.max_iter_ = max_iter

        self.problem_ = problem
        self.algorithm_ = algorithm#NSGA2(pop_size=pop_size, n_offsprings=n_offsprings, sampling=sampling, crossover=crossover, mutation=mutation, eliminate_duplicates=eliminate_duplicates)

        self.pareto_front_ = None # Maybe also previous set, or assign the new results to non-global var
        self.pareto_set_ = None

        self.rules_ = None #ndarray of Rules


    def solve(self, n_gen:int=200, visualise:bool=False) -> bool:
        """
        :param n_gen:

        :return: True, if the process finished successfully. False, otherwise.
        """

        if not self.F_:
            raise Exception("X is undefined")

        if not self.d_:
            raise Exception("y is undefined")

        drsa = DRSA(self.U_, self.F_, self.d_, self.DM_)
        pareto_front, pareto_set = None, None

        n_iter:int = 0
        while n_iter < self.max_iter_:
        # LOOP:
            # CALCULATION:
            #1. Generate Pareto Front
            pareto_front, pareto_set = self._pareto(n_gen=n_gen)

            #DIALOGUE:
            #2. Present the sample to the DM, possibly together with association rules showing relationships between
            #   attainable values of objective functions in the Pareto optimal set.

            if visualise:
                self._visualise(pareto_front, pareto_set)
            #get input from DM

            self.DM_.give_input(pareto_front, pareto_set)

            #3. If the DM is satisfied with one solution from the sample, then this is the most preferred solution and the
            #   procedure stops. Otherwise, continue.

            #4. Ask the DM to indicate a subset of “good” solutions in the sample.

            #CALCULATION:
            #5. DRSA
            drsa.set_pareto_front(pareto_front)
            drsa.set_pareto_set(pareto_set)
            self.rules_ = drsa.gen_rules()

            #DIALOGUE:
            #6. Present the obtained set of rules to the DM.
            #7. Ask the DM to select the decision rules most adequate to their preferences.

            accepted_rules = self.DM_.select_rules(self.rules_, pareto_front, pareto_set)

            # CALCULATION:
            #8. Adjoin the constraints coming from the rules selected in Step 7 to the set of constraints imposed on the
            #   Pareto optimal set, in order to focus on a part interesting from the point of view of DM’s preferences.

            self.incorporate_rules(accepted_rules)

            #9. Repeat
            n_iter += 1

        self.pareto_front_ = pareto_front
        self.pareto_set_ = pareto_set

        return True



    def _pareto(self, n_gen:int=200) -> [np.ndarray, np.ndarray]:

        # TODO: Incorporate the rules
        termination = get_termination("n_gen", n_gen)

        results = minimize(self.problem_, self.algorithm_, termination, seed=1, verbose=True)

        pareto_front = results.F  # Objective values
        pareto_set = results.X  # Corresponding decision variables

        return pareto_front, pareto_set


    def _visualise(self, pareto_front, pareto_set) -> None:
        pass

    def incorporate_rules(self, accepted_rules):
        self.rules_ = accepted_rules # Is this enough?




