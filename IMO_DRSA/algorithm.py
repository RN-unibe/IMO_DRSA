from typing import Dict, Any, List

import numpy as np
from numpy import ndarray

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

from decision_maker import BaseDM
from IMO_DRSA.problem import DRSAProblem

import types
import json
from datetime import datetime

class DRSA():

    def __init__(self,
                 U,
                 F,
                 d,
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
        self.P_ = F # This contains a subset of the FUNCTIONS of F_, NOT it's INDICES! I think this is given by DM

        self.pareto_front_ = pareto_front
        self.pareto_set_ = pareto_set
        self.rules_ = {}

        self.iteration_logs = []



    def create_constraints(self):
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


        pass





    def _create_constraint(self, id, args, body):
        """
        Create a function dynamically and bind it as a rule.

        Example usage:
        self.create_function('greet', ['id'], "print(f'Hello, {name}!')")

        self.greet("world")  # Output: Hello, world!
        print(self.list_rules())  # Output: ['greet']

        :param id:
        :param args:
        :param body:
        :return:
        """
        args_str = ', '.join(args)
        func_code = f"def {id}(self, {args_str}):\n"
        func_code += '\n'.join(f"    {line}" for line in body.split('\n'))

        local_ns = {}
        exec(func_code, globals(), local_ns)

        func = local_ns[id]
        bound_rule = types.MethodType(func, self)
        setattr(self, id, bound_rule)
        self.rules_[id] = bound_rule



    def _gen_sorting(self):
        pass

    def _gen_unions(self, t):
        upward = [i for i in range(len(self.d_)) if self.d_[i] >= t]
        downward = [i for i in range(len(self.d_)) if self.d_[i] <= t]

        return upward, downward


    def _gen_p_dominating_set(self, x):
        return [y for y in self.U_ if all(f_i(y) >= f_i(x) for f_i in self.P_)]

    def _gen_p_dominated_set(self, x):
        return [y for y in self.U_ if all(f_i(y) <= f_i(x) for f_i in self.P_)]


    def _gen_p_lower_approx(self, cl_ge_t, t):
        return [x for x in self.U_ if all(self.d_[y] >= t for y in self._gen_p_dominating_set(x))]

    def _gen_p_upper_approx(self, cl_ge_t, t):
        return [x for x in self.U_ if any(y in cl_ge_t for y in self._gen_p_dominated_set(x))]


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

        subsets = chain.from_iterable(combinations(self.rules_, r) for r in range(len(self.rules_) + 1))

        for subset in subsets:
            if self._set_is_complete(subset):
                # get some way to see the "better" rule subset
                return False

        return True

    def _set_is_complete(self, rules) -> bool:

        for x in self.U_:
            if not any(rule(x) for rule in rules):
                print(f"No rule matched x = {x}")
                return False

        return True


    def set_pareto_front(self, pareto_front):
        self.pareto_front_ = pareto_front

    def set_pareto_set(self, pareto_set):
        self.pareto_set_ = pareto_set

    def set_d(self, d):
        self.d_ = d

    def log_iteration(self, rules, constraints, good_solutions, comments=None):
        """
        Example usage:
        self.log_iteration(rules=active_rules,
                   constraints=generated_constraints,
                   good_solutions=dm_selected_ids,
                   comments="Rule set accepted by DM in iteration 3")

        :param rules:
        :param constraints:
        :param good_solutions:
        :param comments:
        :return:
        """

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": len(self.iteration_logs) + 1,
            "rules": rules,  # a list of rule objects or rule strings
            "constraints": constraints,  # functions or human-readable forms
            "good_solutions": good_solutions,  # list of solution vectors or indices
            "comments": comments  # optional explanation (e.g., selected by DM)
        }

        self.iteration_logs.append(log_entry)


    def explain(self, to_file=False):
        """
        Show which rules were used in each iteration.

        :return:
        """
        print("=== DRSA Iteration Summary ===")

        for i, log in enumerate(self.iteration_logs):
            print(f"\n--- Iteration {i + 1} ---")
            print("Selected 'good' solutions:")
            for x in log["good_solutions"]:
                print(f"  x = {x}")
            print("Generated Rules:")
            for rule in log["rules"]:
                print(f"  Rule: {str(rule)}")
            print("Applied Constraints:")
            for c in log["constraints"]:
                print(f"  Constraint: {c}")  # maybe use rule.label if function
            if log["comments"]:
                print("Notes:", log["comments"])

        if to_file:
            timestamp = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

            with open(f"drsa_log_{timestamp}.json", "w") as f:
                json.dump(self.iteration_logs, f, indent=4)


########################################################################################################################
#
########################################################################################################################


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




