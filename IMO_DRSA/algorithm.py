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
from collections import defaultdict

class DRSA():

    def __init__(self,
                 U:ndarray,
                 F,
                 d,
                 DM:BaseDM,
                 pareto_front=None,
                 pareto_set=None,
                 vectorised_f:bool=True):
        """
        Dominance-based Rough Set Approach (DRSA) "is a methodology of multiple criteria decision analysis aiming at obtaining
        a representation of the DM’s preferences in terms of easily understandable “if ..., then ...” decision rules,
        on the basis of some exemplary decisions (past decisions or simulated decisions) given by the DM." (Greco et al., 2008)



        """


        self.U_ = U
        self.F_ = F
        self.d_ = d
        self.DM_ = DM

        self.n_vars_ = len(self.U_)
        self.n_classes_ = 0 #Init later?

        # I = [i for i in range(F.size)]
        self.P_ = F # This contains a subset of the FUNCTIONS of F_, NOT it's INDICES!

        if not vectorised_f:
            self.f_F_ = np.array([[f(x) for f in self.F_] for x in self.U_], dtype=float)
            self.f_P_ = np.array([[f(x) for f in self.P_] for x in self.U_], dtype=float)

        else :
            self.f_F_ = np.column_stack([f(self.U_) for f in self.F_])
            self.f_P_ = np.column_stack([f(self.U_) for f in self.P_])

        self.Cl_ = None # The sorting of U, instance of set()
        self.upward_ = None
        self.downward_ = None

        self.p_dominating_ = None
        self.p_dominated_ = None

        self.pareto_front_ = pareto_front
        self.pareto_set_ = pareto_set
        self.rules_ = {}

        self.iteration_logs = []

        self.dominates_ = {}
        self.dominated_by_ = {}

        self.lower_approx_geq_ = {}
        self.lower_approx_leq_ = {}
        self.upper_approx_geq_ = {}
        self.upper_approx_leq_ = {}

        self.bn_P_t_upper_ = {}
        self.bn_P_t_lower_ = {}

    def fit(self):
        """
        Full DRSA pipeline:
          1. Sort U into decision classes (from self.d_)
          2. Compute Cl≥_t and Cl≤_t unions
          3. Compute dominance cones D⁺(x) and D⁻(x)
          4. Compute P-lower and P-upper approximations
          5. Compute boundary regions

        """

        # Step 1 – Sort U into classes
        self._sorting()  # must populate self.Cl_ as a list of sets
        self.n_classes_ = len(self.Cl_)

        # Step 2 – Compute class unions
        self._compute_unions()

        # Step 3 – Compute all dominance cones
        self.p_dominating_ = {}
        self.p_dominated_ = {}
        for x in self.U_:
            self._compute_P_dominance_sets(x)

        # Step 4 – Initialize approximation dicts
        self.lower_approx_geq_ = {}
        self.lower_approx_leq_ = {}
        self.upper_approx_geq_ = {}
        self.upper_approx_leq_ = {}

        self._compute_P_approximations()

        # Step 5 – Compute boundary regions
        self.bn_P_t_upper_ = {}
        self.bn_P_t_lower_ = {}
        self.p_boundaries()

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



    def _sorting(self):
        """
        Groups U into decision classes based on self.d_ (should return class indices like 0, 1, 2).
        Populates self.Cl_ as a list of sets, sorted by class index.
        """

        classes = defaultdict(set)
        for i, label in enumerate(self.d_):
            classes[label].add(i)

        # Sort by class index to ensure order: C0, C1, C2, ...
        self.Cl_ = [classes[i] for i in sorted(classes)]


    def _compute_unions(self):
        """

        """
        for t in range(0, len(self.Cl_) - 1):
            self.upward_[t] = [set().union(*self.Cl_[i:]) for i in range(t, len(self.Cl_))]
            self.downward_[t] = [set().union(*self.Cl_[:i + 1]) for i in range(t)]

    def _compute_P_dominance_sets(self, x: int):
        """
        Computes:
          - The P-dominating set of object x:
              D⁺(x) = { y ∈ U | f_P(x) ≥ f_P(y) }
          - The P-dominated set of object x:
              D⁻(x) = { y ∈ U | f_P(x) ≤ f_P(y) }

        Stores the result in:
            self.p_dominating_[x] = set of indices y dominated by x
            self.p_dominated_[x]  = set of indices y dominating x

        :param x: Index of the object in self.f_P_
        """

        dominates_mask = np.all(self.f_P_[x] >= self.f_P_, axis=1)
        dominated_mask = np.all(self.f_P_[x] <= self.f_P_, axis=1)

        self.p_dominating_[x] = set(np.where(dominates_mask)[0])
        self.p_dominated_[x] = set(np.where(dominated_mask)[0])

    def _compute_P_approximations(self):
        for t in range(0, len(self.Cl_) - 1):
            Cl_geq_t = self.upward_[t]  # Cl_t^{>=}
            Cl_leq_t = self.downward_[t]  # Cl_t^{<=}

            self.lower_approx_geq_[t] = {x for x in self.U_ if self.p_dominating_[x].issubset(Cl_geq_t)}
            self.lower_approx_leq_[t] = {x for x in self.U_ if self.p_dominated_[x].issubset(Cl_leq_t)}

            self.upper_approx_geq_[t] = {x for x in self.U_ if not self.p_dominated_[x].isdisjoint(Cl_geq_t)}
            self.upper_approx_leq_[t] = {x for x in self.U_ if not self.p_dominating_[x].isdisjoint(Cl_leq_t)}

    def p_boundaries(self):
        # Bn_p
        for t in range(0, len(self.Cl_)-1):
            self.bn_P_t_upper_[t] = self.upper_approx_geq_[t] - self.lower_approx_geq_[t]
            self.bn_P_t_lower_[t] = self.upper_approx_leq_[t] - self.lower_approx_leq_[t]

    def compute_p_consistent_set(self):
        """
        Computes the P-consistent set Cn_P
        and stores it in self.consistent_set_
        """
        boundary_union = set()

        # Union over all t for both boundaries
        for t in range(self.n_classes_):
            if t in self.bn_P_t_upper_:
                boundary_union.update(self.bn_P_t_upper_[t])
            if t in self.bn_P_t_lower_:
                boundary_union.update(self.bn_P_t_lower_[t])

        self.p_consistent_set_ = set(self.U_) - boundary_union


    def is_p_consistent(self, x):

        if x in self.p_consistent_set_ :
            return True

        return False

    def reduct(self):
        gamma_p = self.approx_quality()
        # TODO: Need to be able to do everything it with F as well!!!!!




    def core(self, cl):
        pass


    def is_robust(self, rule) -> bool:
        return False

    def relative_support(self):
        pass

    def confidence_ration(self):
        pass

    def approx_quality(self):
        # gamma_p(Cl)
        return len(self.p_consistent_set_) / len(self.U_)


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




