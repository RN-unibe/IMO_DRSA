from typing import Callable, List
from types import MethodType

import numpy as np

from pymoo.core.problem import Problem, ElementwiseProblem


class ProblemExtender():

    @staticmethod
    def enable_dynamic_constraints(problem: Problem):
        """
        Mutate a problem so that each call to _evaluate runs the original logic
        and then appends all constraints in problem._extra_constraints.
        """

        if not hasattr(problem, "_orig_evaluate"):
            problem._orig_evaluate = problem._evaluate
            problem._extra_constraints = None

            def _evaluate(self, x, out, *args, **kwargs):
                self._orig_evaluate(x, out, *args, **kwargs)

                if self._extra_constraints is not None:
                    G_base = out.get("G", None)

                    if self.elementwise:
                        G_extra = np.array([g(x) for g in self._extra_constraints])

                    else:
                        G_extra = np.column_stack([g(x) for g in self._extra_constraints])

                        if G_base is not None and G_base.ndim == 1:
                            G_base = G_base.reshape(-1, 1)

                    if G_base is None:
                        out["G"] = G_extra
                    else:
                        axis = 0 if self.elementwise else 1
                        out["G"] = np.concatenate([G_base, G_extra], axis=axis)

            problem._evaluate = MethodType(_evaluate, problem)

            def add_constraints(self, constraints: List[Callable]):
                """
                Add more inequality constraints on the fly.
                Each g(x) should return either a scalar (elementwise=True)
                or a 1d array length pop_size (elementwise=False).
                """
                if self._extra_constraints is None:
                    self._extra_constraints = []

                self._extra_constraints.extend(constraints)
                self.n_ieq_constr = getattr(self, "n_ieq_constr", 0) + len(constraints)

            problem.add_constraints = MethodType(add_constraints, problem)

        return problem


# ---------------------------------------------------------------------------------------------------------- #
# Dummy base-problems for testing
# ---------------------------------------------------------------------------------------------------------- #

class DummyElementwiseProblem(ElementwiseProblem):
    """
    A minimal base Problem with no constraints: F(x) = sum(x).
    """

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=np.array([0.0, 0.0]),
                         xu=np.array([1.0, 1.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x.sum()


class DummyBatchProblem(Problem):
    """
    A base Problem that already has one inequality constraint:
      G_base(x) = x0 - x1  (must be <= 0)
    """

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_ieq_constr=1,
                         xl=np.array([0.0, 0.0]),
                         xu=np.array([1.0, 1.0]),
                         elementwise=False)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x.sum(axis=1)
        out["G"] = x[:, 0] - x[:, 1]


# ---------------------------------------------------------------------------------------------------------- #
# Settable Dummy Problem
# ---------------------------------------------------------------------------------------------------------- #
class DynamicDummyBatchProblem(Problem):
    """
    To directly set specific objectives and constraints in the test.
    """

    def __init__(self, F, H=None, G=None,
                 n_var=2,
                 n_obj=1,
                 n_ieq_constr=0,
                 n_eq_constr=0,
                 xl=np.array([0.0, 0.0]),
                 xu=np.array([1.0, 1.0])):

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_ieq_constr=n_ieq_constr,
                         n_eq_constr=n_eq_constr,
                         xl=xl,
                         xu=xu,
                         elementwise=False)

        self.F = F
        self.H = H
        self.G = G

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.F

        if self.H is not None:
            out["H"] = self.H

        if self.G is not None:
            out["G"] = self.G
