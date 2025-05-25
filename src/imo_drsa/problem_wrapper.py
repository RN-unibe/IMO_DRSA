from typing import Callable, List

import numpy as np

from pymoo.core.problem import Problem, ElementwiseProblem


class ProblemWrapper(Problem):
    def __init__(self, base_problem:Problem,
                 constraints:List[Callable]=None,
                 **kwargs):
        """
        Wrap any pymoo Problem and add extra inequality constraints.

        :param base_problem: an existing pymoo.core.problem.Problem
        :param constraints: a list of callables
        :param kwargs: any extra args you’d normally pass (e.g. elementwise=False)
        """

        # total constraints = whatever the base had + our extras
        n_ieq_constr = getattr(base_problem, "n_ieq_constr", 0)
        n_constr = getattr(base_problem, "n_constr", 0)

        if constraints is not None:
            n_ieq_constr = n_ieq_constr + len(constraints)
            n_constr = n_constr + len(constraints)


        super().__init__(n_var=base_problem.n_var,
                         n_obj=base_problem.n_obj,
                         n_constr=n_constr,
                         n_ieq_constr=n_ieq_constr,
                         xl=base_problem.xl,
                         xu=base_problem.xu,
                         elementwise=base_problem.elementwise,
                         **kwargs)

        self.base_problem = base_problem
        self.constraints = constraints


    def _evaluate(self, x, out, *args, **kwargs):
        self.base_problem._evaluate(x, out, *args, **kwargs)

        if self.constraints is not None:
            G_base = out.get("G", None)
            if G_base is not None:
                G_base = np.asarray(G_base)

                if not self.elementwise and  G_base.ndim == 1:
                    G_base = G_base.reshape(-1, 1)

            if self.elementwise:
                extra = np.array([g(x) for g in self.constraints])
                G_extra = extra
            else:
                g_extra = [g(x) for g in self.constraints]
                G_extra = np.column_stack(g_extra)

            if G_base is None:
                out["G"] = G_extra
            else:
                if self.elementwise:
                    out["G"] = np.concatenate([G_base, G_extra], axis=0)
                else:
                    out["G"] = np.concatenate([G_base, G_extra], axis=1)


    def set_constraints(self, constraints:List[Callable]):
        self.constraints = constraints
        self.n_ieq_constr = self.base_problem.n_ieq_constr + len(constraints)


    def extend_constraints(self, new_constraints:List[Callable]):
        self.constraints.extend(new_constraints)
        self.n_ieq_constr = self.n_ieq_constr + len(new_constraints)



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
                        n_constr=0,
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
                        n_constr=1,
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
                        n_constr=n_ieq_constr + n_eq_constr,
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












