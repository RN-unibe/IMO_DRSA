from typing import Callable, List

import numpy as np

from pymoo.core.problem import Problem


class ProblemWrapper(Problem):
    def __init__(self, base_problem:Problem,
                 constraints:List[Callable]=None,
                 **kwargs):
        """
        Wrap any pymoo Problem and add extra inequality constraints.

        :param base_problem:    an existing pymoo.core.problem.Problem
        :param constraints:     a list of callables, each c(X)->array of shape (n_samples,)
        :param kwargs:          any extra args you’d normally pass (e.g. elementwise=False)
        """

        # total constraints = whatever the base had + our extras
        n_ieq_constr = getattr(base_problem, "n_ieq_constr", 0)

        if constraints is None:
            constraints = []

        n_ieq_constr = n_ieq_constr + len(constraints)

        super().__init__(n_var=base_problem.n_var,
                         n_obj=base_problem.n_obj,
                         n_constr=base_problem.n_constr,
                         n_ieq_constr=n_ieq_constr,
                         xl=base_problem.xl,
                         xu=base_problem.xu,
                         **kwargs)

        self.base_problem = base_problem
        self.constraints = constraints


    def _evaluate(self, x, out, *args, **kwargs):
        self.base_problem._evaluate(x, out, *args, **kwargs)

        G_base = out.get("G", None)

        if G_base is None: # no base constraints
            G_base = np.zeros((x.shape[0] if x.ndim > 1 else 1, 0), float)
        else:
            G_base = np.atleast_2d(G_base)

        if x.ndim == 1: #for problems, which do elementwise evaluations
            g_vals = [g(x) for g in self.constraints]
            G_extra = np.atleast_2d(g_vals)

        else: #for problems, which do batch evaluation
            g_vals = [g(x) for g in self.constraints]
            G_extra = np.vstack(g_vals).T

        out["G"] = np.concatenate([G_base, G_extra], axis=1)


    def set_constraints(self, constraints:List[Callable]):
        self.constraints = constraints
        self.n_ieq_constr = self.base_problem.n_ieq_constr + len(constraints)


    def extend_constraints(self, new_constraints:List[Callable]):
        self.constraints.extend(new_constraints)
        self.n_ieq_constr = self.n_ieq_constr + len(new_constraints)
























