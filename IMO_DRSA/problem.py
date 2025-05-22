import numpy as np

from pymoo.core.problem import ElementwiseProblem


class DRSAProblem(ElementwiseProblem):
    def __init__(self, P, n_var, n_obj, xl, xu, constr=None, n_ieq_constr=0):
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)
        self.P = P
        self.constr = constr

    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate objectives
        vals = np.array([f(x) for f in self.P])
        out["F"] = vals

        # Evaluate constraints (g(x) <= 0)
        if self.n_ieq_constr > 0:
            G_vals = np.array([g(x) for g in self.constr])
            out["G"] = G_vals







