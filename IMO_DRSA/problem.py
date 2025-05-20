import numpy as np

from pymoo.core.problem import ElementwiseProblem


class DRSAProblem(ElementwiseProblem):
    def __init__(self, P, constr, n_var, n_obj, n_ieq_constr, xl, xu):
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)
        self.P = P
        self.constr = constr

    def _evaluate(self, X, out, *args, **kwargs):
        # Evaluate objectives
        vals = np.array([[f(x) for f in self.P] for x in X])
        out["F"] = vals

        # Evaluate constraints (g(x) <= 0)
        if self.n_ieq_constr > 0:
            G_vals = np.array([[g(x) for g in self.constr] for x in X])
            out["G"] = G_vals




class ExampleProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]



