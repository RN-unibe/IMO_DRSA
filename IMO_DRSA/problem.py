import numpy as np

from pymoo.core.problem import ElementwiseProblem


class DRSAProblem(ElementwiseProblem):
    def __init__(self, drsa_constr, n_var, n_obj, n_constr, xl, xu):


        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         xl=xl,
                         xu=xu)

        self.drsa_constraints_ = drsa_constr


    def set_drsa_constraints(self, drsa_constr):
        self.drsa_constraints_ = drsa_constr


    def _evaluate(self, x, out, *args, **kwargs):
        # TODO

        # Normal objectives
        out["F"] = [...]

        # Standard constraints (if any)
        out["G"] = [...]

        # Add DRSA-induced constraints
        if self.drsa_constraints_:
            drsa_g = [c(x) for c in self.drsa_constraints_]
            if "G" in out:
                out["G"] = np.concatenate([out["G"], drsa_g])
            else:
                out["G"] = drsa_g


    def _calc_pareto_front(self, *args, **kwargs):
        pass

    def _calc_pareto_set(self, *args, **kwargs):
        pass




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