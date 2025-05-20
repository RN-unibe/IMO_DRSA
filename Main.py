import numpy as np


from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from IMO_DRSA.drsa import IMO_DRSA
from IMO_DRSA.problem import DRSAProblem


if __name__ == "__main__":

    drsa_rob = DRSAProblem(n_var=2, n_obj=2, n_ieq_constr=2, xl=np.array([-2, -2]), xu=np.array([2, 2]))
    ref_points = np.array([[0.5, 0.2], [0.1, 0.6]])
    imo_drsa = IMO_DRSA(ref_points=ref_points, pop_size=100)

    res = minimize(drsa_rob,
                   imo_drsa,
                   ("n_gen", 100),
                   verbose=False,
                   seed=1)

    plot = Scatter()
    plot.add(res.F, edgecolor="red", facecolor="none")
    plot.show()