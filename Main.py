import numpy as np


from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from playground.MyOP import MyOP
from playground.MyAlgorithm import MyAlgorithm


if __name__ == "__main__":

    problem = MyOP(n_var=2, n_obj=2, n_ieq_constr=2, xl=np.array([-2, -2]), xu=np.array([2, 2]))
    ref_points = np.array([[0.5, 0.2], [0.1, 0.6]])
    algorithm = MyAlgorithm(ref_points=ref_points, pop_size=100)

    res = minimize(problem,
                   algorithm,
                   ("n_gen", 100),
                   verbose=False,
                   seed=1)

    plot = Scatter()
    plot.add(res.F, edgecolor="red", facecolor="none")
    plot.show()