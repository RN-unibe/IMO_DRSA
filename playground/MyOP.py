import numpy as np

from pymoo.core.problem import ElementwiseProblem



class MyOP(ElementwiseProblem):
    """

    """

    def __init__(self, **kwargs):
        super().__init__(elementwise=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        :param x:
        :param out:
        :param args:
        :param kwargs:
        :return:
        """
        f1 = 100 * (x[0] ** 2 + x[1] ** 2)
        f2 = (x[0] - 1) ** 2 + x[1] ** 2

        g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


