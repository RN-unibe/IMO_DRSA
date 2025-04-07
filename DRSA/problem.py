import numpy as np

from pymoo.core.problem import ElementwiseProblem



class MyOP(ElementwiseProblem):
    """

    """

    def __init__(self, **kwargs):
        super().__init__(elementwise=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        pass