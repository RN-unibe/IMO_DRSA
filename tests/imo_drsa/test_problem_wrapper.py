import unittest
from unittest import TestCase

import numpy as np

from src.imo_drsa.problem_wrapper import ProblemWrapper, DummyElementwiseProblem, DummyBatchProblem


class TestProblemWrapper(TestCase):

    # ---------------------------------------------------------------------------------------------------------- #
    # Elementwise Problem
    # ---------------------------------------------------------------------------------------------------------- #
    def test_no_extra_constraints(self):
        base = DummyElementwiseProblem()
        wrapper = ProblemWrapper(base, constraints=None)

        X = np.array([[0.1, 0.2],
                      [0.3, 0.4],
                      [0.5, 0.6]])

        w_res = wrapper.evaluate(X)
        b_res = base.evaluate(X)

        self.assertListEqual(w_res.tolist(), b_res.tolist())



    def test_extra_constraints(self):
        base = DummyElementwiseProblem()

        def g1(x):
            return x.sum()

        def g2(x):
            return x[0] * x[1]

        wrapper = ProblemWrapper(base, constraints=[g1, g2])
        X = np.array([[0.1, 0.2],
                      [0.3, 0.4],
                      [0.5, 0.6]])

        res = wrapper.evaluate(X)

        print(res)


    # ---------------------------------------------------------------------------------------------------------- #
    # Batch Problem
    # ---------------------------------------------------------------------------------------------------------- #

    def test_batch_no_extra_constraints(self):
        base = DummyBatchProblem()
        wrapper = ProblemWrapper(base, constraints=None)

        X = np.array([[0.1, 0.2],
                      [0.3, 0.4],
                      [0.5, 0.6]])

        w_res = wrapper.evaluate(X)
        b_res = base.evaluate(X)

        self.assertListEqual(w_res.tolist(), b_res.tolist())



    def test_batch_extra_constraints(self):
        base = DummyBatchProblem()

        def g1(x):
            return x.sum()

        def g2(x):
            return x[0] * x[1]

        wrapper = ProblemWrapper(base, constraints=[g1, g2])
        X = np.array([[0.1, 0.2],
                      [0.3, 0.4],
                      [0.5, 0.6]])

        res = wrapper.evaluate(X)

        print(res)


if __name__ == "__main__":
    unittest.main()
