import unittest
from unittest import TestCase

import numpy as np

from src.imo_drsa.problem_extender import ProblemExtender, DummyElementwiseProblem, DummyBatchProblem, \
    DynamicDummyBatchProblem


class TestProblemWrapper(TestCase):

    # ---------------------------------------------------------------------------------------------------------- #
    # Elementwise Problem
    # ---------------------------------------------------------------------------------------------------------- #
    def test_elementwise_no_extra_constraints(self):
        base = DummyElementwiseProblem()
        wrapper = ProblemExtender.enable_dynamic_constraints(base)

        X = np.array([[0.1, 0.2],
                      [0.3, 0.4],
                      [0.5, 0.6]])

        w_res = wrapper.evaluate(X)
        b_res = base.evaluate(X)

        self.assertListEqual(w_res.tolist(), b_res.tolist())

    def test_elementwise_extra_constraints(self):
        def g0(x):
            return x.sum()

        def g1(x):
            return x[0] * x[1]

        X = np.array([[0.1, 0.2],
                      [0.3, 0.4],
                      [0.5, 0.6]])

        F = X.sum(axis=1)
        G = [[g0(x) for x in X], [g1(x) for x in X]]

        dynamic = DynamicDummyBatchProblem(F=F, G=G, n_var=X.shape[1], n_obj=1, n_ieq_constr=2, n_eq_constr=0)
        base = DummyElementwiseProblem()
        wrapper = ProblemExtender.enable_dynamic_constraints(base)
        wrapper.add_constraints(constraints=[g0, g1])

        w_res = wrapper.evaluate(X)
        d_res = dynamic.evaluate(X)

        self.assertListEqual(w_res[0].tolist(), d_res[0].tolist())
        self.assertListEqual(w_res[1].tolist(), d_res[1].tolist())

    # ---------------------------------------------------------------------------------------------------------- #
    # Batch Problem
    # ---------------------------------------------------------------------------------------------------------- #
    def test_batch_no_extra_constraints(self):
        base = DummyBatchProblem()
        wrapper = ProblemExtender.enable_dynamic_constraints(base)

        X = np.array([[0.1, 0.2],
                      [0.3, 0.4],
                      [0.5, 0.6]])

        F = X.sum(axis=1)
        G = X[:, 0] - X[:, 1]
        dynamic = DynamicDummyBatchProblem(F=F, G=G, n_var=X.shape[1], n_obj=1, n_ieq_constr=1, n_eq_constr=0)

        w_res = wrapper.evaluate(X)
        d_res = dynamic.evaluate(X)

        self.assertListEqual(w_res[0].tolist(), d_res[0].tolist())
        self.assertListEqual(w_res[1].tolist(), d_res[1].tolist())

    def test_batch_extra_constraints(self):
        def g0(X):
            return X[:, 0] - X[:, 1]

        def g1(X):
            return X.sum(axis=1)

        def g2(X):
            return X[:, 0] * X[:, 1]

        X = np.array([[0.1, 0.2],
                      [0.3, 0.4],
                      [0.5, 0.6]])

        F = X.sum(axis=1)
        G = np.stack([g0(X), g1(X), g2(X)], axis=1)

        dynamic = DynamicDummyBatchProblem(F=F, G=G, n_var=X.shape[1], n_obj=1, n_ieq_constr=3, n_eq_constr=0)
        base = DummyBatchProblem()
        wrapper = ProblemExtender.enable_dynamic_constraints(base)
        wrapper.add_constraints(constraints=[g1, g2])

        w_res = wrapper.evaluate(X)
        d_res = dynamic.evaluate(X)

        self.assertListEqual(w_res[0].tolist(), d_res[0].tolist())
        self.assertListEqual(w_res[1].tolist(), d_res[1].tolist())


if __name__ == "__main__":
    unittest.main()
