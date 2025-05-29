import unittest
from unittest import TestCase

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem

from src.imo_drsa.problem_extender import ProblemExtender, DummyElementwiseProblem, DummyBatchProblem, \
    DynamicDummyBatchProblem


class TestBasicProblemExtender(TestCase):

    # ---------------------------------------------------------------------------------------------------------- #
    # Elementwise Problem
    # ---------------------------------------------------------------------------------------------------------- #
    def test_elementwise_no_extra_constraints(self):
        # Just enabling dynamic constraints should not change the output (without constraints)
        base = DummyElementwiseProblem()
        wrapper = ProblemExtender.enable_dynamic_constraints(base)

        X = np.array([[0.1, 0.2],
                      [0.3, 0.4],
                      [0.5, 0.6]])

        w_res = wrapper.evaluate(X)
        b_res = base.evaluate(X)

        self.assertListEqual(w_res.tolist(), b_res.tolist())

    def test_elementwise_extra_constraints(self):
        # Just enabling dynamic constraints should not change the output (with constraints)

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
        # Just enabling dynamic constraints should not change the output (without constraints)

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
        # Just enabling dynamic constraints should not change the output (with constraints)

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


# ---------------------------------------------------------------------------------------------------------- #
# Test with actual Test Problem from pymoo
# ---------------------------------------------------------------------------------------------------------- #

class TestProblemExtenderBNH(TestCase):

    def setUp(self):
        # Load the BNH test problem (2 vars, 2 objectives, 2 constraints)
        self.problem = get_problem("bnh")
        # Enable dynamic constraints
        self.ext_problem = ProblemExtender.enable_dynamic_constraints(self.problem)

    def test_base_evaluate_unchanged(self):
        # A small batch of two points
        X = np.array([[0.5, 0.5],
                      [1.0, 0.5]])
        # Evaluate original constraints via _orig_evaluate
        out_orig = {}
        self.ext_problem._orig_evaluate(X, out_orig)
        G_base = np.asarray(out_orig["G"])

        # Evaluate via the new wrapped _evaluate (no extra constraints yet)
        out_wrapped = {}
        self.ext_problem._evaluate(X, out_wrapped)
        G_wrapped = np.asarray(out_wrapped["G"])

        # They must match exactly
        np.testing.assert_array_equal(G_wrapped, G_base)

    def test_add_single_dynamic_constraint(self):
        X = np.array([[0.5, 0.5],
                      [1.0, 0.5]])
        # Original G
        out0 = {}
        self.ext_problem._orig_evaluate(X, out0)
        G0 = np.asarray(out0["G"])
        n0 = G0.shape[1]

        # Define and add one new inequality constraint: sum(x)-1 <= 0
        def g_extra(x):
            # x is a (pop_size, n_var) array
            return x[:, 0] + x[:, 1] - 1.0

        self.ext_problem.add_constraints([g_extra])

        # Now evaluate again
        out1 = {}
        self.ext_problem._evaluate(X, out1)
        G1 = np.asarray(out1["G"])

        # Should have one more column
        self.assertEqual(G1.shape[0], G0.shape[0])
        self.assertEqual(G1.shape[1], n0 + 1)

        # First n0 columns unchanged
        np.testing.assert_array_equal(G1[:, :n0], G0)

        # Last column equals our g_extra
        expected = g_extra(X).reshape(-1, 1)
        np.testing.assert_array_almost_equal(G1[:, -1:].reshape(-1, 1), expected)

        # n_ieq_constr should have increased by 1
        self.assertEqual(self.ext_problem.n_ieq_constr, n0 + 1)

    def test_add_multiple_dynamic_constraints(self):
        # Capture current count
        before = self.ext_problem.n_ieq_constr

        # Define two more simple constraints
        def g1(x):
            return x[:, 0] - 0.2

        def g2(x):
            return x[:, 1] - 0.3

        self.ext_problem.add_constraints([g1, g2])

        # Should increase by 2
        self.assertEqual(self.ext_problem.n_ieq_constr, before + 2)

    def test_enable_dynamic_constraints_idempotent(self):
        # Wrapping twice should not re-wrap
        eval_before = self.ext_problem._evaluate
        ProblemExtender.enable_dynamic_constraints(self.ext_problem)
        eval_after = self.ext_problem._evaluate
        self.assertIs(eval_before, eval_after)

    def test_add_constraints_method_exists(self):
        # After enabling, add_constraints should be available
        self.assertTrue(hasattr(self.ext_problem, "add_constraints"))
        self.assertTrue(callable(self.ext_problem.add_constraints))

    def test_optimisation_with_additional_constraints(self):
        # Adding new constraints should not break minimize
        algorithm = NSGA2(pop_size=10)

        def g1(x):
            return x[:, 0] - 0.2

        res_prev = minimize(self.problem, algorithm, termination=('n_gen', 10), verbose=False)

        self.ext_problem.add_constraints([g1])
        res_post = minimize(self.ext_problem, algorithm, termination=('n_gen', 10), verbose=False)

        self.assertEqual(res_prev.G.shape[0], res_post.G.shape[0])
        self.assertNotEqual(res_prev.G.shape[1], res_post.G.shape[1])


if __name__ == "__main__":
    unittest.main()
