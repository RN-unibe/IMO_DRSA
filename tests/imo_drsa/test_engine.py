import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from pymoo.problems import get_problem

from src.imo_drsa.decision_maker import InteractiveDM, AutomatedDM
from src.imo_drsa.drsa import DRSA
from src.imo_drsa.engine import IMO_DRSAEngine



class TestIMO_DRSAEngine(TestCase):

    def test_pareto_front(self):
        prob = get_problem("bnh")

        engine = IMO_DRSAEngine().fit(prob)
        X, T = engine.calculate_pareto_front(n_gen=5)  # just to see if it works

        self.assertTrue(X is not None)
        self.assertTrue(T is not None)


    def test_generate_constraints(self):
        # Single rule with thresholds matching x[0]=1, x[1]=2
        rules = [({0: 1.0, 1: 2.0}, 'd>=2', 0.5, 0.9, 'certain', 'up',
                  "[CERTAIN] IF f_1 >= 1.0 AND f_2 >= 2.0 THEN d >= 2 (support=0.50, confidence=0.90)"),
                 ({0: 0.5, 1: 1.5}, 'd>=2', 0.3, 0.7, 'possible', 'up',
                  "[POSSIBLE] IF f_0 >= 0.5 AND f_1 >= 1.5 THEN d >= 2 (support=0.30, confidence=0.70)")]

        model = IMO_DRSAEngine()

        def make_F(i):
            return lambda x: x[i]

        model.objectives = [make_F(i) for i in range(2)]

        dr = model.generate_constraints(rules, elementwise=True)

        x = [1, 2]
        # Both constraints from the first rule should be zero at x
        self.assertEqual(dr[0](x), 0)
        self.assertEqual(dr[1](x), 0)

        # And the two "possible" constraints should not be zero for x
        self.assertNotEqual(dr[2](x), 0)
        self.assertNotEqual(dr[3](x), 0)


# ---------------------------------------------------------------------------------------------------------- #
# Testing it with different test problems with automated responses
# ---------------------------------------------------------------------------------------------------------- #
class TestIMO_DRSAEngineProblemSolving(TestCase):

    def setUp(self):
        self.verbose = False
        self.visualise = self.verbose
        self.to_file = self.verbose
        self.max_iter = 4
        np.random.seed(42)

    # ---------------------------------------------------------------------------------------------------------- #
    # Double-objective problems
    # ---------------------------------------------------------------------------------------------------------- #
    def test_solve_bnh(self):
        # Test if the solve function can terminate at all and to observe its behaviour
        dm = AutomatedDM(max_rounds=1)
        problem = get_problem("bnh")

        def f0(x):
            return 4 * x[0] * x[0] + 4 * x[1] * x[1]

        def f1(x):
            term1 = x[0] - 5
            term2 = x[1] - 5

            return term1 * term1 + term2 * term2

        objectives = [lambda x: -f0(x),  lambda x: -f1(x)]

        engine = IMO_DRSAEngine().fit(problem=problem, gain_type_objectives=objectives, verbose=self.verbose, to_file=self.to_file)

        success = engine.run(dm, max_iter=self.max_iter)

        self.assertTrue(success)

    def test_solve_osy(self):
        # Test if the solve function can terminate at all and to observe its behaviour
        dm = AutomatedDM(max_rounds=1)
        problem = get_problem("osy")

        def f0(x):
            x1, x2, x3, x4, x5, _ = x
            return -(25 * (x1 - 2) ** 2 +
                     (x2 - 2) ** 2 +
                     (x3 - 1) ** 2 +
                     (x4 - 4) ** 2 +
                     (x5 - 1) ** 2)

        def f1(x):
            return np.sum(np.asarray(x) ** 2)

        objectives = [lambda x: -f0(x),  lambda x: -f1(x)]

        engine = IMO_DRSAEngine().fit(problem=problem, gain_type_objectives=objectives, verbose=self.verbose, to_file=self.to_file)

        success = engine.run(dm, max_iter=self.max_iter)

        self.assertTrue(success)

    def test_solve_tnk(self):
        # Test if the solve function can terminate at all and to observe its behaviour
        dm = AutomatedDM(max_rounds=1)
        problem = get_problem("tnk")

        def f0(x):
            return np.asarray(x[0])

        def f1(x):
            return np.asarray(x[1])

        objectives = [lambda x: -f0(x),  lambda x: -f1(x)]

        engine = IMO_DRSAEngine().fit(problem=problem, gain_type_objectives=objectives, verbose=self.verbose, to_file=self.to_file)

        success = engine.run(dm, max_iter=self.max_iter)

        self.assertTrue(success)

    # ---------------------------------------------------------------------------------------------------------- #
    # Triple-objective problems
    # ---------------------------------------------------------------------------------------------------------- #
    def test_solve_dtlz1_3_obj(self):
        # Test if the solve function can terminate at all and to observe its behaviour
        dm = AutomatedDM(max_rounds=1)
        problem = get_problem("dtlz1")

        def dtlz1(n_obj):
            def g(x):
                x = np.asarray(x)
                n_var = x.size
                k = n_var - n_obj + 1
                xm = x[-k:]
                return -(100 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5))))) # "-" because they need to be gain type

            objectives = []
            for i in range(n_obj):
                objectives.append(lambda x, i=i: (0.5 * (1 + g(x))
                                                  * (np.prod(x[: n_obj - i - 1]) if (n_obj - i - 1) > 0 else 1)
                                                  * (1 if i == 0 else (1 - x[n_obj - i]))))
            return objectives

        objectives = dtlz1(3)

        engine = IMO_DRSAEngine().fit(problem=problem, gain_type_objectives=objectives, verbose=self.verbose, to_file=self.to_file)

        success = engine.run(dm, max_iter=self.max_iter)

        self.assertTrue(success)



class TestGenerateConstraintsEdgeCases(unittest.TestCase):

    def setUp(self):
        self.engine = IMO_DRSAEngine()



        self.engine.objectives = [
            lambda x: x[0],
            lambda x: 2.0 * x[1]
        ]

    def test_no_rules_returns_empty_list(self):
        constraints_none = self.engine.generate_constraints(selected_rules=None, elementwise=True)
        self.assertIsInstance(constraints_none, list)
        self.assertEqual(len(constraints_none), 0)

        constraints_empty = self.engine.generate_constraints(selected_rules=[], elementwise=True)
        self.assertEqual(len(constraints_empty), 0)

    def test_thresholds_strictly_enforced(self):
        # Edge: If x slightly above threshold, constraint should go negative.
        rule = ({0: 2.0}, 'c', 0.1, 0.2, 'k', 'd', 'desc')
        selected = [rule]
        cons = self.engine.generate_constraints(selected, elementwise=True)

        self.assertEqual(cons[0]([2.0, 99.0]), 0.0)
        val = cons[0]([2.1, 0.0])
        self.assertTrue(val < 0.0)

    def test_generate_constraints_with_non_integer_indices(self):
        rule_bad = ({5: 10.0}, 'c', 0.1, 0.2, 'k', 'd', 'desc')
        constraints = self.engine.generate_constraints([rule_bad], elementwise=True)

        self.assertEqual(len(constraints), 1)
        with self.assertRaises(IndexError):
            _ = constraints[0]([0.0, 0.0])  # no f5 in objectives

    def test_generate_constraints_elementwise_defaults_to_problem_flag(self):

        class DummyProblem:
            elementwise = False
        self.engine.problem = DummyProblem()

        rule = ({0: 1.0}, 'c', 0.1, 0.2, 'k', 'd', 'desc')
        self.engine.objectives = [lambda x: x[0]]

        constraints = self.engine.generate_constraints([rule])
        self.assertEqual(len(constraints), 1)

        X = np.array([[1.0], [0.5]])
        out = constraints[0](X)

        np.testing.assert_allclose(out, np.array([0.0, 0.5]))


class TestUndoLast(unittest.TestCase):

    def test_undo_last_restores_previous_state(self):
        engine = IMO_DRSAEngine()

        old_problem = object()
        old_X_pareto = np.array([[1.0, 2.0], [3.0, 4.0]])
        old_F_pareto = np.array([[10.0, 20.0], [30.0, 40.0]])
        old_rules = ['rule_old']
        old_sample_X = np.array([[5.0, 6.0]])
        old_sample_F = np.array([[50.0, 60.0]])

        history_entry = {
            'problem': old_problem,
            'X_pareto': old_X_pareto,
            'F_pareto': old_F_pareto,
            'X_pareto_sample': old_sample_X,
            'F_pareto_sample': old_sample_F,
            'rules': old_rules
        }
        engine.history.append(history_entry)

        engine.problem = object()       # different object
        engine.X_pareto = np.zeros((1, 1))
        engine.F_pareto = np.zeros((1, 1))
        engine.rules = []

        returned_Xs, returned_Fs = engine.undo_last()


        self.assertIs(engine.problem, old_problem)
        np.testing.assert_array_equal(engine.X_pareto, old_X_pareto)
        np.testing.assert_array_equal(engine.F_pareto, old_F_pareto)
        self.assertEqual(engine.rules, old_rules)

        np.testing.assert_array_equal(returned_Xs, old_sample_X)
        np.testing.assert_array_equal(returned_Fs, old_sample_F)

        self.assertEqual(len(engine.history), 0)

    def test_undo_last_when_history_empty_raises(self):
        engine = IMO_DRSAEngine()

        with self.assertRaises(IndexError):
            engine.undo_last()



if __name__ == '__main__':
    unittest.main()
