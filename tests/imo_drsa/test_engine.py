import unittest
from unittest import TestCase

import numpy as np

from pymoo.problems import get_problem

from src.imo_drsa.decision_maker import DummyDM, InteractiveDM, AutomatedDM
from src.imo_drsa.engine import IMO_DRSAEngine



class TestIMO_DRSAEngine(TestCase):

    def test_pareto_front(self):
        prob = get_problem("bnh")

        engine = IMO_DRSAEngine().fit(prob)
        # engine.problem.add_constraints()
        engine.calculate_pareto_front(n_gen=5)  # just to see if it works

        # engine.visualise()

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
        self.verbose = True
        self.visualise = self.verbose
        self.max_iter = 4
        np.random.seed(42)

    # ---------------------------------------------------------------------------------------------------------- #
    # Double-objective problems
    # ---------------------------------------------------------------------------------------------------------- #
    def test_solve_bnh(self):
        # Test if the solve function can terminate at all and to observe its behaviour
        dm = AutomatedDM()
        problem = get_problem("bnh")

        def f0(x):
            return 4 * x[0] * x[0] + 4 * x[1] * x[1]

        def f1(x):
            term1 = x[0] - 5
            term2 = x[1] - 5

            return term1 * term1 + term2 * term2

        objectives = [f0, f1]

        engine = IMO_DRSAEngine().fit(problem=problem, objectives=objectives, verbose=self.verbose)

        success = engine.run(dm, visualise=self.visualise, max_iter=self.max_iter)

        self.assertTrue(success)

    def test_solve_osy(self):
        # Test if the solve function can terminate at all and to observe its behaviour
        dm = AutomatedDM()
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

        objectives = [f0, f1]

        engine = IMO_DRSAEngine().fit(problem=problem, objectives=objectives, verbose=self.verbose)

        success = engine.run(dm, visualise=self.visualise, max_iter=self.max_iter)

        self.assertTrue(success)

    def test_solve_tnk(self):
        # Test if the solve function can terminate at all and to observe its behaviour
        dm = AutomatedDM()
        problem = get_problem("tnk")

        def f0(x):
            return np.asarray(x[0])

        def f1(x):
            return np.asarray(x[1])

        objectives = [f0, f1]

        engine = IMO_DRSAEngine().fit(problem=problem, objectives=objectives, verbose=self.verbose)

        success = engine.run(dm, visualise=self.visualise, max_iter=self.max_iter)

        self.assertTrue(success)

    # ---------------------------------------------------------------------------------------------------------- #
    # Triple-objective problems
    # ---------------------------------------------------------------------------------------------------------- #
    def test_solve_dtlz1_3_obj(self):
        # Test if the solve function can terminate at all and to observe its behaviour
        dm = AutomatedDM()
        problem = get_problem("dtlz1")

        def dtlz1(n_obj):
            def g(x):
                x = np.asarray(x)
                n_var = x.size
                k = n_var - n_obj + 1
                xm = x[-k:]
                return 100 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5))))

            objectives = []
            for i in range(n_obj):
                objectives.append(lambda x, i=i: (0.5 * (1 + g(x))
                                                  * (np.prod(x[: n_obj - i - 1]) if (n_obj - i - 1) > 0 else 1)
                                                  * (1 if i == 0 else (1 - x[n_obj - i]))))
            return objectives

        objectives = dtlz1(3)

        engine = IMO_DRSAEngine().fit(problem=problem, objectives=objectives, verbose=self.verbose)

        success = engine.run(dm, visualise=self.visualise, max_iter=self.max_iter)

        self.assertTrue(success)



if __name__ == '__main__':
    unittest.main()
