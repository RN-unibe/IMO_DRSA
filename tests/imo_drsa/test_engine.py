import unittest
from unittest import TestCase

import numpy as np

from pymoo.problems import get_problem

from src.imo_drsa.decision_maker import DummyDM, InteractiveDM
from src.imo_drsa.engine import IMO_DRSAEngine


class TestIMO_DRSAEngine(TestCase):
    np.random.seed(42)

    def test_pareto_front(self):
        prob = get_problem("bnh")

        engine = IMO_DRSAEngine().fit(prob)
        #engine.problem.add_constraints()
        engine.get_pareto_front(n_gen=5) # just to see if it works

        #engine.visualise()


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



    def test_solve(self):
        dm = DummyDM()
        problem = get_problem("bnh")

        def f0(x):
            return 4 * x[0] * x[0] + 4 * x[1] * x[1]

        def f1(x):
            term1 = x[0] - 5
            term2 = x[1] - 5

            return term1 * term1 + term2 * term2

        objectives = [f0, f1]

        engine = IMO_DRSAEngine().fit(problem=problem, objectives=objectives)

        engine.solve(dm)




if __name__ == '__main__':
    unittest.main()