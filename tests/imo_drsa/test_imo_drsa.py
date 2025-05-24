import unittest
from unittest import TestCase

import numpy as np

from pymoo.problems import get_problem

from src.imo_drsa.imo_drsa import IMO_DRSA


class TestIMO_DRSA(TestCase):
    np.random.seed(42)

    def test_pareto_front(self):
        prob = get_problem("zdt1")
        universe = np.random.rand(200, 30)

        def make_F(i):
            return lambda x: x[0] + x[1] * i

        objectives = [make_F(i) for i in range(prob.n_obj)]

        # Pass by keyword to match __init__ signature
        imo = IMO_DRSA(universe=universe, objectives=objectives)
        X, F_out = imo.pareto_front(universe, constraints=None, n_gen=5)

        self.assertEqual(X.shape[1], prob.n_var)
        self.assertEqual(F_out.shape[1], prob.n_obj)

        self.assertTrue(np.all(F_out >= 0.0), "Some objective values < 0")
        self.assertTrue(np.all(F_out <= 1.0), "Some objective values > 1")

    def test_generate_constraints(self):
        # Single rule with thresholds matching x[0]=1, x[1]=2
        rules = [({0: 1.0, 1: 2.0}, 'd>=2', 0.5, 0.9, 'certain', 'up',
                  "[CERTAIN] IF f_1 >= 1.0 AND f_2 >= 2.0 THEN d >= 2 (support=0.50, confidence=0.90)"),
                 ({0: 0.5, 1: 1.5}, 'd>=2', 0.3, 0.7, 'possible', 'up',
                  "[POSSIBLE] IF f_0 >= 0.5 AND f_1 >= 1.5 THEN d >= 2 (support=0.30, confidence=0.70)")]

        model = IMO_DRSA()

        def make_F(i):
            return lambda x: x[i]

        model.objectives = [make_F(i) for i in range(2)]

        dr = model.generate_constraints(rules)

        x = [1, 2]
        # Both constraints from the first rule should be zero at x
        self.assertEqual(dr[0](x), 0)
        self.assertEqual(dr[1](x), 0)

        # And the two "possible" constraints should not be zero for x
        self.assertNotEqual(dr[2](x), 0)
        self.assertNotEqual(dr[3](x), 0)

if __name__ == '__main__':
    unittest.main()