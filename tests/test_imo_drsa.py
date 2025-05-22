from unittest import TestCase
from unittest.mock import patch
from unittest import main


import numpy as np

from pymoo.problems import get_problem

from IMO_DRSA.drsa import DRSA
from IMO_DRSA.imo_drsa import IMO_DRSA


class TestIMO_DRSA(TestCase):
    np.random.seed(42)

    def test_pareto_front(self):

        prob = get_problem("zdt1")

        U = np.random.rand(200, 30)

        def make_F(i):
            return lambda x: x[0] + x[1] * i

        F = [make_F(i) for i in range(prob.n_obj)]

        imo = IMO_DRSA(U=U, F=F)
        X, F_out = imo.pareto_front(U, F, constraints=None, pop_size=10, n_gen=5)

        self.assertEqual(X.shape[1], prob.n_var)
        self.assertEqual(F_out.shape[1], prob.n_obj)
        self.assertTrue(np.all(F_out >= 0.0), "Some objective values < 0")
        self.assertTrue(np.all(F_out <= 1.0), "Some objective values > 1")

    def test_get_association_rules(self):
        self.fail()


    def test_generate_constraints(self):
        rules = [
            ({0: 1.0, 1: 2.0}, 'd>=2', 0.5, 0.9, 'certain', 'up'),
            #({2: 3.0}, 'd>=2', 0.3, 0.7, 'possible', 'up),
        ]

        model = IMO_DRSA()
        drsa = DRSA()
        drsa.explain_rules(rules, verbose=True)

        def make_F(i):
            return lambda x: x[i]

        F = [make_F(i) for i in range(2)]

        dr = model.generate_constraints(F, rules)

        self.assertTrue(dr[0]([1, 2]) == 0)
        self.assertTrue(dr[1]([1, 2]) == 0)

