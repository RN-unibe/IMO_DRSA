from unittest import TestCase
from unittest.mock import patch
from unittest import main


import numpy as np

from pymoo.problems import get_problem

from IMO_DRSA.imo_drsa import IMO_DRSA

class DummyResult:
    def __init__(self, X, F):
        self.X = X
        self.F = F

class TestIMO_DRSA(TestCase):

    def test_init_sets_attributes_and_I(self):
        U = [(-1, 1), (0, 2)]
        F = [lambda x: x[0]**2, lambda x: x[1]**2, lambda x: x[0] + x[1]]
        d = "decision_attr"
        imo = IMO_DRSA(U=U, F=F, d=d)

        # U, F, P, d
        self.assertIs(imo.U, U)
        self.assertIs(imo.F, F)
        self.assertIs(imo.P, F)
        self.assertEqual(imo.d, d)

        # I should be [0, 1] for len(F)=3
        self.assertEqual(imo.I, [0, 1])

    def test_set_P_filters_P_to_reduct(self):
        imo = IMO_DRSA()
        imo.P = np.array([10, 20, 30, 40])
        imo.set_P(reduct=[20, 40])
        np.testing.assert_array_equal(imo.P, np.array([20, 40]))


    @patch('IMO_DRSA.imo_drsa.minimize')
    def test_pareto_front_returns_X_and_F(self, mock_minimize):
        # Prepare simple bounds
        U = [(-1, 1), (-2, 2)]
        F = [lambda x: x[0]**2, lambda x: x[1]**2]
        imo = IMO_DRSA(U=U, F=F, d=None)

        # Dummy outputs
        dummy_X = np.array([[0.0, 0.0], [1.0, -1.0]])
        dummy_F = np.array([[0.0, 0.0], [1.0, 1.0]])
        mock_minimize.return_value = DummyResult(X=dummy_X, F=dummy_F)

        # Call with specific pop_size and n_gen
        X_out, F_out = imo.pareto_front(dummy_X, constraints=None, pop_size=50, n_gen=5)

        # Check that minimize was called with the correct termination tuple
        called_kwargs = mock_minimize.call_args[1]
        self.assertEqual(called_kwargs.get('termination'), ('n_gen', 5))
        self.assertFalse(called_kwargs.get('verbose', True) is True)

        # And that we got back our dummy arrays
        np.testing.assert_array_equal(X_out, dummy_X)
        np.testing.assert_array_equal(F_out, dummy_F)


class TestIMO_DRSAParetoFrontWithZDT1(TestCase):

    def test_pareto_front_on_zdt1(self):
        prob = get_problem("zdt1")

        U = list(zip(prob.xl, prob.xu))

        def make_F(i):
            return lambda x: prob.evaluate(x.reshape(1, -1))[0, i]
        F = [make_F(i) for i in range(prob.n_obj)]

        imo_drsa = IMO_DRSA(U=U, F=F)
        X, F_out = imo_drsa.pareto_front(U, constraints=None, pop_size=10, n_gen=5)


        self.assertEqual(X.shape[1], prob.n_var)

        self.assertEqual(F_out.shape[1], prob.n_obj)

        self.assertTrue(np.all(F_out >= 0.0), "Some objective values < 0")
        self.assertTrue(np.all(F_out <= 1.0), "Some objective values > 1")

