import unittest
from unittest import TestCase

import numpy as np

from src.imo_drsa.drsa import DRSA


class TestDRSA(TestCase):
    np.random.seed(42)

    def setUp(self):
        self.T = np.array([[5, 7],
                           [3, 4],
                           [6, 2],
                           [4, 6],
                           [2, 5]])
        self.d = np.array([1, 1, 3, 2, 2])
        # Initialize DRSA with full criteria
        self.drsa = DRSA(self.T, self.d, criteria=(0, 1))

    def test_positive_cone(self):
        pos = self.drsa.positive_cone((0, 1))

        expected = np.array([
            [True, True, False, True, True],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, True, False, True, True],
            [False, False, False, False, True]])

        self.assertTrue(np.array_equal(pos, expected), f"Positive cone incorrect: {pos}")

    def test_negative_cone(self):
        neg = self.drsa.negative_cone((0, 1))
        # Negative cone should be the transpose of the positive cone
        pos = self.drsa.positive_cone((0, 1))

        expected = pos.T

        self.assertTrue(np.array_equal(neg, expected), f"Negative cone incorrect: {neg}")

    def test_approximations(self):
        low_up = self.drsa.lower_approx_up((0, 1), threshold=2)
        up_up = self.drsa.upper_approx_up((0, 1), threshold=2)

        self.assertListEqual(low_up.tolist(), [False, False, True, False, False])
        self.assertListEqual(up_up.tolist(), [True, False, True, True, True])

    def test_quality(self):
        gamma = self.drsa.quality((0, 1))
        # For the given data, consistency yields 2 out of 5 objects => 0.4
        self.assertAlmostEqual(gamma, 0.4, places=6)

    def test_find_reducts(self):
        reducts = self.drsa.find_reducts()
        # Only the full set of criteria is a minimal reduct here
        self.assertEqual(reducts, [(0, 1)])

    def test_generate_association_rules(self):
        # Using min_support=1 to only accept rules covering all objects in premise
        rules = self.drsa.find_association_rules(self.T, (0, 1), min_support=1)
        self.assertIsInstance(rules, dict)

        # With two criteria, only one pair (0,1) should appear
        self.assertEqual(set(rules.keys()), {(0, 1)})

        rule = rules[(0, 1)]
        self.assertIsInstance(rule, dict)

        for key in ('if', 'then', 'support', 'confidence'):
            self.assertIn(key, rule)

        self.assertGreaterEqual(rule['support'], 0)
        self.assertLessEqual(rule['support'], 1)
        self.assertGreaterEqual(rule['confidence'], 0)
        self.assertLessEqual(rule['confidence'], 1)


    # The following tests validate three-objective rules
    def test_generate_association_rules_three_objectives(self):
        T = np.array([[5, 7, 1],
                      [3, 4, 2],
                      [6, 2, 6],
                      [4, 6, 5],
                      [2, 5, 3]])

        d = np.array([1, 1, 3, 2, 2])

        drsa = DRSA(T, d, criteria=(0, 1, 2))
        rules = drsa.find_association_rules(T, (0, 1, 2), min_support=1)

        expected_pairs = {(0, 1), (0, 2), (1, 2)}

        self.assertIsInstance(rules, dict)
        self.assertEqual(set(rules.keys()), expected_pairs)

        for pair, rule in rules.items():
            self.assertIsInstance(rule, dict, f"Rule for pair {pair} is None or not a dict")

            for key in ('if', 'then', 'support', 'confidence'):
                self.assertIn(key, rule, f"Missing '{key}' in rule for {pair}")

            self.assertGreaterEqual(rule['support'], 0)
            self.assertLessEqual(rule['support'], 1)
            self.assertGreaterEqual(rule['confidence'], 0)
            self.assertLessEqual(rule['confidence'], 1)


    def test_general(self):
        # Validate pipeline end-to-end
        T = np.array([[5, 7],
                      [3, 4],
                      [6, 2],
                      [4, 6],
                      [2, 5]])

        d = np.array([1, 1, 3, 2, 2])

        drsa = DRSA()
        drsa.fit(T, d, criteria=(0, 1))

        low_up = drsa.lower_approx_up((0, 1), threshold=2)
        up_up = drsa.upper_approx_up((0, 1), threshold=2)

        self.assertListEqual(low_up.tolist(), [False, False, True, False, False])
        self.assertListEqual(up_up.tolist(), [True, False, True, True, True])

        gamma = drsa.quality((0, 1))
        self.assertAlmostEqual(gamma, 0.4, places=6)

        reducts = drsa.find_reducts()
        self.assertEqual(reducts, [(0, 1)])

        rules = drsa.induce_decision_rules(criteria=(0, 1), direction='up', threshold=2)

        # Must produce at least one rule
        self.assertTrue(len(rules) > 0)


if __name__ == '__main__':
    unittest.main()
