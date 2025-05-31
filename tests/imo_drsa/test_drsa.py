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
        self.drsa = DRSA(F_pareto_gain_type=self.T, criteria=(0, 1), decision_attribute=self.d)

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
        self.drsa.direction = "up"
        gamma = self.drsa.quality((0, 1))
        # For the given data, consistency yields 2 out of 5 objects => 0.4
        self.assertAlmostEqual(gamma, 0.4, places=6)

    def test_find_reducts(self):
        self.drsa.direction="up"
        reducts = self.drsa.find_reducts()
        # Only the full set of criteria is a minimal reduct here
        self.assertEqual(reducts, [(0, 1)])

    def test_general(self):
        self.drsa.direction = "up"
        # Validate pipeline end-to-end
        T = np.array([[5, 7],
                      [3, 4],
                      [6, 2],
                      [4, 6],
                      [2, 5]])

        d = np.array([1, 1, 3, 2, 2])

        drsa = DRSA()
        drsa.fit(F_pareto_gain_type=T, criteria=(0, 1), decision_attribute=d)

        low_up = drsa.lower_approx_up((0, 1), threshold=2)
        up_up = drsa.upper_approx_up((0, 1), threshold=2)

        self.assertListEqual(low_up.tolist(), [False, False, True, False, False])
        self.assertListEqual(up_up.tolist(), [True, False, True, True, True])

        gamma = drsa.quality((0, 1))
        print(gamma)
        self.assertAlmostEqual(gamma, 0.4, places=6)

        reducts = drsa.find_reducts()
        self.assertEqual(reducts, [(0, 1)])

        rules = drsa.induce_decision_rules(criteria=(0, 1), threshold=2)

        # Must produce at least one rule
        self.assertTrue(len(rules) > 0)

class TestDRSA2D(TestCase):
    def setUp(self):
        # Simple 2D dataset
        self.pareto_set = np.array([[1, 2], [2, 1]])
        self.dec_attr = np.array([1, 2])
        self.criteria = (0, 1)
        self.drsa = DRSA(F_pareto_gain_type=self.pareto_set, criteria=self.criteria, decision_attribute=self.dec_attr)

    def test_positive_cone(self):
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(self.drsa.positive_cone(self.criteria), expected)

    def test_negative_cone(self):
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(self.drsa.negative_cone(self.criteria), expected)

    def test_lower_upper_approx_up(self):
        # For threshold 2
        lower_expected = np.array([False, True])
        upper_expected = np.array([False, True])
        lower = self.drsa.lower_approx_up(self.criteria, 2)
        upper = self.drsa.upper_approx_up(self.criteria, 2)
        np.testing.assert_array_equal(lower, lower_expected)
        np.testing.assert_array_equal(upper, upper_expected)

    def test_lower_upper_approx_down(self):
        # For threshold 1
        lower_expected = np.array([True, False])
        upper_expected = np.array([True, False])
        lower = self.drsa.lower_approx_down(self.criteria, 1)
        upper = self.drsa.upper_approx_down(self.criteria, 1)
        np.testing.assert_array_equal(lower, lower_expected)
        np.testing.assert_array_equal(upper, upper_expected)


class TestDRSA1D(TestCase):
    def setUp(self):
        # Simple 1D dataset
        self.pareto_set = np.array([[1], [2], [3]])
        self.dec_attr = np.array([1, 2, 3])
        self.criteria = (0,)
        self.drsa = DRSA(F_pareto_gain_type=self.pareto_set, criteria=self.criteria, decision_attribute=self.dec_attr)

    def test_quality_and_reducts(self):
        # Full quality should be 1.0
        q = self.drsa.quality(self.criteria)
        self.assertEqual(q, 1.0)
        # Only one criterion, so reduct is [(0,)]
        reducts = self.drsa.find_reducts()
        self.assertEqual(reducts, [(0,)])

    def test_induce_decision_rules_not_minimal_or_robust(self):
        # Threshold 2 for 'up' direction
        rules = self.drsa.induce_decision_rules(criteria=self.criteria, threshold=2, minimal=False)
        # Expect certain rules for indices 1 and 2, i.e. at least one rule with profile 0:2
        self.assertTrue(any(rule[0].get(0) == 2 for rule in rules if rule[4] == 'certain'))



class TestAssociationRules(TestCase):


    def test_find_association_rules_multiple_criteria(self):
        # Three‐objective dataset
        T = np.array([
            [5, 7, 1],
            [3, 4, 2],
            [6, 2, 6],
            [4, 6, 5],
            [2, 5, 3],
        ])

        rules = DRSA.find_association_rules(F_pareto=T, criteria=(0, 1, 2), min_support=0.1, min_confidence=0.8)


        summary, _ = DRSA.summarize_association_rules(rules)

        self.assertEqual(len(summary), 18)


    def test_find_association_rules(self):
        row1 = [i for i in range(5, 10)]
        row2 = [i for i in range(4, -1, -1)]

        T = np.array([row1, row2]).T

        rules = DRSA.find_association_rules(F_pareto=T, criteria=(0, 1), min_support=0.1, min_confidence=0.8)


        summary, s = DRSA.summarize_association_rules(rules)

        expected1 = ("If objective 1 is higher, objective 2 tends to be lower", 0.4, 1.0)
        expected2 = ("If objective 1 is higher, objective 2 tends to be higher", 0.2, 1.0)


        self.assertTrue(expected1 in summary)
        self.assertTrue(expected2 not in summary)





if __name__ == '__main__':
    unittest.main()
