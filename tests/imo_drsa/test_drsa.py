import sys
import unittest
from unittest import TestCase

import numpy as np

from src.imo_drsa.drsa import DRSA

# Ensure the module path includes the directory where drsa.py is located
sys.path.insert(0, '/mnt/data')


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
        drsa.fit(T, criteria=(0, 1), decision_attribute=d)

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


class TestDRSA2D(unittest.TestCase):
    def setUp(self):
        # Simple 2D dataset
        self.pareto_set = np.array([[1, 2], [2, 1]])
        self.dec_attr = np.array([1, 2])
        self.criteria = (0, 1)
        self.drsa = DRSA(self.pareto_set, self.dec_attr, self.criteria)

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


class TestDRSA1D(unittest.TestCase):
    def setUp(self):
        # Simple 1D dataset
        self.pareto_set = np.array([[1], [2], [3]])
        self.dec_attr = np.array([1, 2, 3])
        self.criteria = (0,)
        self.drsa = DRSA(self.pareto_set, self.dec_attr, self.criteria)

    def test_quality_and_reducts(self):
        # Full quality should be 1.0
        q = self.drsa.quality(self.criteria)
        self.assertEqual(q, 1.0)
        # Only one criterion, so reduct is [(0,)]
        reducts = self.drsa.find_reducts()
        self.assertEqual(reducts, [(0,)])

    def test_induce_decision_rules(self):
        # Threshold 2 for 'up' direction
        rules = self.drsa.induce_decision_rules(criteria=self.criteria, direction='up', threshold=2)
        # Expect certain rules for indices 1 and 2, i.e. at least one rule with profile 0:2
        self.assertTrue(any(rule[0].get(0) == 2 for rule in rules if rule[4] == 'certain'))


class TestDecisionRuleFormatting(unittest.TestCase):
    def setUp(self):
        self.pareto_set = np.array([[1]])
        self.dec_attr = np.array([1])
        self.criteria = (0,)
        self.drsa = DRSA(self.pareto_set, self.dec_attr, self.criteria)

    def test_make_rule_description(self):
        profile = {0: 5, 1: 10}
        desc = self.drsa.make_rule_description(profile, "d >= 2", support=0.75, confidence=0.80, kind='certain',
                                               direction='up')
        self.assertIn("CERTAIN", desc)
        self.assertIn("f_1 >= 5 AND f_2 >= 10", desc)
        self.assertIn("support=0.75", desc)
        self.assertIn("confidence=0.80", desc)


class TestAssociationRules(unittest.TestCase):
    def setUp(self):
        # Reuse the 2D DRSA for association rules
        self.drsa = DRSA(np.array([[1, 1], [2, 2], [3, 3]]), np.array([1, 2, 3]), (0, 1))

    def test_find_single_rule(self):
        # Uniform arrays to get a clear rule
        f_i = np.array([1, 1, 1])
        f_j = np.array([0, 0, 0])
        rule = self.drsa.find_single_rule(f_i, f_j, min_support=0.1, min_confidence=0.1)
        self.assertIsInstance(rule, dict)
        self.assertEqual(rule['if'], "x >= 1")
        self.assertEqual(rule['then'], "y >= 0")
        self.assertAlmostEqual(rule['support'], 1.0)
        self.assertAlmostEqual(rule['confidence'], 1.0)

    def test_find_association_rules(self):
        pareto_set = np.array([[1, 1], [2, 2], [3, 3]])
        drsa2 = DRSA(pareto_set, np.array([1, 2, 3]), (0, 1))
        rules = drsa2.find_association_rules(pareto_set, criteria=(0, 1), min_support=0.1, min_confidence=0.1)
        self.assertIn((0, 1), rules)
        rule = rules[(0, 1)]
        if rule:
            self.assertIn('f_0(x)', rule['if'])
            self.assertIn('f_1(x)', rule['then'])


class TestExplainRules(unittest.TestCase):
    def test_explain_association_rule(self):
        rule = {'if': 'x >= 1', 'then': 'y >= 2', 'support': 0.2, 'confidence': 0.3}
        descs = DRSA.explain_rules([rule], verbose=False)
        self.assertEqual(descs[0], "if x >= 1 then y >= 2  (support=0.20, confidence=0.30)")

    def test_explain_invalid_format(self):
        # Expect a generic Exception due to invalid rule format
        with self.assertRaises(Exception):
            DRSA.explain_rules([42], verbose=False)


if __name__ == '__main__':
    unittest.main()
