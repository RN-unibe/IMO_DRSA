import sys
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
        self.drsa = DRSA(self.T, criteria=(0, 1), decision_attribute=self.d)

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
        self.drsa = DRSA(self.pareto_set, self.criteria, self.dec_attr)

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
        self.drsa = DRSA(self.pareto_set, self.criteria, self.dec_attr)

    def test_quality_and_reducts(self):
        # Full quality should be 1.0
        q = self.drsa.quality(self.criteria)
        self.assertEqual(q, 1.0)
        # Only one criterion, so reduct is [(0,)]
        reducts = self.drsa.find_reducts()
        self.assertEqual(reducts, [(0,)])

    def test_induce_decision_rules_not_minimal_or_robust(self):
        # Threshold 2 for 'up' direction
        rules = self.drsa.induce_decision_rules(criteria=self.criteria, direction='up', threshold=2, minimal=False,
                                                robust=False)
        # Expect certain rules for indices 1 and 2, i.e. at least one rule with profile 0:2
        self.assertTrue(any(rule[0].get(0) == 2 for rule in rules if rule[4] == 'certain'))


class TestDecisionRuleFormatting(unittest.TestCase):
    def setUp(self):
        self.pareto_set = np.array([[1]])
        self.dec_attr = np.array([1])
        self.criteria = (0,)
        self.drsa = DRSA(self.pareto_set, self.criteria, self.dec_attr)

    def test_make_rule_description(self):
        profile = {0: 5, 1: 10}
        desc = self.drsa.make_rule_description(profile, "d >= 2", support=0.75, confidence=0.80, kind='certain',
                                               direction='up')
        self.assertIn("CERTAIN", desc)
        self.assertIn("f_1 >= 5 AND f_2 >= 10", desc)
        self.assertIn("support=0.75", desc)
        self.assertIn("confidence=0.80", desc)


class TestAssociationRules(unittest.TestCase):

    def test_induce_association_rules_default(self):
        drsa = DRSA().fit(pareto_set=np.array([[1, 1], [2, 2], [3, 3]]), criteria=(0, 1),
                          decision_attribute=np.array([1, 2, 3]))

        rules = drsa.find_association_rules(criteria=(0, 1), min_support=0.1, min_confidence=0.1)
        # Should return a list of rule tuples
        self.assertIsInstance(rules, list)
        self.assertTrue(len(rules) >= 1, "Expected at least one association rule.")

        for rule in rules:
            # Each rule is a 7-tuple
            self.assertIsInstance(rule, tuple)
            self.assertEqual(len(rule), 7)

            profile, conclusion, support, confidence, kind, relation, desc = rule

            self.assertIsInstance(profile, tuple)
            self.assertIsInstance(conclusion, tuple)

            self.assertIsInstance(support, float)
            self.assertGreaterEqual(support, 0.0)
            self.assertLessEqual(support, 1.0)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

            self.assertIsInstance(kind, str)
            self.assertIsInstance(relation, str)
            self.assertIsInstance(desc, str)
            self.assertTrue(desc.startswith(f"[{kind.upper()}] IF"), f"Description does not start correctly: {desc}")

    def test_induce_association_rules_multiple_criteria(self):
        # Three‐objective dataset
        T = np.array([
            [5, 7, 1],
            [3, 4, 2],
            [6, 2, 6],
            [4, 6, 5],
            [2, 5, 3],
        ])
        d = np.array([1, 1, 3, 2, 2])
        drsa = DRSA().fit(T, (0, 1, 2), d)

        # Mine all rules (up to 2‐antecedents, 2‐consequents by default)
        rules = drsa.find_association_rules(min_support=0.0, min_confidence=0.0, max_antecedent=1, max_consequent=1)

        # We expect exactly three directed pairs:
        #   0→2, 2→0 and 1→2
        expected_relations = {'0->1', '1->0', '2->1', '0->2', '2->0', '1->2'}
        found_relations = {rule[5] for rule in rules}  # rule[5] is the 'relation' field

        print(found_relations)

        self.assertEqual(expected_relations, found_relations)

        # Sanity‐check the structure of each rule
        for lhs, rhs, support, confidence, kind, relation, desc in rules:
            # each side should be a tuple of (feat, threshold, sym, fn)
            self.assertTrue(isinstance(lhs, tuple) and all(len(cond) == 4 for cond in lhs))
            self.assertTrue(isinstance(rhs, tuple) and all(len(cond) == 4 for cond in rhs))
            self.assertEqual(kind, "assoc")
            self.assertIn("support=", desc)
            self.assertIn("confidence=", desc)

    def test_explain_association_rule_tuple(self):
        desc = "[ASSOC] IF f_0 >= 1 THEN f_1 >= 2 (support=0.20, confidence=0.30)"
        rule = ({0: 1}, {1: 2}, 0.2, 0.3, 'ASSOC', '->', desc)
        descs = DRSA.explain_rules([rule], verbose=False)
        self.assertEqual(descs, [desc])

    def test_explain_invalid_format(self):
        # Passing a non-sequence should raise TypeError
        with self.assertRaises(TypeError):
            DRSA.explain_rules([42], verbose=False)


if __name__ == '__main__':
    unittest.main()
