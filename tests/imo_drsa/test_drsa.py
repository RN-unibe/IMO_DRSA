import sys
import unittest
from typing import io
from unittest import TestCase

import numpy as np

from src.imo_drsa.drsa import DRSA


class TestDRSA(TestCase):
    np.random.seed(42)

    def setUp(self):
        self.F = np.array([[5, 7],
                           [3, 4],
                           [6, 2],
                           [4, 6],
                           [2, 5]])
        self.d = np.array([1, 1, 3, 2, 2])
        # Initialize DRSA with full criteria
        self.drsa = DRSA().fit(F_pareto_gain_type=self.F, criteria=(0, 1), decision_attribute=self.d)

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


        self.assertTrue(np.array_equal(neg, pos.T), f"Negative cone incorrect: {neg}")

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
        self.drsa = DRSA().fit(F_pareto_gain_type=self.pareto_set, criteria=self.criteria, decision_attribute=self.dec_attr)

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
        self.drsa = DRSA().fit(F_pareto_gain_type=self.pareto_set, criteria=self.criteria, decision_attribute=self.dec_attr)

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


class TestDRSAEdgeCases(unittest.TestCase):

    def setUp(self):
        # A tiny 2x2 dataset for basic DRSA calls
        self.simple_2x2 = np.array([[1, 2],
                                     [2, 1]])
        self.simple_classes_2x2 = np.array([1, 2])

        # DRSA instance that has NOT been fitted
        self.drsa_unfitted = DRSA()

        # DRSA instance fitted on the simple 2x2
        self.drsa_2x2 = DRSA().fit(F_pareto_gain_type=self.simple_2x2, criteria=(0, 1), decision_attribute=self.simple_classes_2x2)

    # -------------------------------------------------------------------------
    # Methods called before fit() must raise AssertionError
    # -------------------------------------------------------------------------
    def test_methods_before_fit_raise(self):
        dr = DRSA()
        with self.assertRaises(AssertionError):
            _ = dr.positive_cone((0,))
        with self.assertRaises(AssertionError):
            _ = dr.negative_cone((0,))
        with self.assertRaises(AssertionError):
            _ = dr.lower_approx_up((0,), threshold=1)
        with self.assertRaises(AssertionError):
            _ = dr.upper_approx_down((0,), threshold=1)
        with self.assertRaises(AssertionError):
            _ = dr.quality((0,))
        with self.assertRaises(AssertionError):
            _ = dr.find_reducts()
        with self.assertRaises(AssertionError):
            _ = dr.core()
        with self.assertRaises(AssertionError):
            _ = dr.induce_decision_rules(criteria=(0,), threshold=1)
        with self.assertRaises(AssertionError):
            # subsumes requires is_fit = True
            fake_rule = ({0: 1}, "x is 'good'", 0.5, 1.0, "certain", "up", "desc")
            _ = dr.subsumes(fake_rule, fake_rule)
        with self.assertRaises(AssertionError):
            # is_robust also requires is_fit = True
            fake_rule = ({0: 1}, "x is 'good'", 0.5, 1.0, "possible", "up", "desc")
            _ = dr.is_robust(fake_rule)

    # -------------------------------------------------------------------------
    # Test core(...) on data with MULTIPLE reducts
    # -------------------------------------------------------------------------
    def test_core_multiple_reducts(self):
        # Construct a dataset where each single criterion is by itself sufficient
        # 4x2 F:
        #   [1,1] -> class 1
        #   [1,2] -> class 2
        #   [2,1] -> class 2
        #   [2,2] -> class 2
        F = np.array([
            [1, 1],
            [2, 2],
            [2, 2],
            [2, 2]
        ])
        dec = np.array([1, 2, 2, 2])

        dr = DRSA().fit(F_pareto_gain_type=F, criteria=(0, 1), decision_attribute=dec, direction = "up")

        # Both (0,) and (1,) are minimal reducts, because either criterion alone
        # can separate class=2 from class=1 perfectly.
        reducts = dr.find_reducts()
        print(reducts)
        self.assertCountEqual(reducts, [(0,), (1,)])

        # core = intersection of all reducts = empty tuple
        core_set = dr.core(reducts=reducts)

        self.assertEqual(core_set, ())

    # -------------------------------------------------------------------------
    # Test subsumes(...) logic explicitly (both "up" and "down")
    # -------------------------------------------------------------------------
    def test_subsumes_logic_up_and_down(self):
        # "Up" direction: use drsa_2x2 (F = [[1,2],[2,1]], dec = [1,2])
        dr = self.drsa_2x2

        # r1: {0:2, 1:1} covers only object with f0>=2 AND f1>=1
        r1 = ({0: 2, 1: 1}, "x is 'good'", 0.5, 1.0, "certain", "up", "desc1")
        # r2: {0:2} weaker; any object with f0>=2
        r2 = ({0: 2}, "x is 'good'", 0.5, 1.0, "certain", "up", "desc2")

        # r2 is weaker than r1, so r2 subsumes r1 in "up"
        self.assertTrue(dr.subsumes(r1, r2))
        # But r1 does NOT subsume r2
        self.assertFalse(dr.subsumes(r2, r1))

        # "Down" direction: build a 3x1 dataset where higher F means lower class
        F_down = np.array([[1], [2], [3]])
        dec_down = np.array([3, 2, 1])  # inverse mapping
        dr_down = DRSA().fit(F_pareto_gain_type=F_down, criteria=(0,), decision_attribute=dec_down, direction = "down")

        # In "down" mode, condition "f0 <= 2" (weaker) subsumes "f0 <= 1" (stronger)
        r_down_strong = ({0: 1}, "x is 'good'", 0.33, 1.0, "possible", "down", "descA")
        r_down_weak = ({0: 2}, "x is 'good'", 0.66, 0.90, "possible", "down", "descB")

        # {0:2} is weaker than {0:1} in "down," so it subsumes
        self.assertTrue(dr_down.subsumes(r_down_strong, r_down_weak))
        # but not vice versa
        self.assertFalse(dr_down.subsumes(r_down_weak, r_down_strong))

    # -------------------------------------------------------------------------
    # Test is_robust(...) for both robust and non-robust rules
    # -------------------------------------------------------------------------
    def test_is_robust(self):
        # 3x1 "up" dataset: F = [1],[2],[3], dec = [1,2,3]
        F_1d = np.array([[1], [2], [3]])
        dec_1d = np.array([1, 2, 3])
        dr = DRSA().fit(F_pareto_gain_type=F_1d, criteria=(0,), decision_attribute=dec_1d, direction = "up")

        # profile_exact matches row [2] exactly
        r_exact = ({0: 2}, "x is 'good'", 0.33, 1.0, "certain", "up", "descX")
        self.assertTrue(dr.is_robust(r_exact))

        # profile_non does not match exactly (2.5), so not robust
        r_non = ({0: 2.5}, "x is 'good'", 0.33, 0.5, "possible", "up", "descY")
        self.assertFalse(dr.is_robust(r_non))

    # -------------------------------------------------------------------------
    # Test induce_decision_rules in "down" direction, both minimal=True and minimal=False
    # -------------------------------------------------------------------------
    def test_induce_decision_rules_down_minimal_and_not(self):
        # 4x1 dataset: F = [1,2,2,3], classes = [1,2,2,3]
        F = np.array([[1], [2], [2], [3]])
        classes = np.array([1, 2, 2, 3])
        dr = DRSA().fit(F_pareto_gain_type=F, criteria=(0,), decision_attribute=classes, direction="down")

        # minimal=False: should return at least one certain rule for "f0 <= 2"
        rules_all = dr.induce_decision_rules(criteria=(0,), threshold=2, minimal=False)
        self.assertTrue(any(r[4] == "certain" for r in rules_all))

        # minimal=True: no two rules should mutually subsume one another
        rules_minimal = dr.induce_decision_rules(criteria=(0,), threshold=2, minimal=True)
        for i, r_i in enumerate(rules_minimal):
            for j, r_j in enumerate(rules_minimal):
                if i != j:
                    # if both subsume each other, they are identical or not minimal
                    self.assertFalse(dr.subsumes(r_i, r_j) and dr.subsumes(r_j, r_i))



    # -------------------------------------------------------------------------
    # Test make_association_rule_description(...) directly
    # -------------------------------------------------------------------------
    def test_make_association_rule_description(self):
        # Antecedent and consequent as frozensets of "f_i<=Qj"
        ant = frozenset(["f_1<=Q2", "f_3<=Q4"])
        con = frozenset(["f_2<=Q3"])
        desc = DRSA.make_association_rule_description(ant, con, support=0.25, confidence=0.75)

        # We expect each "<=Q#" to be replaced by "<=#"
        # and a proper prefix "[ASSOC] IF … THEN … (support=…, confidence=…)"
        self.assertTrue(desc.startswith("[ASSOC] IF "))
        self.assertIn("f_1<=2", desc)
        self.assertIn("f_3<=4", desc)
        self.assertIn("f_2<=3", desc)
        self.assertIn("support=0.25", desc)
        self.assertIn("confidence=0.75", desc)

    # -------------------------------------------------------------------------
    # Test find_association_rules(...) when NO rules exist
    # -------------------------------------------------------------------------
    def test_find_association_rules_no_rules(self):
        # 3x2 dataset where each "bin" will be unique
        T = np.array([
            [10, 20],
            [30, 40],
            [50, 60]
        ])
        # Choose min_support=0.6 so that any rule requiring >1 item fails
        rules = DRSA.find_association_rules(
            F_pareto=T,
            criteria=(0, 1),
            min_support=0.6,
            min_confidence=0.9
        )
        self.assertIsInstance(rules, list)
        self.assertEqual(len(rules), 0)




if __name__ == '__main__':
    unittest.main()
