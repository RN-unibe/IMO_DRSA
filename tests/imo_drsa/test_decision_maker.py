import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.imo_drsa.decision_maker import BaseDM, InteractiveDM, AutomatedDM
from src.imo_drsa.drsa import DRSA

# ---------------------------------------------------------------------------------------------------------- #
# Interactive DM tests
# ---------------------------------------------------------------------------------------------------------- #
class TestInteractiveDM(TestCase):

    @patch('builtins.input', return_value='\n')
    @patch('builtins.print')
    def test_classify(self, mock_input, mock_print):
        dm = InteractiveDM()
        T = np.array([1, 1, 1, 1])
        X = np.array([1, 1, 1, 1])

        expected_out = np.ones(len(T), dtype=int)

        out = dm.classify(T, X, None)
        self.assertListEqual(out.tolist(), expected_out.tolist())



# ---------------------------------------------------------------------------------------------------------- #
# Automated DM tests
# ---------------------------------------------------------------------------------------------------------- #
class TestAutomatedDM(TestCase):

    @patch('builtins.print')
    def test_select(self, mock_print):
        dm = AutomatedDM()

        rules = [({0: 1.0, 1: 2.0}, 'd>=2', 0.5, 0.9, 'certain', 'up',
                  "[CERTAIN] IF f_1 >= 1.0 AND f_2 >= 2.0 THEN d >= 2 (support=0.50, confidence=0.90)"),
                 ({0: 0.5, 1: 1.5}, 'd>=2', 0.3, 0.7, 'possible', 'up',
                  "[POSSIBLE] IF f_0 >= 0.5 AND f_1 >= 1.5 THEN d >= 2 (support=0.30, confidence=0.70)")]

        chosen = dm.select(rules)

        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0], rules[0])



if __name__ == '__main__':
    unittest.main()