from unittest import TestCase
import numpy as np

from IMO_DRSA.drsa import DRSA


class TestDRSA(TestCase):

    def test_positive_cone(self):
        self.fail()



    def test_negative_cone(self):
        self.fail()



    def test_lower_approx_up(self):
        self.fail()



    def test_upper_approx_up(self):
        self.fail()



    def test_lower_approx_down(self):
        self.fail()



    def test_upper_approx_down(self):
        self.fail()



    def test_quality(self):
        self.fail()



    def test_find_reducts(self):
        self.fail()



    def test_induce_rules(self):
        self.fail()


    def test_generate_constraints(self):
        self.fail()


    def test_drsa(self):
        # Two criteria f1, f2; five objects; and a 3-class decision d in {1,2,3}
        X = np.array([
            [5, 7],  # obj0
            [3, 4],  # obj1
            [6, 2],  # obj2
            [4, 6],  # obj3
            [2, 5],  # obj4
        ])
        d = np.array([1, 1, 3, 2, 2])  # decision classes for obj0, ..., obj4

        drsa = DRSA(X, d, criteria=[0, 1])

        #pos_cone = drsa.positive_cone((0, 1))
        #print(pos_cone)

        low_up = drsa.lower_approx_up((0, 1), t=2)  # boolean mask length 5
        up_up = drsa.upper_approx_up((0, 1), t=2)

        self.assertListEqual(low_up.tolist(), [False, False, True, False, False])
        self.assertListEqual(up_up.tolist(), [True, False, True, True, True])

        gamma = drsa.quality((0, 1))
        print(f"Quality of full set P={{f1,f2}}: {gamma:.2f}")

        reducts = drsa.find_reducts()
        print("Minimal reducts:", reducts)

        rules = drsa.induce_rules((0, 1), union_type='up', t=2)

        drsa.explain_rules(rules, verbose=True)
