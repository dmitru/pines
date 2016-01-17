
import unittest

import numpy as np

from pines.metrics import list_to_discrete_rv
from pines.trees import BinaryDecisionTree, BinaryDecisionTreeSplit

class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        pass

    def test_list_to_discrete_rv(self):
        cases = [
            ([0, 2, 2, 3], ([0, 2, 3], [0.25, 0.5, 0.25])),
        ]
        for case in cases:
            arg = case[0]
            result_expected = case[1]
            result_actual = list_to_discrete_rv(np.array(arg))
            self.assertTrue(np.all(result_actual[0] == result_expected[0]))
            self.assertTrue(np.all(result_actual[1] == result_expected[1]))

if __name__ == '__main__':
    unittest.main()