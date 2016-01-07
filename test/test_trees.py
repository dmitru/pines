
import unittest
from trees import DecisionTree, DecisionTreeSplit

class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        pass

    def test_empty_tree_str(self):
        tree = DecisionTree(features_count=1)
        self.assertEqual(tree.num_of_leaves(), 1)
        print(tree)

    def test_one_split(self):
        tree = DecisionTree(features_count=1)
        split = DecisionTreeSplit(0, 0.0)
        tree.split_node(0, split)
        self.assertEqual(tree.num_of_leaves(), 2)
        print(tree)

    def test_multiple_splits(self):
        tree = DecisionTree(features_count=1)
        split = DecisionTreeSplit(0, 0.0)
        for split_count in range(1, 10):
            tree.split_node(tree.leaves()[0], split)
            print(tree)
            self.assertEqual(tree.num_of_leaves(), split_count + 1)


if __name__ == '__main__':
    unittest.main()