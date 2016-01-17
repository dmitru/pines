
import unittest

from pines.trees import BinaryDecisionTree, BinaryDecisionTreeSplit

class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        pass

    def test_empty_tree_str(self):
        tree = BinaryDecisionTree(n_features=1)
        self.assertEqual(tree.num_of_leaves(), 1)
        print(tree)

    def test_one_split(self):
        tree = BinaryDecisionTree(n_features=1)
        split = BinaryDecisionTreeSplit(feature_id=0, value=0.0)
        tree.split_node(0, split)
        self.assertEqual(tree.num_of_leaves(), 2)
        print(tree)

    def test_multiple_splits(self):
        tree = BinaryDecisionTree(n_features=1)
        split = BinaryDecisionTreeSplit(feature_id=0, value=0.0)
        for split_count in range(1, 10):
            tree.split_node(tree.leaves()[0], split)
            self.assertEqual(tree.num_of_leaves(), split_count + 1)
        print(tree)

    def test_depth(self):
        tree = BinaryDecisionTree(n_features=1)
        split = BinaryDecisionTreeSplit(feature_id=0, value=0.0)
        for split_count in range(15):
            tree.split_node(tree.leaves()[0], split)
        print(tree)

        self.assertEqual(1, tree.depth(0))

        self.assertEqual(2, tree.depth(1))
        self.assertEqual(2, tree.depth(2))

        self.assertEqual(3, tree.depth(3))
        self.assertEqual(3, tree.depth(4))
        self.assertEqual(3, tree.depth(5))
        self.assertEqual(3, tree.depth(6))

        self.assertEqual(4, tree.depth(7))
        self.assertEqual(4, tree.depth(8))
        self.assertEqual(4, tree.depth(9))
        self.assertEqual(4, tree.depth(10))
        self.assertEqual(4, tree.depth(11))
        self.assertEqual(4, tree.depth(12))
        self.assertEqual(4, tree.depth(13))
        self.assertEqual(4, tree.depth(14))

    def test_nodes_at_level(self):
        tree = BinaryDecisionTree(n_features=1)
        split = BinaryDecisionTreeSplit(feature_id=0, value=0.0)
        tree.split_node(0, split)
        tree.split_node(2, split)
        print(tree)
        self.assertEqual([0], tree.nodes_at_level(1))
        self.assertEqual([1, 2], tree.nodes_at_level(2))
        self.assertEqual([5, 6], tree.nodes_at_level(3))
        self.assertEqual([], tree.nodes_at_level(4))



if __name__ == '__main__':
    unittest.main()