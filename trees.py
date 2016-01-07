# coding=utf-8

import numpy as np
from copy import deepcopy

class DecisionTreeSplit(object):
    """
    DecisionTreeSplit describes a split to perform on a DecisionTree.

    Typical use of this class is for a TreeGrower to decide which split to make,
    create a DecisionTreeSplit and pass it to the DecisionTree, so that the latter
    can update its internal structures.
    """
    def __init__(self, feature_id, treshold):
        """

        :param feature_id: int, id of the feature on which the split is made
        :param treshold: number, threshold for the split
        :return:
        """
        self.feature_id = feature_id
        self.treshold = treshold

class DecisionTree(object):
    """
    Implements a binary decision tree with array-based representation.

    This class itself doesn't contain logic for selection of best splits, etc;
    instead, it receives DecisionTreeSplit that describe splits and updates the tree accordingly.
    """

    def __init__(self, features_count):
        """
        :param features_count: number of features in dataset. Features have 0-based indices
        """
        self._capacity = 0

        self._features_count = features_count
        self._is_leaf = np.zeros(0, dtype='bool')
        self._splits = []

        self.values = np.zeros(0)

        initial_capacity = 8
        self._reallocate_if_needed(initial_capacity)
        self._init_root()

    def _reallocate_if_needed(self, required_capacity):
        if self._capacity <= required_capacity:
            self.values.resize(required_capacity)
            self._is_leaf.resize(required_capacity)
            self._splits = self._splits + [None for _ in range(required_capacity - len(self._splits))]
            self._capacity = required_capacity

    def _init_root(self):
        self._is_leaf[0] = True
        self._latest_used_node_id = 0

    def num_of_leaves(self):
        return np.sum(self._is_leaf[:self._latest_used_node_id + 1])

    def is_leaf(self, node_id):
        assert node_id >= 0 and node_id <= self._latest_used_node_id
        return self._is_leaf[node_id]

    def __str__(self):
        def helper(cur_node_id):
            if cur_node_id > self._latest_used_node_id:
                return 'X'
            if self.is_leaf(cur_node_id):
                return str(cur_node_id)
            else:
                return '({}, {}, {})'.format(
                        cur_node_id, helper(self.left_child(cur_node_id)), helper(self.right_child(cur_node_id)))
        return helper(0)

    def left_child(self, node_id):
        return (node_id + 1) * 2 - 1

    def right_child(self, node_id):
        return (node_id + 1) * 2

    def leaves(self):
        return np.where(self._is_leaf == True)[0]

    def split_node(self, node_id, split):
        """
        Modifies the tree, applying the specified node split.
        The node that is being splitted must be a leaf.
        After the split, the number of leaves increases by one.

        :param split: DecisionTreeSplit, describes the split to perform
        """
        assert node_id >= 0 and node_id <= self._latest_used_node_id
        assert split.feature_id >= 0 and split.feature_id < self._features_count
        assert self.is_leaf(node_id) == True

        left_child_id = self.left_child(node_id)
        right_child_id = self.right_child(node_id)

        if right_child_id >= self._capacity:
            self._reallocate_if_needed(2 * self._capacity + 1)

        self._splits[node_id] = deepcopy(split)
        self._is_leaf[node_id] = False
        self._is_leaf[left_child_id] = True
        self._is_leaf[right_child_id] = True
        self._latest_used_node_id = max(self._latest_used_node_id, right_child_id)


