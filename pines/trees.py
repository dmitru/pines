# coding=utf-8

import numpy as np
from copy import deepcopy

class BinaryDecisionTreeSplit(object):
    def __init__(self, feature_id, value):
        self.feature_id = feature_id
        self.value = value

class BinaryDecisionTree(object):
    """
    Implements a binary decision tree with array-based representation.

    This class itself doesn't contain logic for selection of best splits, etc;
    instead, it receives DecisionTreeSplit that describe splits and updates the tree accordingly.
    """

    def __init__(self, n_features):
        """
        :param n_features: number of features in dataset. Features have 0-based indices
        """
        self._capacity = 0

        self._n_features = n_features
        self._is_leaf = np.zeros(0, dtype='bool')
        self._is_node = np.zeros(0, dtype='bool')
        self._leaf_values = np.zeros(0)
        self._leaf_functions = []
        self._leaf_n_samples = np.zeros(0)
        self._splits = []

        self._capacity = 0
        self._reallocate_if_needed(required_capacity=1)
        self._init_root()

    def _reallocate_if_needed(self, required_capacity):
        if self._capacity <= required_capacity:
            self._is_leaf.resize(required_capacity)
            self._is_node.resize(required_capacity)
            self._leaf_values.resize(required_capacity)
            self._leaf_functions = self._grow_list(self._leaf_functions, required_capacity)
            self._leaf_n_samples.resize(required_capacity)
            self._splits = self._grow_list(self._splits, required_capacity)
            self._capacity = required_capacity

    def _init_root(self):
        self._is_leaf[0] = True
        self._is_node[0] = True
        self._latest_used_node_id = 0

    def num_of_leaves(self):
        return np.sum(self._is_leaf[:self._latest_used_node_id + 1])

    def num_of_nodes(self):
        return self._latest_used_node_id

    def is_leaf(self, node_id):
        assert node_id >= 0 and node_id <= self._latest_used_node_id
        return self._is_leaf[node_id]

    def leaf_mask(self):
        return self._is_leaf[:self._latest_used_node_id + 1]

    def __str__(self):
        def helper(cur_node_id, padding='', is_last_leaf_on_level=True):
            if cur_node_id > self._latest_used_node_id or not self._is_node[cur_node_id]:
                return ''

            if self._is_leaf[cur_node_id]:
                node_str = '{}: {:.2f} (n={})'.format(
                            cur_node_id, self._leaf_values[cur_node_id],
                            int(self._leaf_n_samples[cur_node_id]))
            else:
                node_str = '{}: [x[{}] < {:.2f}]? (n={})'.format(
                        cur_node_id,
                        self._splits[cur_node_id].feature_id,
                        self._splits[cur_node_id].value,
                        int(self._leaf_n_samples[cur_node_id])
                )
            result = padding + ("└── " if is_last_leaf_on_level else  "├── ") + node_str + '\n'
            if is_last_leaf_on_level:
                new_padding = padding + '    '
            else:
                new_padding = padding + '|   '

            result += helper(self.left_child(cur_node_id), new_padding, False)
            result += helper(self.right_child(cur_node_id), new_padding, True)
            return result

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
        assert split.feature_id >= 0 and split.feature_id < self._n_features
        assert self.is_leaf(node_id) == True

        left_child_id = self.left_child(node_id)
        right_child_id = self.right_child(node_id)

        if right_child_id >= self._capacity:
            self._reallocate_if_needed(2 * self._capacity + 1)

        self._splits[node_id] = deepcopy(split)
        self._is_leaf[node_id] = False
        self._is_node[left_child_id] = True
        self._is_node[right_child_id] = True
        self._is_leaf[left_child_id] = True
        self._is_leaf[right_child_id] = True
        self._latest_used_node_id = max(self._latest_used_node_id, right_child_id)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        def predict_one(x):
            current_node = self.root()
            while not self.is_leaf(current_node):
                current_split = self._splits[current_node]
                if x[current_split.feature_id] < current_split.value:
                    current_node = self.left_child(current_node)
                else:
                    current_node = self.right_child(current_node)
            if self._leaf_functions[current_node] is not None:
                func, args = self._leaf_functions[current_node]
                return func(args)
            return self._leaf_values[current_node]

        sample_size, features_count = X.shape
        assert features_count == self._n_features
        result = np.zeros(sample_size)
        for i in range(sample_size):
            x = X[i]
            result[i] = predict_one(x)
        return result

    def apply(self, X):
        """

        Args:
            X: numpy 2d array
                Instance-features matrix

        Returns: numpy int array
            Array of leaf indices, corresponding to classified instances
        """
        def apply_one(x):
            current_node = self.root()
            while not self.is_leaf(current_node):
                current_split = self._splits[current_node]
                if x[current_split.feature_id] < current_split.value:
                    current_node = self.left_child(current_node)
                else:
                    current_node = self.right_child(current_node)
            return current_node

        sample_size, features_count = X.shape
        assert features_count == self._n_features
        result = np.zeros(sample_size)
        for i in range(sample_size):
            x = X[i]
            result[i] = apply_one(x)
        return result

    def root(self):
        """
        :return: Id of the root node
        """
        return 0

    def depth(self, node_id):
        assert node_id >= 0 and node_id <= self._latest_used_node_id
        return np.floor(np.log2(node_id + 1)) + 1

    def nodes_at_level(self, level, kind='all'):
        """
        Args:
            level: Depth level in the tree, starting from 1 for the root node.
            kind: 'all', 'internal_nodes' or 'leaves'

        Returns:
            List of node ids at the specified level.
        """
        assert kind in ['all', 'internal_nodes', 'leaves']
        result = []
        for node_id in range(2 ** (level - 1) - 1, min(2 ** level - 1, self._latest_used_node_id + 1)):
            if kind == 'all':
                result.append(node_id)
            elif kind == 'internal_nodes':
                if self._is_node[node_id]:
                    result.append(node_id)
            else:
                if self._is_leaf[node_id]:
                    result.append(node_id)
        return result

    def _grow_list(self, list, required_capacity, fill_value=None):
        """
        Returns a list that is at least as long as required_capacity, filling the missing elements with
        fill_value if needed.
        If the length of the list is already greater than required_capacity, returns unmodified list.
        :param list:
        :param required_capacity:
        :param fill_value:
        :return:
        """
        if len(list) >= required_capacity:
            return list
        return list + [fill_value for _ in range(required_capacity - len(list))]
