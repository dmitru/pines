# coding=utf-8

import numpy as np
import scipy.stats
import logging

from pines import metrics
from pines.metrics import list_to_discrete_rv
from pines.trees import BinaryDecisionTree, BinaryDecisionTreeSplit

class TreeSplitCART(BinaryDecisionTreeSplit):
    def __init__(self, feature_id, value, impurity):
        super(TreeSplitCART, self).__init__(feature_id, value)
        self.impurity = impurity


def resolve_split_criterion(criterion):
    if criterion == 'gini':
        return metrics.gini_index
    elif criterion == 'entropy':
        return metrics.entropy
    elif criterion == 'mse':
        return metrics.mse
    else:
        raise ValueError('Unknown criterion {}'.format(criterion))


class TreeBuilderCART(object):
    """
    TreeBuilderCART implements CART algorithm for building decision trees.

    References:
        - T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 10th printing
        - Sources of sklearn
    """
    logger = logging.getLogger("TreeBuilderCART")
    debug = False

    def __init__(self, mode, max_depth=10, min_samples_per_leaf=5,
                 leaf_prediction_rule='majority',
                 criterion='auto'):
        """

        :param max_depth:
        Maximum depth of the decision tree

        :param min_samples_per_leaf:
        Minimum number of samples in a leaf node after which
        the further splitting stops
        """
        assert mode in ['classifier', 'regressor']
        self.is_regression = mode == 'regressor'
        self.max_depth = max_depth
        self.min_samples_per_leaf = min_samples_per_leaf
        if criterion == 'auto':
            if mode == 'classifier':
                criterion = 'gini'
            else:
                criterion = 'mse'
        self.split_criterion = resolve_split_criterion(criterion)
        self.leaf_prediction_rule = leaf_prediction_rule

    def build_tree(self, X, y):
        """
        Builds a tree fitted to data set (X, y).
        """
        n_samples, n_features = X.shape
        tree = BinaryDecisionTree(n_features=n_features)

        leaf_to_split = tree.root()
        self._build_tree_recursive(tree, leaf_to_split, X, y)
        self._prune_tree(tree, X, y)
        return tree

    def _build_tree_recursive(self, tree, cur_node, X, y):
        n_samples, n_features = X.shape
        if n_samples < 1:
            return

        tree._leaf_n_samples[cur_node] = len(y)
        if self.is_regression:
            tree._leaf_values[cur_node] = np.mean(y)
        else:
            if self.leaf_prediction_rule == 'majority':
                tree._leaf_values[cur_node] = scipy.stats.mode(y).mode[0]
            elif self.leaf_prediction_rule == 'distribution':
                values, probabilities = list_to_discrete_rv(y)
                distribution = scipy.stats.rv_discrete(values=(values, probabilities))
                func = lambda d: d.rvs()
                tree._leaf_functions[cur_node] = (func, distribution)
            else:
                raise ValueError('Invalid value for leaf_prediction_rule: {}'.format(self.leaf_prediction_rule))

        leaf_reached = False
        if n_samples <= self.min_samples_per_leaf:
            leaf_reached = True
        depth = tree.depth(cur_node)
        if depth >= self.max_depth:
            leaf_reached = True

        if leaf_reached:
            return

        if TreeBuilderCART.debug:
            TreeBuilderCART.logger.debug('Split at {}, n = {}'.format(cur_node, n_samples))
        best_split = self.find_best_split(X, y)
        if best_split is None:
            return

        tree.split_node(cur_node, best_split)

        left_child = tree.left_child(cur_node)
        right_child = tree.right_child(cur_node)
        X_left, X_right, y_left, y_right = self.split_dataset(
                X, y, best_split.feature_id, best_split.value)
        self._build_tree_recursive(tree, left_child, X_left, y_left)
        self._build_tree_recursive(tree, right_child, X_right, y_right)

    def _prune_tree(self, tree, X, y):
        # TODO: add tree pruning
        pass

    def _feature_splits(self, X, y, feature_id):
        splits = []
        x = X[:, feature_id]
        sorted_xy = sorted(zip(x, y))
        if self.is_regression:
            split_value = np.random.uniform(sorted_xy[0][0], sorted_xy[-1][0])
            _, _, y_left, y_right = self.split_dataset(X, y, feature_id, split_value)
            impurity = self.compute_split_impurity(y, y_left, y_right)
            split = TreeSplitCART(feature_id, value=split_value, impurity=impurity)
            splits.append(split)
        else:
            for i in range(1, len(sorted_xy)):
                if sorted_xy[i-1][1] != sorted_xy[i][1]:
                    split_value = (sorted_xy[i - 1][0] + sorted_xy[i][0]) / 2.0
                    _, _, y_left, y_right = self.split_dataset(X, y, feature_id, split_value)
                    impurity = self.compute_split_impurity(y, y_left, y_right)
                    split = TreeSplitCART(feature_id, value=split_value, impurity=impurity)
                    splits.append(split)
        return splits

    def find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_split = None
        for feature_id in range(n_features):
            splits = self._feature_splits(X, y, feature_id)
            for split in splits:
                if best_split is None or split.impurity < best_split.impurity:
                    best_split = split
        return best_split

    def split_dataset(self, X, y, feature_id, value):
        mask = X[:, feature_id] <= value
        return X[mask], X[~mask], y[mask], y[~mask]

    def compute_split_impurity(self, y, y_left, y_right):
        splits = [y_left, y_right]
        return sum([self.split_criterion(split) * float(len(split))/len(y) for split in splits])
