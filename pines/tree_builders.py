# coding=utf-8

import numpy as np
import scipy.stats
import logging

from pines.metrics import list_to_discrete_rv
from pines.trees import BinaryDecisionTree, BinaryDecisionTreeSplit

from multiprocessing import Pool

from pines.utils import split_dataset, compute_split_gain_helper, compute_split_gain, SplitCriterion


class TreeSplitCART(BinaryDecisionTreeSplit):
    def __init__(self, feature_id, value, gain):
        super(TreeSplitCART, self).__init__(feature_id, value)
        self.gain = gain

class ProblemType:
    REGRESSION = 'regressor'
    CLASSIFICATION = 'classification'

class TreeType:
    CART = 'cart'
    OBLIVIOUS = 'oblivious'
    OBLIVIOUS_CART = 'oblivious-cart'

    @staticmethod
    def get_tree_builder(tree_type):
        """
        Resolves a name of a tree building method to the `TreeBuilder*` class,
        implementing this method.

        Args:
            tree_type (string), the name of a kind of the tree: 'cart', 'oblivious' and so on

        Returns:
            A `TreeBuilder` class, implementing the `tree_type` tree building method.
        """
        if tree_type == TreeType.CART:
            return TreeBuilderCART
        elif tree_type == TreeType.OBLIVIOUS:
            return TreeBuilderOblivious
        else:
            raise ValueError('Unknown tree_type: {}'.format(tree_type))


class TreeBuilderCART(object):
    """
    TreeBuilderCART implements CART algorithm (currently, without pruning phase)
    for building decision trees.

    Instances of this class can be used to generate CART trees from a dataset.

    References:
        - T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 10th printing
        - Sources of sklearn
    """

    logger = logging.getLogger("TreeBuilderCART")
    debug = False

    def __init__(self, problem, max_depth=10, min_samples_per_leaf=5,
                 max_n_splits=1,
                 leaf_prediction_rule='majority',
                 criterion='auto', n_jobs=1, **kwargs):
        """
        Initializes a new tree builder and validates the parameters.

        Args:
            problem (string):
                Can be either 'classification' or 'regression'

            max_depth (int): default is 10
                A stopping criteria, the maximum depth of the tree

            min_samples_per_leaf (int): default is 5
                A stopping criteria, stop building the subtree if
                the number of samples in it is less than `min_samples_per_leaf`

            max_n_splits (int): default is 1
                Number of splitting values to consider when choosing the best split

            leaf_prediction_rule (string): default is 'majority'
                Requires `mode` == 'regressor'. Can be either 'majority' or 'distribution'

                When 'majority', the prediction for an object will be the most frequent target among all samples
                that ended up in the same leaf during training.

                When 'distribution', the prediction for an object will be a random variable sampled from a discrete
                distribution of targets for all training samples ended up in the leaf.

            criterion (string):
                A criterion used for estimating quality of a split.
                When `mode` == 'regressor', it can only be 'mse'.
                When `mode` == 'classifier', it can be either 'gini' or 'entropy'.

            n_jobs (int): the size of process pool to use when building the tree.
                When None, use the number of cores in the system
        """
        assert problem in [ProblemType.REGRESSION, ProblemType.CLASSIFICATION]
        self.is_regression = problem == ProblemType.REGRESSION
        self.max_depth = max_depth
        self.max_n_splits = max_n_splits
        self.min_samples_per_leaf = min_samples_per_leaf

        if criterion == 'auto':
            if problem == ProblemType.CLASSIFICATION:
                criterion = SplitCriterion.GINI
            else:
                criterion = SplitCriterion.MSE
        else:
            if problem == ProblemType.CLASSIFICATION:
                assert criterion in [SplitCriterion.GINI, SplitCriterion.ENTROPY]
            else:
                assert criterion in [SplitCriterion.MSE]

        self.split_criterion = SplitCriterion.resolve_split_criterion(criterion)
        self.leaf_prediction_rule = leaf_prediction_rule
        self.n_jobs = n_jobs
        self.pool = None

    def build_tree(self, X, y):
        """

        Args:
            X: object-features matrix
            y: target vector

        Returns:
            A `BinaryDecisionTree` fitted to the dataset.

            The actual structure of the tree depends both on dataset and the parameters
            passed to the `TreeBuilderCART` constructor.

        """
        n_samples, n_features = X.shape
        tree = BinaryDecisionTree(n_features=n_features)
        if self.n_jobs > 1:
            self.pool = Pool(self.n_jobs)

        leaf_to_split = tree.root()
        self._build_tree_recursive(tree, leaf_to_split, X, y)
        self._prune_tree(tree, X, y)
        if TreeBuilderCART.debug:
            TreeBuilderCART.logger.debug(tree)
        self.pool = None
        return tree

    def _build_tree_recursive(self, tree, cur_node, X, y):
        n_samples, n_features = X.shape
        if n_samples < 1:
            return

        leaf_reached = False
        if n_samples <= self.min_samples_per_leaf:
            leaf_reached = True
        depth = tree.depth(cur_node)
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_reached = True

        best_split = None
        if not leaf_reached:
            if TreeBuilderCART.debug:
                TreeBuilderCART.logger.debug('Split at {}, n = {}'.format(cur_node, n_samples))

            best_split = self.find_best_split(X, y)
            if best_split is None:
                leaf_reached = True

        tree._leaf_n_samples[cur_node] = len(y)
        if leaf_reached:
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
        else:
            tree.split_node(cur_node, best_split)

            left_child = tree.left_child(cur_node)
            right_child = tree.right_child(cur_node)
            X_left, X_right, y_left, y_right = split_dataset(
                    X, y, best_split.feature_id, best_split.value)
            self._build_tree_recursive(tree, left_child, X_left, y_left)
            self._build_tree_recursive(tree, right_child, X_right, y_right)

    def _prune_tree(self, tree, X, y):
        # TODO: add tree pruning
        pass

    def _compute_split_values(self, X, y, feature_id):
        x = X[:, feature_id]
        split_values = []
        if self.is_regression:
            min_x, max_x = np.min(x), np.max(x)
            for _ in range(self.max_n_splits):
                for i in range(self.max_n_splits):
                    split_value = np.random.uniform(min_x, max_x)
                    split_values.append(split_value)
        else:
            sorted_xy = sorted(zip(x, y))
            for i in range(1, len(sorted_xy)):
                if sorted_xy[i-1][1] != sorted_xy[i][1]:
                    split_value = (sorted_xy[i - 1][0] + sorted_xy[i][0]) / 2.0
                    split_values.append(split_value)
            if len(split_values) > self.max_n_splits:
                np.random.shuffle(split_values)
                split_values = split_values[:self.max_n_splits]
        return split_values

    def find_best_split(self, X, y):
        n_samples, n_features = X.shape
        args = []
        for feature_id in range(n_features):
            for split_value in self._compute_split_values(X, y, feature_id):
                args.append([self.split_criterion, X, y, feature_id, split_value])
        if self.pool is not None:
            split_gains = self.pool.map(compute_split_gain_helper, args)
        else:
            split_gains = map(compute_split_gain_helper, args)

        splits = []
        for arg, gain in zip(args, split_gains):
            _, _, _, feature_id, split_value = arg
            if gain is not None and gain > 0:
                split = TreeSplitCART(feature_id, value=split_value, gain=gain)
                splits.append(split)

        best_split = None
        for split in splits:
            if best_split is None or split.gain > best_split.gain:
                    best_split = split
        return best_split


class TreeSplitOblivious(BinaryDecisionTreeSplit):
    def __init__(self, feature_id, value, gain, node_id):
        super(TreeSplitOblivious, self).__init__(feature_id, value)
        self.gain = gain
        self.node_id = node_id



class TreeBuilderOblivious(object):
    """
    TreeBuilderOblivion implements the idea called Oblivious Decision Trees.
    It builds trees that use the same splitting attribute for all nodes at the same level.
    """
    logger = logging.getLogger("TreeBuilderOblivious")
    debug = False

    def __init__(self, problem, max_depth=10, min_samples_per_leaf=4,
                 max_n_splits=1,
                 leaf_prediction_rule='majority',
                 criterion='auto', **kwargs):
        """
        Initializes a new tree builder and validates the parameters.

        Args:
            problem (string):
                Can be either 'classification' or 'regression'

            max_depth (int): default is 10
                A stopping criteria, the maximum depth of the tree

            min_samples_per_leaf (int): default is 5
                A stopping criteria, stop building the subtree if
                the number of samples in it is less than `min_samples_per_leaf`

            max_n_splits (int): default is 1
                Number of splitting values to consider when choosing the best split

            leaf_prediction_rule (string): default is 'majority'
                Requires `mode` == 'regressor'. Can be either 'majority' or 'distribution'

                When 'majority', the prediction for an object will be the most frequent target among all samples
                that ended up in the same leaf during training.

                When 'distribution', the prediction for an object will be a random variable sampled from a discrete
                distribution of targets for all training samples ended up in the leaf.

            criterion (string):
                A criterion used for estimating quality of a split.
                When `mode` == 'regressor', it can only be 'mse'.
                When `mode` == 'classifier', it can be either 'gini' or 'entropy'.
        """
        assert problem in [ProblemType.REGRESSION, ProblemType.CLASSIFICATION]
        self.is_regression = problem == ProblemType.REGRESSION
        self.max_depth = max_depth
        self.max_n_splits = max_n_splits
        self.min_samples_per_leaf = min_samples_per_leaf

        if criterion == 'auto':
            if problem == ProblemType.CLASSIFICATION:
                criterion = SplitCriterion.GINI
            else:
                criterion = SplitCriterion.MSE
        else:
            if problem == ProblemType.CLASSIFICATION:
                assert criterion in [SplitCriterion.GINI, SplitCriterion.ENTROPY]
            else:
                assert criterion in [SplitCriterion.MSE]

        self.split_criterion = SplitCriterion.resolve_split_criterion(criterion)
        self.leaf_prediction_rule = leaf_prediction_rule

    def build_tree(self, X, y):
        """
        Builds a tree fitted to data set (X, y).
        """
        n_samples, n_features = X.shape
        tree = BinaryDecisionTree(n_features=n_features)

        cur_level = 1
        self._dataset = np.copy(X), np.copy(y)
        self._data_per_node = {
            0: self._dataset
        }
        self._build_tree_recursive(tree, cur_level)
        self._prune_tree(tree, X, y)
        if TreeBuilderOblivious.debug:
            TreeBuilderOblivious.logger.debug(tree)
        return tree

    def _build_tree_recursive(self, tree, cur_level):
        nodes_on_current_level = tree.nodes_at_level(cur_level)
        leaves_need_to_split, leaves = [], []
        for node_id in nodes_on_current_level:
            X, y = self._data_per_node[node_id]
            tree._leaf_n_samples[node_id] = len(y)
            n_samples = X.shape[0]
            need_to_split = True
            if n_samples <= self.min_samples_per_leaf:
                need_to_split = False
            if self.max_depth is not None and cur_level >= self.max_depth:
                need_to_split = False
            if need_to_split:
                leaves_need_to_split.append(node_id)
            else:
                leaves.append(node_id)

        # Split the nodes
        best_layer_split = self.find_best_layer_split(leaves_need_to_split)
        at_least_one_split_is_made = False
        for node_id, node_split in zip(leaves_need_to_split, best_layer_split):
            if node_split is not None:
                at_least_one_split_is_made = True
                self.apply_node_split(tree, node_split)
            else:
                leaves.append(node_id)

        # Process nodes that won't be splitted and are going to become leaves in the final tree
        for node_id in leaves:
            _, y = self._data_per_node[node_id]
            if self.is_regression:
                tree._leaf_values[node_id] = np.mean(y)
            else:
                if self.leaf_prediction_rule == 'majority':
                    tree._leaf_values[node_id] = scipy.stats.mode(y).mode[0]
                elif self.leaf_prediction_rule == 'distribution':
                    values, probabilities = list_to_discrete_rv(y)
                    distribution = scipy.stats.rv_discrete(values=(values, probabilities))
                    func = lambda d: d.rvs()
                    tree._leaf_functions[node_id] = (func, distribution)
                else:
                    raise ValueError('Invalid value for leaf_prediction_rule: {}'.format(self.leaf_prediction_rule))

        if self.max_depth is not None and cur_level < self.max_depth and at_least_one_split_is_made:
            self._build_tree_recursive(tree, cur_level + 1)

    def _prune_tree(self, tree, X, y):
        # TODO: add tree pruning
        pass

    def _feature_splits(self, nodes, feature_id):
        splits = []
        for node in nodes:
            node_splits = []
            assert node in self._data_per_node
            X, y = self._data_per_node[node]
            x = X[:, feature_id]
            split_values = []
            if self.is_regression:
                min_x, max_x = np.min(x), np.max(x)
                for _ in range(self.max_n_splits):
                    for i in range(self.max_n_splits):
                        split_value = np.random.uniform(min_x, max_x)
                        split_values.append(split_value)
            else:
                sorted_xy = sorted(zip(x, y))
                for i in range(1, len(sorted_xy)):
                    if sorted_xy[i-1][1] != sorted_xy[i][1]:
                        split_value = (sorted_xy[i - 1][0] + sorted_xy[i][0]) / 2.0
                        split_values.append(split_value)
                if len(split_values) > self.max_n_splits:
                    np.random.shuffle(split_values)
                    split_values = split_values[:self.max_n_splits]
            for split_value in split_values:
                _, _, y_left, y_right = self.split_dataset(X, y, feature_id, split_value)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = compute_split_gain(self.split_criterion, y, y_left, y_right)
                if gain > 0:
                    split = TreeSplitOblivious(feature_id, value=split_value, gain=gain, node_id=node)
                    node_splits.append(split)
            if len(node_splits) > 0:
                best_node_split = max(node_splits, key=lambda split: split.gain)
            else:
                best_node_split = None
            splits.append(best_node_split)
        total_gain = sum(map(lambda split: split.gain, filter(lambda x: x is not None, splits)))
        return splits, total_gain

    def split_dataset(self, X, y, feature_id, value):
        mask = X[:, feature_id] <= value
        return X[mask], X[~mask], y[mask], y[~mask]

    def compute_split_gain(self, y, y_left, y_right):
        splits = [y_left, y_right]
        return self.split_criterion(y) - \
               sum([self.split_criterion(split) * float(len(split))/len(y) for split in splits])

    def find_best_layer_split(self, nodes):
        """

        Args:
            nodes: a list of nodes all belonging to the same depth level of the tree.

        Returns:
            list of `TreeSplitWithGain` objects, one for each node passed in `nodes` parameter.

        """
        n_samples, n_features = self._dataset[0].shape
        best_splits = None
        best_splits_total_gain = None
        for feature_id in range(n_features):
            splits, total_gain = self._feature_splits(nodes, feature_id)
            if best_splits is None or total_gain > best_splits_total_gain:
                best_splits = splits
                best_splits_total_gain = total_gain
        return best_splits

    def apply_node_split(self, tree, node_split):
        node = node_split.node_id
        tree.split_node(node, node_split)
        X, y = self._data_per_node[node]
        X_left, X_right, y_left, y_right = self.split_dataset(X, y, node_split.feature_id, node_split.value)
        left_child_id = tree.left_child(node)
        right_child_id = tree.right_child(node)
        self._data_per_node[left_child_id] = X_left, y_left
        self._data_per_node[right_child_id] = X_right, y_right
