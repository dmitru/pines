# coding=utf-8

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import NotFittedError

from pines.tree_builders import TreeType, ProblemType


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, tree_type=TreeType.CART, **kwargs):
        """
        Builds a decision tree for a classification problem.
        Args:
            tree_type (string): One of 'cart' or 'oblivious', default is 'cart'
            **kwargs: arguments to pass to a `TreeBuilder` instance

        Returns: self
        """
        super(DecisionTreeClassifier, self).__init__()
        self.tree_ = None
        self.tree_type = tree_type
        self._tree_builder_kwargs = kwargs
        self._tree_builder_class = TreeType.get_tree_builder(tree_type)

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, dtype=np.float64)

        data_size, n_features = X.shape
        self._n_features = n_features

        self._tree_builder = self._tree_builder_class(
            problem=ProblemType.CLASSIFICATION,
            **self._tree_builder_kwargs
        )
        self.tree_ = self._tree_builder.build_tree(X, y)
        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = self._validate_X_predict(X, check_input=True)
        return self.tree_.predict(X)

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.tree_ is None:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        if check_input:
            X = check_array(X, dtype='f')

        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self._n_features, n_features))

        return X


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tree_type=TreeType.CART, **kwargs):
        """
        Builds a decision tree for a classification problem.
        Args:
            tree_type (string): One of 'cart' or 'oblivious', default is 'cart'
            **kwargs: arguments to pass to a `TreeBuilder` instance

        Returns: self
        """
        super(DecisionTreeRegressor, self).__init__()
        self._tree = None
        self.tree_type = tree_type
        self._tree_builder_kwargs = kwargs
        self._tree_builder_class = TreeType.get_tree_builder(tree_type)

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, dtype=np.float64)
        data_size, n_features = X.shape
        self._n_features = n_features

        self._tree_builder = self._tree_builder_class(
            problem=ProblemType.REGRESSION,
            **self._tree_builder_kwargs
        )
        self._tree = self._tree_builder.build_tree(X, y)
        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = self._validate_X_predict(X, check_input=True)
        return self._tree.predict(X)

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self._tree is None:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        if check_input:
            X = check_array(X, dtype='f')

        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self._n_features, n_features))

        return X
