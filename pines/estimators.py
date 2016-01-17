# coding=utf-8

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import NotFittedError

from pines.tree_builders import TreeBuilderCART


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.tree_ = None
        # TODO: validate parameters
        self.tree_builder_kwargs = kwargs

    def fit(self, X, y):
        """

        :param X:
        :param y:
        """
        X, y = check_X_y(X, y, dtype=np.float64)

        data_size, n_features = X.shape
        self._n_features = n_features

        self._tree_builder = TreeBuilderCART(mode='classifier', **self.tree_builder_kwargs)
        self.tree_ = self._tree_builder.build_tree(X, y)
        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
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
    def __init__(self, **kwargs):
        self._tree = None
        self.tree_builder_kwargs = kwargs

    def fit(self, X, y):
        """

        :param X:
        :param y:
        """
        X, y = check_X_y(X, y, dtype=np.float64)

        data_size, n_features = X.shape
        self._n_features = n_features

        self._tree_builder = TreeBuilderCART(mode='regressor', **self.tree_builder_kwargs)
        self._tree = self._tree_builder.build_tree(X, y)
        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
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
