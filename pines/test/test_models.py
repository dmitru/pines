import logging
import unittest

import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.utils.estimator_checks import check_estimator

from sklearn.datasets import load_iris, load_breast_cancer, load_boston
from sklearn.cross_validation import cross_val_score, train_test_split

from pines.estimators import DecisionTreeClassifier, DecisionTreeRegressor
from pines.tree_builders import TreeBuilderCART


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.tree_type = 'cart'

    def test_iris(self):
        dataset = load_iris()
        score = np.mean(cross_val_score(
                DecisionTreeClassifier(tree_type=self.tree_type), dataset.data, dataset.target, cv=10))
        print('iris: tree_type: {}, score = {}'.format(self.tree_type, score))
        self.assertTrue(score > 0.8)


    def test_breast_cancer(self):
        dataset = load_breast_cancer()
        score = np.mean(cross_val_score(
                DecisionTreeClassifier(tree_type=self.tree_type), dataset.data, dataset.target, cv=10))
        print('breast_cancer: tree_type: {}, score = {}'.format(self.tree_type, score))
        self.assertTrue(score > 0.8)


class TestObliviousDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.tree_type = 'oblivious'

    def test_iris(self):
        dataset = load_iris()
        score = np.mean(cross_val_score(
                DecisionTreeClassifier(tree_type=self.tree_type), dataset.data, dataset.target, cv=10))
        self.assertTrue(score > 0.8)
        print('iris: tree_type: {}, score = {}'.format(self.tree_type, score))

    def test_breast_cancer(self):
        dataset = load_breast_cancer()
        score = np.mean(cross_val_score(
                DecisionTreeClassifier(tree_type=self.tree_type), dataset.data, dataset.target, cv=10))
        self.assertTrue(score > 0.8)
        print('breast_cancer: tree_type: {}, score = {}'.format(self.tree_type, score))


class TestDecisionTreeRegressor(unittest.TestCase):

    def setUp(self):
        TreeBuilderCART.logger.setLevel(logging.DEBUG)

    def test_boston(self):
        from sklearn.tree import DecisionTreeRegressor as DecisionTreeRegressorSklearn
        model = DecisionTreeRegressor(max_n_splits=3)
        model_sklearn = DecisionTreeRegressorSklearn()

        dataset = load_boston()
        mse = []
        mse_sklearn = []

        for fold in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=0.33)

            model.fit(X_train, y_train)
            y = model.predict(X_test)
            mse.append(mean_squared_error(y, y_test))

            model_sklearn.fit(X_train, y_train)
            y = model_sklearn.predict(X_test)
            mse_sklearn.append(mean_squared_error(y, y_test))

        mean_mse = np.mean(mse)
        mean_mse_sklearn = np.mean(mse_sklearn)
        print(mean_mse, mean_mse_sklearn)
        # Check that our model differs in MSE no worse than 20%
        self.assertTrue(np.abs(mean_mse - mean_mse_sklearn) / mean_mse_sklearn < 0.2)


class TestObliviousDecisionTreeRegressor(unittest.TestCase):

    def setUp(self):
        pass

    def test_boston(self):
        from sklearn.tree import DecisionTreeRegressor as DecisionTreeRegressorSklearn
        model = DecisionTreeRegressor(tree_type='oblivious', max_n_splits=3)
        model_sklearn = DecisionTreeRegressorSklearn()

        dataset = load_boston()
        mse = []
        mse_sklearn = []

        for fold in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=0.33)

            model.fit(X_train, y_train)
            y = model.predict(X_test)
            mse.append(mean_squared_error(y, y_test))

            model_sklearn.fit(X_train, y_train)
            y = model_sklearn.predict(X_test)
            mse_sklearn.append(mean_squared_error(y, y_test))

        mean_mse = np.mean(mse)
        mean_mse_sklearn = np.mean(mse_sklearn)
        print(mean_mse, mean_mse_sklearn)
        # Check that our model differs in MSE no worse than 50%
        self.assertTrue(np.abs(mean_mse - mean_mse_sklearn) / mean_mse_sklearn < 0.5)


    # def test_check_estimators(self):
    #     """
    #     Tests that models adhere to scikit-learn Estimator interface.
    #     """
    #     check_estimator(DecisionTreeClassifier)


if __name__ == '__main__':
    unittest.main()