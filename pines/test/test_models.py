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
        pass

    def test_iris(self):
        dataset = load_iris()
        self.assertTrue(np.mean(cross_val_score(
                DecisionTreeClassifier(criterion='entropy'), dataset.data, dataset.target, cv=5)) > 0.8)

    def test_breast_cancer(self):
        dataset = load_breast_cancer()
        self.assertTrue(np.mean(cross_val_score(
                DecisionTreeClassifier(), dataset.data, dataset.target, cv=5)) > 0.8)


class TestDecisionTreeRegressor(unittest.TestCase):

    def setUp(self):
        TreeBuilderCART.logger.setLevel(logging.DEBUG)

    def test_boston(self):
        from sklearn.tree import DecisionTreeRegressor as DecisionTreeRegressorSklearn
        model = DecisionTreeRegressor()
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
        # Check that our model produces MSE no worse than 20% than the version from sklearn
        self.assertTrue(np.abs(mean_mse - mean_mse_sklearn) / mean_mse_sklearn < 0.2)


    # def test_check_estimators(self):
    #     """
    #     Tests that models adhere to scikit-learn Estimator interface.
    #     """
    #     check_estimator(DecisionTreeClassifier)


if __name__ == '__main__':
    unittest.main()