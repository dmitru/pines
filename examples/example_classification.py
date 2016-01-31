import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_moons, make_gaussian_quantiles

from pines.estimators import DecisionTreeClassifier
from pines.tree_builders import TreeType

if __name__ == '__main__':
    model = DecisionTreeClassifier(max_n_splits=8, max_depth=9, tree_type=TreeType.CART)
    X, y = make_gaussian_quantiles(n_samples=40000, n_features=12)
    model.fit(X, y)
    prediction = model.predict(X)
