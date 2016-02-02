import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.datasets import load_boston, load_breast_cancer, load_iris, make_moons, make_gaussian_quantiles
from sklearn.metrics import mean_squared_error

from mlxtend.evaluate import plot_decision_regions
import matplotlib.pyplot as plt

from pines.estimators import DecisionTreeRegressor, DecisionTreeClassifier
from pines.tree_builders import TreeType

if __name__ == '__main__':
    model = DecisionTreeClassifier(max_n_splits=3, max_depth=10, tree_type=TreeType.OBLIVIOUS)
    X, y = make_gaussian_quantiles(n_samples=10000, n_classes=4)

    model.fit(X, y)
    print(model.tree_)
    plot_decision_regions(X, y, clf=model, res=0.02, legend=2)
    plt.savefig('decision_boundary.png')