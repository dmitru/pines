"""
This example shows how to use hybrid CART/Oblivious Decision Trees for classification problem.

The type of trees constructed is governed by `tree_type` parameter of `DecisionTreeClassifier`.

In this case it is `TreeType.OBLIVIOUS_CART`, which means that the algorithm will start building an Oblivious Decision
tree as long as it can improve the quality of the model, and after that point it will switch to CART tree building
algorithm to finish the construction.

The specific switch criterion is specified by `switch_criterion` option.
"""

import logging
import matplotlib.pyplot as plt

import numpy as np
from mlxtend.evaluate import plot_decision_regions
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_moons, make_gaussian_quantiles

from pines.estimators import DecisionTreeClassifier
from pines.tree_builders import TreeType, ObliviousCartSwitchCriterionType, TreeBuilderObliviousCart

from pylab import rcParams
rcParams['figure.figsize'] = 20, 20

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    TreeBuilderObliviousCart.debug = True
    model = DecisionTreeClassifier(max_n_splits=10, max_depth=20, tree_type=TreeType.OBLIVIOUS_CART,
                                   min_samples_per_leaf=30,
                                   switch_criterion=ObliviousCartSwitchCriterionType.OBLIVIOUS_WHILE_CAN)
    X, y = make_gaussian_quantiles(n_samples=2000, n_classes=4)
    model.fit(X, y)
    prediction = model.predict(X)
    plot_decision_regions(X, y, clf=model, res=0.02, legend=2)
    plt.savefig('decision_boundary.png')
