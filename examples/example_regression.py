import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

from pines.estimators import DecisionTreeRegressor

if __name__ == '__main__':
    model = DecisionTreeRegressor(max_n_splits=10, max_depth=4, tree_type='oblivious')
    dataset = load_boston()

    mse = []
    for fold in range(4):
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.33)

        model.fit(X_train, y_train)
        # print(model._tree)
        y = model.predict(X_test)
        mse.append(mean_squared_error(y, y_test))

    mean_mse = np.mean(mse)
    print(mean_mse)