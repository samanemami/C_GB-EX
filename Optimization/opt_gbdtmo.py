from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.multiclass import type_of_target
from itertools import product
import numpy as np


def grid(estimator: 'model', X: np.array,
         y: np.array, cv: 'int32',
         param_grid: dict,
         random_state: 'int32') -> "Optimize the model":

    score = np.zeros((cv, y.shape[1]))

    if type_of_target(y) == 'multiclass' or 'binary':
        kfold = StratifiedKFold(n_splits=cv, shuffle=True,
                                random_state=random_state)

    else:
        kfold = kfold(n_splits=cv, shuffle=True,
                      random_state=random_state)

    for cv_i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid = [dict(zip(param_grid, v))
                for v in product(*param_grid.values())]
        for param in grid:
            model = estimator(param)
            model.fit(x_train, y_train)
            model.score(x_test, y_test)
