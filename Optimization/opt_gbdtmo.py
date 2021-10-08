import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.multiclass import type_of_target
from gbdtmo_wrapper import classification, regression


def gridsearch(X, y, cv, random_state, path, param_grid):

    grid = [dict(zip(param_grid, v))
            for v in product(*param_grid.values())]
    score = np.zeros((cv, len(grid)))

    if type_of_target(y) == 'multiclass' or 'binary':
        kfold = StratifiedKFold(n_splits=cv, shuffle=True,
                                random_state=random_state)

    else:
        kfold = kfold(n_splits=cv, shuffle=True,
                      random_state=random_state)

    for cv_i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for cv_grid in range(len(grid)):
            learning_rate = (list(grid[cv_grid].values())[1])
            max_depth = (list(grid[cv_grid].values())[0])
            subsample = (list(grid[cv_grid].values())[2])

            if type_of_target(y) == 'multiclass' or 'binary':

                model = classification(max_depth=max_depth,
                                       learning_rate=learning_rate,
                                       random_state=random_state,
                                       num_boosters=100,
                                       lib=path,
                                       subsample=subsample,
                                       verbose=False,
                                       num_eval=0
                                       )

            else:
                model = regression(max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   random_state=random_state,
                                   num_boosters=100,
                                   lib=path,
                                   subsample=subsample,
                                   verbose=False,
                                   num_eval=0
                                   )

            model.fit(x_train, y_train)
            score[cv_i, cv_grid] = model.score(x_test, y_test)
            print('*', end='')

    cv_result = pd.DataFrame(score)
    score = np.mean(score, axis=0)
    score_std = np.std(cv_result, axis=0)
    best_score = np.amax(score) if type_of_target(
        y) == 'multiclass' or 'binary' else np.amin(score)
    score_std = np.amax(score_std) if type_of_target(
        y) == 'multiclass' or 'binary' else np.amin(score_std)
    best_params = []
    if np.where(score == best_score)[0].shape[0] > 1:
        for i, j in enumerate(np.where(score == best_score)[0]):
            best_params.append(grid[np.where(score == best_score)[0][i]])
    else:
        best_params = grid[np.where(score == best_score)[0][0]]

    result = {}
    result['best_score'] = best_score
    result['score_std'] = score_std
    pd.Series(result).to_csv('result.csv')