import sys
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold, KFold
from gbdtmo_wrapper import classification, regression


def ProgressBar(percent, barLen=20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()


def gridsearch(X, y, cv, random_state, path, param_grid, verbose, clf):

    grid = [dict(zip(param_grid, v))
            for v in product(*param_grid.values())]
    score = np.zeros((cv, len(grid))) if clf else np.zeros(
        (cv, len(grid), y.shape[1]))

    if clf:
        kfold = StratifiedKFold(n_splits=cv, shuffle=True,
                                random_state=random_state)

    else:
        kfold = KFold(n_splits=cv, shuffle=True,
                      random_state=random_state)

    for cv_i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for cv_grid in range(len(grid)):
            max_depth = (list(grid[cv_grid].values())[0])
            learning_rate = (list(grid[cv_grid].values())[1])
            subsample = (list(grid[cv_grid].values())[2])

            if clf:

                model = classification(max_depth=max_depth,
                                       learning_rate=learning_rate,
                                       random_state=random_state,
                                       num_boosters=100,
                                       lib=path,
                                       subsample=subsample)

                model.fit(x_train, y_train)
                score[cv_i, cv_grid] = model.score(x_test, y_test)

            else:
                model = regression(max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   random_state=random_state,
                                   num_boosters=100,
                                   lib=path,
                                   subsample=subsample)

                model.fit(x_train, y_train)
                score[cv_i, cv_grid, :] = model.score(x_test, y_test)
            if verbose:
                ProgressBar(cv_grid/abs(len(grid)-1), barLen=len(grid))

    score_mean = np.mean(score, axis=0)
    score_std = np.std(score, axis=0)
    best_score = np.amax(score_mean) if clf else np.amin(score_mean, axis=0)
    score_std = np.amax(
        score_std) if clf else np.amin(score_std)
    best_params = []
    for i, j in enumerate(np.where(score_mean == best_score)[0]):
        best_params.append(grid[np.where(score_mean == best_score)[0][i]])

    result = {}
    result['mean_test_score'] = best_score
    result['std_test_score'] = score_std
    pd.Series(result).to_csv('result.csv')
    index = []
    for i in range(len(best_params)):
        index.append('split -' + str(i))
    pd.DataFrame(best_params, index=index).to_csv('best_parasm.csv')
