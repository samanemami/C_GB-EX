import sys
import numpy as np
import pandas as pd
from itertools import product
import sklearn.datasets as dts
from sklearn.metrics import accuracy_score
from gbdtmo import GBDTMulti, load_lib, plotting
from sklearn.model_selection import StratifiedKFold, KFold

X, y = dts.make_regression(n_targets=4)

random_state = 1
clf = False
cv = 5

path = '/home/user/.local/lib/python~/site-packages/gbdtmo/build/gbdtmo.so'
LIB = load_lib(path)

param_grid = {"max_depth": [2, 5, 10, 20],
              "learning_rate": [0.025, 0.05, 0.1, 0.5, 1],
              "subsample": [0.75, 0.5, 1]}


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


grid = [dict(zip(param_grid, v))
        for v in product(*param_grid.values())]


if clf:
    kfold = StratifiedKFold(n_splits=cv, shuffle=True,
                            random_state=random_state)
    score = np.zeros((cv, len(grid)))
    n_class = len(np.unique(y))
    X, y = np.ascontiguousarray(X, dtype=np.float64), y.astype('int32')
    loss = b"ce"
else:
    kfold = KFold(n_splits=cv, shuffle=True,
                  random_state=random_state)
    score = np.zeros((cv, len(grid), y.shape[1]))
    n_class = y.shape[1]
    X, y = np.ascontiguousarray(X, dtype=np.float64), y.astype('float64')
    loss = b"mse"

for cv_i, (train_index, test_index) in enumerate(kfold.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for cv_grid in range(len(grid)):
        max_depth = (list(grid[cv_grid].values())[0])
        learning_rate = (list(grid[cv_grid].values())[1])
        subsample = (list(grid[cv_grid].values())[2])

        params = {"max_depth": max_depth, "lr": learning_rate, 'loss': loss,
                  'verbose': False, 'subsample': subsample}
        booster = GBDTMulti(LIB, out_dim=n_class, params=params)
        booster.set_data((x_train, y_train))
        booster.train(100)

        if clf:
          acc = accuracy_score(y_test, np.argmax(
              booster.predict(x_test), axis=1))

          score[cv_i, cv_grid] = acc
        else:
          rmse = np.sqrt(np.average(
              (y_test - booster.predict(x_test)) ** 2, axis=0))
          score[cv_i, cv_grid, :] = rmse

        ProgressBar(cv_grid/abs(len(grid)-1), barLen=len(grid))


score_mean = np.mean(score, axis=0)
score_std = np.std(score, axis=0)
best_score = np.amax(score_mean) if clf else np.amin(score_mean, axis=0)
score_std = np.amax(
    score_std) if clf else np.amin(score_std, axis=0)
best_params = []
for i, j in enumerate(np.where(score_mean == best_score)[0]):
    best_params.append(grid[np.where(score_mean == best_score)[0][i]])

result = {}
result['mean_test_score'] = best_score
result['std_test_score'] = score_std
pd.Series(result).to_csv('result.csv')
index = []
for i in range(len(best_params)):
    index.append('grid -' + str(i))
pd.DataFrame(best_params, index=index).to_csv('best_parasm.csv')
