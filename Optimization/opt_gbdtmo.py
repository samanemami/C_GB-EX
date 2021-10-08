#%%
import numpy as np
import pandas as pd
from itertools import product
import sklearn.datasets as dts
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.multiclass import type_of_target
from gbdtmo_wrapper import classification, regression


X, y = dts.load_iris(return_X_y=True)


path = '/home/oem/.local/lib/python3.8/site-packages/gbdtmo/build/gbdtmo.so'
cv = 2
random_state = 1



param_grid = {"max_depth": [2, 5, 10, 20],
              "learning_rate": [0.025, 0.05, 0.1, 0.5, 1],
              "subsample": [0.75, 0.5, 1]}

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

        model = classification(max_depth=max_depth,
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



cv_result = pd.DataFrame(score)
score = np.mean(score, axis=0)
best_param = grid[np.where(score == np.amax(score))[0][0]]
best_score = np.amax(score)
#%%
score
best_param