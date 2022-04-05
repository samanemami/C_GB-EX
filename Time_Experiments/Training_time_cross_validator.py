import numpy as np
import pandas as pd
from time import process_time
import sklearn.datasets as dt
from IPython.display import clear_output
from sklearn.model_selection import ShuffleSplit

from TFBT import BoostedTreesClassifier
from Scikit_CGB import C_GradientBoostingClassifier
from gbdtmo_wrapper import regression, classification
from mart import GradientBoostingClassifier

X, y = dt.load_iris(return_X_y=True)

max_depth = 5
random_state = 1
n = 10
path = 'path to lib.so'
temp_path = 'path to remove tf.logs'

t_cgb = np.zeros((n,))
t_mart = np.zeros((n,))
t_tfbt = np.zeros((n,))
t_gbdtmo = np.zeros((n,))


kfold_gen = ShuffleSplit(n_splits=n, test_size=0.2, random_state=random_state)

for i, (train_index, test_index) in enumerate(kfold_gen.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    cgb = C_GradientBoostingClassifier(max_depth=max_depth,
                                       subsample=1.0,
                                       max_features="sqrt",
                                       learning_rate=0.1,
                                       random_state=random_state,
                                       criterion="mse",
                                       loss="deviance",
                                       n_estimators=100)
    t0 = process_time()
    cgb.fit(x_train, y_train)
    t_cgb[i-1] = (process_time()-t0)

    mart = GradientBoostingClassifier(max_depth=max_depth,
                                      subsample=1.0,
                                      max_features="sqrt",
                                      learning_rate=0.1,
                                      random_state=random_state,
                                      criterion="mse",
                                      n_estimators=100)
    t0 = process_time()
    mart.fit(x_train, y_train)
    t_mart[i-1] = (process_time()-t0)

    tfbt = TFBT.BoostedTreesClassifier(label_vocabulary=None,
                                       n_trees=100,
                                       max_depth=max_depth,
                                       learning_rate=0.025,
                                       max_steps=100,
                                       model_dir=temp_path
                                       )

    tfbt.fit(x_train, y_train)
    t_tfbt[i-1] = (tfbt._model_complexity()[0])

    gbdtm = classification(max_depth=max_depth,
                           learning_rate=0.1,
                           random_state=random_state,
                           num_boosters=100,
                           lib=path,
                           subsample=1.0,
                           verbose=False,
                           num_eval=0)

    gbdtm.fit(x_train, y_train)
    t_gbdtmo[i-1] = (gbdtm._model_complexity()[0])

    clear_output()


result = {}
result['Time_CGB'] = t_cgb
result['Time_MART'] = t_mart
result['Time_TFBT'] = t_tfbt
result['Time_GBDTMO'] = t_gbdtmo

mean = pd.DataFrame(result).agg('mean', axis=0)
std = pd.DataFrame(result).agg('std', axis=0)
concat = pd.concat([mean, std], axis=1, keys=['Mean', 'Std'])
pd.DataFrame(result).to_csv('result.csv')
(concat).to_csv('summary.csv')
