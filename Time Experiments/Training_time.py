#%%
from IPython.display import clear_output
from Scikit_CGB import C_GradientBoostingClassifier
from TFBT import BoostedTreesClassifier
from sklearn.model_selection import ShuffleSplit
import sklearn.datasets as dt
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from time import process_time
import pandas as pd


X, y = dt.load_iris(return_X_y=True)

max_depth = 10
random_state = 1
n = 10


t_cgb = np.zeros((n,))
t_mart = np.zeros((n,))
t_tfbt = np.zeros((n,))
t_gbdtmo = np.zeros((n,))


gbnn_err = []


kfold_gen = ShuffleSplit(n_splits=n, test_size=0.2, random_state=random_state)

for i, (train_index, test_index) in enumerate(kfold_gen.split(X, y)):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    cgb = C_GradientBoostingClassifier(max_depth=max_depth,
                                       subsample=0.5,
                                       #    max_features="sqrt",
                                       learning_rate=0.05,
                                       random_state=random_state,
                                       criterion="mse",
                                       loss="deviance",
                                       n_estimators=100)
    t0 = time()
    cgb.fit(x_train, y_train)
    t_cgb[i-1] = (time()-t0)

    mart = GradientBoostingClassifier(max_depth=max_depth,
                                      subsample=0.75,
                                      max_features="sqrt",
                                      learning_rate=0.025,
                                      random_state=random_state,
                                      criterion="mse",
                                      n_estimators=100)
    t0 = time()
    mart.fit(x_train, y_train)
    t_mart[i-1] = (time()-t0)

    tfbt = TFBT.BoostedTreesClassifier(label_vocabulary=None,
                                       n_trees=100,
                                       max_depth=max_depth,
                                       learning_rate=0.025,
                                       max_steps=100,
                                       model_dir=r'C:\Users\saman\Downloads\iristfbt'
                                       )

    tfbt.fit(x_train, y_train)
    t_tfbt[i-1] = (tfbt._training_time())
    clear_output()
    print('*', end='')

result = {}
result['Time_CGB'] = t_cgb
result['Time_MART'] = t_mart
result['Time_TFBT'] = t_tfbt

mean = pd.DataFrame(result).agg('mean', axis=0)
std = pd.DataFrame(result).agg('std', axis=0)
concat = pd.concat([mean, std], axis=1, keys=['Mean', 'Std'])
pd.DataFrame(result).to_csv('Asus_iris-max depth'+str(max_depth)+'.csv')
(concat).to_csv('Asus_iris_summary_max_depth'+str(max_depth)+'.csv')
mean
