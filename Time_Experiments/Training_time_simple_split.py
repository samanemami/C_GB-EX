import numpy as np
import pandas as pd
from time import process_time
import sklearn.datasets as dt
from gbdtmo import GBDTMulti, load_lib
from TFBT import BoostedTreesClassifier
from Scikit_CGB import C_GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


X, y = dt.load_digits(return_X_y=True)


max_depth = 5
random_state = 1
# Define the path of the Dynamic lib from the related directory
path = '/lustre/home/user/.local/lib/python~/site-packages/gbdtmo/build/gbdtmo.so'
lib = load_lib(path)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state)


cgb = C_GradientBoostingClassifier(max_depth=max_depth,
                                   subsample=1,
                                   max_features="sqrt",
                                   learning_rate=0.1,
                                   random_state=random_state,
                                   criterion="mse",
                                   loss="deviance",
                                   n_estimators=100)
# Computing the training time
t0 = process_time()
cgb.fit(x_train, y_train)
t_cgb = (process_time()-t0)

# Computing the classifier time
t0 = process_time()
for i in range(100):
    cgb.predict(x_test)
p_cgb = process_time() - t0 / (100 * (x_test.shape[0]))

mart = GradientBoostingClassifier(max_depth=max_depth,
                                  subsample=1,
                                  max_features="sqrt",
                                  learning_rate=0.1,
                                  random_state=random_state,
                                  criterion="mse",
                                  n_estimators=100)
# Computing the training time
t0 = process_time()
mart.fit(x_train, y_train)
t_mart = (process_time()-t0)

# Computing the classifier time
t0 = process_time()
for i in range(100):
    mart.predict(x_test)
p_mart = process_time() - t0 / (100 * (x_test.shape[0]))

tfbt = BoostedTreesClassifier(label_vocabulary=None,
                              n_trees=100,
                              max_depth=max_depth,
                              learning_rate=0.1,
                              max_steps=100,
                              model_dir='/tempsTFBT/'
                              )

# Computing the training time
tfbt.fit(x_train, y_train)
t_tfbt = (tfbt._model_complexity()[0])

# Computing the classifier time
t0 = process_time()
for i in range(100):
    tfbt.score(x_test, y_test)
p_tfbt = process_time() - t0 / (100 * (x_test.shape[0]))


params = {"max_depth": max_depth,
          "lr": 0.1,
          'loss': b"ce",
          'verbose': True,
          'seed': random_state}


x_train, y_train = np.ascontiguousarray(
    x_train, dtype=np.float64), y_train.astype(np.int32)
x_test, y_test = np.ascontiguousarray(
    x_test, dtype=np.float64), y_test.astype(np.int32)

booster = GBDTMulti(lib,
                    out_dim=len(np.unique(y)),
                    params=params)

booster.set_data((x_train, y_train), (x_test, y_test))
# Computing the training time
t0 = process_time()
booster.train(100)
t_gbdtmo = (process_time()-t0)

# Computing the classifier time
t0 = process_time()
for i in range(100):
    np.argmax(booster.predict(x_test), axis=1)
p_gbdtmo = process_time() - t0 / (100 * (x_test.shape[0]))


result = {}
result['Training_time_CGB'] = t_cgb
result['Training_time_MART'] = t_mart
result['Training_time_TFBT'] = t_tfbt
result['Training_time_GBDTMO'] = t_gbdtmo

result['clf_time_CGB'] = p_cgb
result['clf_time_MART'] = p_mart
result['clf_time_TFBT'] = p_tfbt
result['clf_time_GBDTMO'] = p_gbdtmo

pd.DataFrame(result, index=['values']).to_csv(
    'digits-max depth'+str(max_depth)+'.csv')
