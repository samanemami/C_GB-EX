import sys
sys.path.append('/lustre/scratch/samanema/Time/')
from gbdtmo import GBDTMulti, load_lib
from sklearn.model_selection import train_test_split
import sklearn.datasets as dt
from time import process_time
import pandas as pd
import numpy as np
from TFBT import BoostedTreesClassifier
from Scikit_CGB import C_GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score

X, y = dt.load_iris(return_X_y=True)



n_class = len(np.unique(y))
max_depth = 5
random_state = 1
path = '/lustre/home/samanema/.local/lib/python3.6/site-packages/gbdtmo/build/gbdtmo.so'
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
t0 = process_time()
cgb.fit(x_train, y_train)
t_cgb = (process_time()-t0)

mart = GradientBoostingClassifier(max_depth=max_depth,
                                  subsample=1,
                                  max_features="sqrt",
                                  learning_rate=0.1,
                                  random_state=random_state,
                                  criterion="mse",
                                  n_estimators=100)
t0 = process_time()
mart.fit(x_train, y_train)
t_mart = (process_time()-t0)

tfbt = BoostedTreesClassifier(label_vocabulary=None,
                              n_trees=100,
                              max_depth=max_depth,
                              learning_rate=0.1,
                              max_steps=100,
                              model_dir='/lustre/scratch/samanema/tempsTFBT/time_vehicle'
                              )

tfbt.fit(x_train, y_train)
t_tfbt = (tfbt._model_complexity()[0])


params = {"max_depth": max_depth, "lr": 0.1,
          'loss': b"ce", 'verbose': True, 'seed': random_state}


lb = LabelEncoder()
y = lb.fit_transform(y)

scl = Normalizer()
x_train = scl.fit_transform(x_train)


lb = LabelEncoder()
y_test = lb.fit_transform(y_test)


x_test = scl.fit_transform(x_test)
y_train = lb.fit_transform(y_train)

x_train, y_train = np.ascontiguousarray(x_train, dtype=np.float64), y_train.astype(np.int32)
x_test, y_test = np.ascontiguousarray(x_test, dtype=np.float64), y_test.astype(np.int32)





booster = GBDTMulti(lib, out_dim=n_class, params=params)
booster.set_data((x_train, y_train), (x_test, y_test))
t0 = process_time()
booster.train(100)
t_gbdtmo = (process_time()-t0)


result = {}
result['Time_CGB'] = t_cgb
result['Time_MART'] = t_mart
result['Time_TFBT'] = t_tfbt
result['Time_GBDTMO'] = t_gbdtmo


pd.DataFrame(result, index=['values']).to_csv(
    'iris-max depth'+str(max_depth)+'.csv')
