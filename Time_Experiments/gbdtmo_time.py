import numpy as np
from time import process_time
import sklearn.datasets as dt
from gbdtmo import GBDTMulti, load_lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer

X, y = dt.load_iris(return_X_y=True)

scl = Normalizer()
X = scl.fit_transform(X)

lb = LabelEncoder()
y = lb.fit_transform(y)

X, y = np.ascontiguousarray(X, dtype=np.float64), y.astype(np.int32)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Define the path of the Dynamic lib from the related directory
path = '/lustre/home/user/.local/lib/python~/site-packages/gbdtmo/build/gbdtmo.so'
lib = load_lib(path)


params = {"max_depth": 5,
          "lr": 0.1,
          'loss': b"ce",
          'verbose': True,
          'seed': 1}

booster = GBDTMulti(lib,
                    out_dim=len(np.unique(y)),
                    params=params)

booster.set_data((x_train, y_train), (x_test, y_test))
t0 = process_time()
booster.train(100)
t_gbdtmo = (process_time()-t0)
print(t_gbdtmo)
