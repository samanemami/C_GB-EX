import sys
import numpy as np
import pandas as pd
import sklearn.datasets as dts
from gbdtmo import GBDTMulti, load_lib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

X, y = dts.load_iris(return_X_y=True)
X, y = np.ascontiguousarray(X, dtype=np.float64), y.astype('int32')


def opt(cv=2, num=100, random_state=None, loss=b"ce"):

    path = '~/python/site-packages/gbdtmo/build/gbdtmo.so'
    LIB = load_lib(path)

    if loss == b"ce":
        kfold = StratifiedKFold(n_splits=cv, shuffle=False)
        n_class = len(np.unique(y))
    else:
        kfold = KFold(n_splits=cv, shuffle=False)
        n_class = y.shape[1]

    lr = float(sys.argv[1])
    depth = int(sys.argv[2])
    data = str(sys.argv[3])

    dftrain, dfeval, ytrain, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    params = {"max_depth": depth, "lr": lr, 'loss': loss,
              'verbose': False, 'subsample': 1.0}

    booster = GBDTMulti(LIB, out_dim=n_class, params=params)

    if data.startswith('train'):
        index = list(kfold.split(dftrain, ytrain))[0]

        if data == 'train1':
            train = index[0]
            val = index[1]

        else:
            val = index[0]
            train = index[1]

        x_train, x_test = dftrain[train], dftrain[val]
        y_train, y_test = ytrain[train], ytrain[val]

        booster.set_data((x_train, y_train))
        booster.train(num)

        score = accuracy_score(y_test, np.argmax(
            booster.predict(x_test), axis=1))
    else:
        booster.set_data(dftrain, ytrain)
        booster.train(num)

    score = accuracy_score(y_eval, np.argmax(
        booster.predict(dfeval), axis=1))


if __name__ == '__main__':
    opt(cv=2,
        num=100,
        random_state=int(sys.argv[4]),
        loss=b"ce")
