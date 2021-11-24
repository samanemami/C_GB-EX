import sys
import numpy as np
import pandas as pd
import sklearn.datasets as dts
from gbdtmo import GBDTMulti, load_lib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

X, y = dts.load_iris(return_X_y=True)
X, y = np.ascontiguousarray(X, dtype=np.float64), y.astype('int32')


def convert(arg):
    char = ["{", "}", ","]
    for i in char:
        try:
            arg = arg.replace(i, "")
        except:
            pass
    return arg


def opt(cv=2, num=100, random_state=None, loss=b"ce"):

    score = []
    param = {"max_depth": [], "lr": []}
    path = '~/python/site-packages/gbdtmo/build/gbdtmo.so'
    LIB = load_lib(path)

    if loss == b"ce":
        kfold = StratifiedKFold(n_splits=cv, shuffle=True,
                                random_state=random_state)
        n_class = len(np.unique(y))
    else:
        kfold = KFold(n_splits=cv, shuffle=False,
                      random_state=random_state)
        n_class = y.shape[1]

    lr = float(convert(sys.argv[1]))
    depth = int(convert(sys.argv[2]))
    data = sys.argv[3]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    params = {"max_depth": depth, "lr": lr, 'loss': loss,
              'verbose': False, 'subsample': 1.0}

    booster = GBDTMulti(LIB, out_dim=n_class, params=params)

    if data.startswith('train'):
        index = list(kfold.split(x_train, y_train))[0]

        if data == 'train1':
            train = index[0]
            val = index[1]

        else:
            val = index[0]
            train = index[1]

        x_train, x_test = x_train[train], x_train[val]
        y_train, y_test = y_train[train], y_train[val]

        booster.set_data((x_train, y_train))
        booster.train(num)
    else:
        booster.set_data(x_train, y_train)
        booster.train(num)

        score.append(accuracy_score(y_test, np.argmax(
            booster.predict(x_test), axis=1)))

        param['lr'].append(lr)
        param['max_depth'].append(depth)



if __name__ == '__main__':
    opt(cv=2,
        num=100,
        random_state=int(sys.argv[4]),
        loss=b"ce")
