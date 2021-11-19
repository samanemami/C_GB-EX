import sys
import numpy as np
import sklearn.datasets as dts
from gbdtmo import GBDTMulti, load_lib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold

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


def opt(cv=2, T=100, random_state=1, loss=b"ce"):

    acc = []
    path = '/lustre/home/samanema/.local/lib/python3.6/site-packages/gbdtmo/build/gbdtmo.so'
    LIB = load_lib(path)

    kfold = StratifiedKFold(n_splits=cv, shuffle=True,
                            random_state=random_state)

    lr = float(convert(sys.argv[1]))
    depth = int(convert(sys.argv[2]))
    data = sys.argv[3]

    params = {"max_depth": depth, "lr": lr, 'loss': loss,
              'verbose': False, 'subsample': 1.0}

    booster = GBDTMulti(LIB, out_dim=len(np.unique(y)), params=params)
    for cv_i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if data.startswith('train'):
            index = list(kfold.split(x_train, y_train))[0]

            if data == 'train1':
                train = index[0]
                val = index[1]

            else:
                val = index[0]
                train = index[1]

            x_train, x_test = X[train], X[val]
            y_train, y_test = y[train], y[val]
            booster.set_data((x_train, y_train))
            booster.train(T)

            acc.append(accuracy_score(y_test, np.argmax(
                booster.predict(x_test), axis=1)))
        else:
            booster.set_data((x_train, y_train))
            booster.train(T)

            acc.append(accuracy_score(y_test, np.argmax(
                booster.predict(x_test), axis=1)))

    return acc


if __name__ == '__main__':
    opt(cv=2, T=100, random_state=1, loss=b"ce")
