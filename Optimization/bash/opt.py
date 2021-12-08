# Check the score of regression
# unload the library
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import sklearn.datasets as dts
from sklearn.metrics import r2_score
from gbdtmo import GBDTMulti, load_lib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

X, y = dts.load_digits(return_X_y=True)
X, y = np.ascontiguousarray(X, dtype=np.float64), y.astype(np.int32)


def opt(cv=2, num=100, random_state=None, loss=b"ce", unload_lib=False):

    path = '/home/user/.local/lib/python3.8/site-packages/gbdtmo/build/gbdtmo.so'
    LIB = load_lib(path)

    if loss == b"ce":
        kfold = StratifiedKFold(n_splits=cv, shuffle=False)
        n_class = len(np.unique(y))
    else:
        kfold = KFold(n_splits=cv, shuffle=False)
        n_class = y.shape[1]

    data = str(sys.argv[3])

    dftrain, dfeval, ytrain, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    if data.startswith('train'):
        index = list(kfold.split(dftrain, ytrain))[0]

        lr = float(sys.argv[1])
        depth = int(sys.argv[2])

        params = {"max_depth": depth, "lr": lr, 'loss': loss,
                  'verbose': False, 'subsample': 1.0, 'seed': random_state}

        booster = GBDTMulti(LIB, out_dim=n_class, params=params)

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
        if loss == b"ce":
            score = accuracy_score(y_test, np.argmax(
                booster.predict(x_test), axis=1))
        else:
            score = r2_score(y_test, booster.predict(x_test))

        pd.DataFrame([[score, depth, lr]], columns=[
                     'score', 'max_depth', 'learning_rate']).to_csv('results.csv', header=False, index=False)
    else:
        param = pd.read_csv('mean_test_score.csv', header=None)
        depth = param.iloc[param.iloc[:, 0].idxmax(), 1]
        lr = param.iloc[param.iloc[:, 0].idxmax(), 2]
        params = {"max_depth": depth, "lr": lr, 'loss': loss,
                  'verbose': False, 'subsample': 1.0, 'seed': random_state}

        booster = GBDTMulti(LIB, out_dim=n_class, params=params)
        booster.set_data((dftrain, ytrain))
        booster.train(num)
        if loss == b"ce":
            score = accuracy_score(y_eval, np.argmax(
                booster.predict(dfeval), axis=1))
        else:
            score = np.mean(
                np.sqrt(np.power(y_eval - booster.predict(dfeval), 2).sum(axis=1)))
            rmse = np.sqrt(np.average(
                (y_eval - booster.predict(dfeval))**2, axis=0))

        pd.DataFrame([[score, depth, lr]], columns=[
                     'score', 'max_depth', 'learning_rate']).to_csv('mean_generalization_score.csv', index=False)
        try:
            pd.Series(rmse).to_csv('RMSE.csv')
        except:
            pass

        try:
            for root, dirs, files in os.walk(pathlib.Path().resolve()):
                os.remove(os.path.join(root, 'results.csv'))
        except:
            pass

    if unload_lib:
        _del_ctype()


def _del_ctype(lib):
    lib.FreeLibrary(
        lib._handle) if sys.platform == 'win32' else lib.dlclose(lib._handle)
    del lib
    lib = None


if __name__ == '__main__':
    opt(cv=2,
        num=100,
        random_state=1,
        loss=b"ce",
        unload_lib=False)
