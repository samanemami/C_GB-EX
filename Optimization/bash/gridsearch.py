import os
import sys
import pathlib
import numpy as np
import pandas as pd
from gbdtmo import GBDTMulti, load_lib
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


def gridsearchcv(X, y, num_train, num_test, random_state, loss):

    path = '/home/user/.local/lib/python~/site-packages/gbdtmo/build/gbdtmo.so'
    lib = load_lib(path)
    cv = 2

    if loss == b"ce":
        kfold = StratifiedKFold(n_splits=cv, shuffle=False)
        n_class = len(np.unique(y))
        X, y = np.ascontiguousarray(X, dtype=np.float64), y.astype(np.int32)
    else:
        kfold = KFold(n_splits=cv, shuffle=False)
        n_class = y.shape[1]
        X, y = np.ascontiguousarray(
            X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64)

    data = str(sys.argv[3])

    dftrain, dfeval, ytrain, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    if data.startswith('train'):
        index = list(kfold.split(dftrain, ytrain))[0]

        lr = float(sys.argv[1])
        depth = int(sys.argv[2])

        params = {"max_depth": depth, "lr": lr, 'loss': loss, 'verbose': False}

        booster = GBDTMulti(lib, out_dim=n_class, params=params)

        if data == 'train1':
            train = index[0]
            val = index[1]

        else:
            val = index[0]
            train = index[1]

        x_train, x_test = dftrain[train], dftrain[val]
        y_train, y_test = ytrain[train], ytrain[val]

        booster.set_data((x_train, y_train), (x_test, y_test))
        booster.train(num_train)
        if loss == b"ce":
            pred = np.argmax(booster.predict(x_test), axis=1)
            score = accuracy_score(y_test, pred)
            print(confusion_matrix(y_test, pred))
            print('------------')
        else:
            pred = booster.predict(x_test)
            score = r2_score(y_test, pred)

        pd.DataFrame([[score, depth, lr]], columns=[
                     'score', 'max_depth', 'learning_rate']).to_csv('results.csv', header=False, index=False)
    else:
        param = pd.read_csv('mean_test_score.csv', header=None)
        depth = param.iloc[param.iloc[:, 0].idxmax(), 1]
        lr = param.iloc[param.iloc[:, 0].idxmax(), 2]
        params = {"max_depth": depth, "lr": lr, 'loss': loss, 'verbose': False}

        booster = GBDTMulti(lib, out_dim=n_class, params=params)
        booster.set_data((dftrain, ytrain), (dfeval, y_eval))
        booster.train(num_test)
        if loss == b"ce":
            pred = np.argmax(booster.predict(dfeval), axis=1)
            score = accuracy_score(y_eval, pred)
            cf = confusion_matrix(y_eval, pred)
            pd.DataFrame(cf).to_csv('CF_Matrix.csv', index=False, header=False)
        else:
            pred = booster.predict(dfeval)
            score = np.mean(np.sqrt(np.power(y_eval - pred, 2).sum(axis=1)))
            rmse = np.sqrt(np.average(
                (y_eval - booster.predict(dfeval))**2, axis=0))

        pd.DataFrame([[score, depth, lr]], columns=[
                     'score', 'max_depth', 'learning_rate']).to_csv('mean_generalization_score.csv', index=False)
        pd.DataFrame(pred).to_csv('pred.csv', index=None, header=None)
        try:
            pd.Series(rmse).to_csv('RMSE.csv')
        except:
            pass

        try:
            for root, dirs, files in os.walk(pathlib.Path().resolve()):
                os.remove(os.path.join(root, 'results.csv'))
        except:
            pass
