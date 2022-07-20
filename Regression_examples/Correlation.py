# %%

import sys
sys.path.append(r'D:\Academic\Ph.D\Programming\Py\PhD Thesis\CGB\C-GB')

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from cgb import cgb_reg
import matplotlib.pyplot as plt
from scipy.stats import invgauss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


warnings.simplefilter('ignore')
random_state = 1
np.random.seed(random_state)


def data(dt):
    if dt == 'Energy':
        cl = [
            'relative_compactness', 'surface_area', 'wall_area', 'roof_area',
            'overall_height', 'orientation', 'glazing_area',
            'glazing_area_distribution', 'heating_load', 'cooling_load'
        ]
        data = pd.read_csv(
            r'D:\Academic\Ph.D\Programming\Datasets\Regression\energy.data',
            names=cl)
        X = data.drop(['heating_load', 'cooling_load'], axis=1).values
        y = (data[['heating_load', 'cooling_load']]).values
    elif dt == 'atp1d':
        path = r'D:\Academic\Ph.D\Programming\Datasets\Regression\atp1d.csv'
        df = pd.read_csv(path)
        X = (df.iloc[:, 0:411]).values
        y = (df.iloc[:, 411:417]).values
    else:
        path = r'D:\Academic\Ph.D\Programming\Datasets\Regression\atp7d.csv'
        df = pd.read_csv(path)
        X = (df.iloc[:, 0:411]).values
        y = (df.iloc[:, 411:417]).values
    return X, y


def model(max_depth, learning_rate, subsample, max_features, dt, random_state):

    print(dt)
    X, y = data(dt)

    kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
    pred_cgb = np.zeros_like(y)
    pred_gb = np.zeros_like(y)

    for (train_index, test_index) in (kfold.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        c_gb = cgb_reg(max_depth=5,
                       subsample=subsample,
                       max_features=max_features,
                       learning_rate=learning_rate,
                       random_state=random_state,
                       criterion="squared_error",
                       n_estimators=100)

        c_gb.fit(x_train, y_train)
        pred_cgb[test_index] = c_gb.predict(x_test)

        for i in range(y.shape[1]):
            gb = GradientBoostingRegressor(max_depth=max_depth,
                                           subsample=subsample,
                                           max_features=max_features,
                                           learning_rate=learning_rate,
                                           random_state=random_state,
                                           criterion="squared_error",
                                           n_estimators=100)

            gb.fit(x_train, y_train[:, i])
            pred_gb[test_index, i] = gb.predict(x_test)

        print('*', end='')

    return pred_cgb, pred_gb, y


if __name__ == '__main__':

    datasets = ['Energy', 'atp1d', 'atp7d']
    fig1, ax1 = plt.subplots(1, 3, figsize=(20, 5))
    fig2, ax2 = plt.subplots(1, 3, figsize=(20, 5))

    for i, dataset in enumerate(datasets):

        if dataset == 'Energy':
            max_depth = 5
            learning_rate = 0.1
            subsample = 0.75
            max_features = None

        elif dataset == 'atp1d':
            max_depth = 20
            learning_rate = 0.05
            subsample = 0.5
            max_features = None
        else:
            max_depth = 20
            learning_rate = 0.05
            subsample = 0.5
            max_features = None

        pred_cgb, pred_gb, y = model(max_depth=max_depth,
                                     learning_rate=learning_rate,
                                     subsample=subsample,
                                     max_features=max_features,
                                     dt=dataset,
                                     random_state=random_state)

        corr_gb = (pd.DataFrame(y - pred_gb)).corr('pearson')
        corr_cgb = (pd.DataFrame(y - pred_cgb)).corr('pearson')

        sns.heatmap(data=corr_gb, ax=ax1[i], annot=True, cmap='Blues')
        sns.heatmap(data=corr_cgb, ax=ax2[i], annot=True, cmap='Blues')

        ax1[i].set_title(dataset)
        ax2[i].set_title(dataset)

        fig1.suptitle("GB")
        fig2.suptitle("CGB")

        fig1.savefig("GB.jpg", dpi=700)
        fig1.savefig("GB.eps")

        fig2.savefig("CGB.jpg", dpi=700)
        fig2.savefig("CGB.eps")
