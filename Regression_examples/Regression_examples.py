from sklearn.ensemble import GradientBoostingRegressor
from Scikit_CGB import C_GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import sklearn.datasets as dts
import numpy as np
import warnings


warnings.simplefilter('ignore')
random_state = 123
np.random.seed(random_state)

X, y = dts.make_regression(n_samples=200,
                           n_targets=2,
                           random_state=random_state)


kfold = KFold(n_splits=10,
              shuffle=True,
              random_state=random_state
              )


def reg(max_depth=2):

    pred_mart = np.zeros_like(y)
    pred_cgb = np.zeros_like(y)

    for _, (train_index, test_index) in enumerate(kfold.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        c_gb = C_GradientBoostingRegressor(max_depth=max_depth,
                                           subsample=1,
                                           max_features="sqrt",
                                           learning_rate=0.1,
                                           random_state=1,
                                           criterion="mse",
                                           n_estimators=100)

        c_gb.fit(x_train, y_train)
        pred_cgb[test_index] = c_gb.predict(x_test)

        mart = GradientBoostingRegressor(max_depth=max_depth,
                                         subsample=1,
                                         max_features="sqrt",
                                         learning_rate=0.1,
                                         random_state=1,
                                         criterion="mse",
                                         n_estimators=100)
        for i in range(y.shape[1]):
            mart.fit(x_train, y_train[:, i])
            pred_mart[test_index, i] = mart.predict(x_test)

    return pred_cgb, pred_mart


def scatter(y, pred_cgb, pred_mart):

    plt.scatter(
        y[:, 0],
        y[:, 1],
        edgecolor="k",
        c="navy",
        s=20,
        marker="s",
        alpha=0.5,
        label="Real values",
    )

    plt.scatter(
        pred_cgb[:, 0],
        pred_cgb[:, 1],
        edgecolor="k",
        c="cornflowerblue",
        s=20,
        alpha=0.5,
        label='C-GB=%.2f' % r2_score(y, pred_cgb)
    )

    plt.scatter(
        pred_mart[:, 0],
        pred_mart[:, 1],
        edgecolor="k",
        c="c",
        s=20,
        marker="^",
        alpha=0.5,
        label='MART=%.2f' % r2_score(y, pred_mart),
    )

    plt.xlabel("target 1")
    plt.ylabel("target 2")
    plt.legend()


if __name__ == '__main__':

    plt.figure(figsize=(10, 7))
    for i, j in enumerate([2, 5, 10, 20]):
        pred_cgb, pred_mart = reg(max_depth=j)
        plt.subplot(2, 2, i+1)
        scatter(y, pred_cgb, pred_mart)
        plt.suptitle('Comparing C-GB and MART Regressors')
        plt.title('Max depth=' + str(j))
    plt.subplots_adjust(hspace=0.3, wspace=.3)
    plt.savefig('Scatter_regression.eps')
