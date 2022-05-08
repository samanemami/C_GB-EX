import warnings
import numpy as np
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from Scikit_CGB import C_GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor


warnings.simplefilter('ignore')
random_state = 123
np.random.seed(random_state)

X, y = dts.make_regression(n_samples=1500,
                           n_targets=2,
                           random_state=random_state)


def reg(X, y, max_depth, random_state):

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    pred_mart = np.zeros_like(y_test)

    c_gb = C_GradientBoostingRegressor(max_depth=max_depth,
                                       subsample=0.75,
                                       max_features="sqrt",
                                       learning_rate=0.1,
                                       random_state=random_state,
                                       criterion="mse",
                                       n_estimators=100)

    c_gb.fit(x_train, y_train)
    pred_cgb = c_gb.predict(x_test)

    mart = GradientBoostingRegressor(max_depth=max_depth,
                                     subsample=0.75,
                                     max_features="sqrt",
                                     learning_rate=0.1,
                                     random_state=random_state,
                                     criterion="mse",
                                     n_estimators=100)
    for i in range(y.shape[1]):
        mart.fit(x_train, y_train[:, i])
        pred_mart[:, i] = mart.predict(x_test)

    return pred_cgb, pred_mart, y_test


def scatter(y, pred_cgb, pred_mart):

    plt.scatter(
        y[:, 0],
        y[:, 1],
        edgecolor="k",
        c="navy",
        s=5,
        marker="s",
        alpha=0.3,
        label="Real values",
    )

    plt.scatter(
        pred_cgb[:, 0],
        pred_cgb[:, 1],
        edgecolor="k",
        c="cornflowerblue",
        s=5,
        alpha=0.3,
        label='C-GB=%.2f' % r2_score(y, pred_cgb)
    )

    plt.scatter(
        pred_mart[:, 0],
        pred_mart[:, 1],
        edgecolor="k",
        c="c",
        s=5,
        marker="^",
        alpha=0.3,
        label='MART=%.2f' % r2_score(y, pred_mart),
    )

    plt.xlabel("target 1")
    plt.ylabel("target 2")
    # plt.xscale("log")
    plt.legend()


def hexbin(x, y, xlabel, ylabel, title):
    plt.hexbin(x, y, gridsize=15, mincnt=1, edgecolors="none", cmap="inferno")
    plt.scatter(x, y, s=2, c="white")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()


if __name__ == '__main__':

    n = 10
    depth = [2, 5, 10, 20]
    plt.figure(figsize=(10, 7))
    for _, j in enumerate(depth):
        y_test = reg(X=X,
                     y=y,
                     max_depth=j,
                     random_state=_)[2]

        cgb = np.zeros_like(y_test)
        mart = np.zeros_like(y_test)
        Y = np.zeros_like(y_test)
        for i in range(n):
            pred_cgb, pred_mart, y_test = reg(X=X,
                                              y=y,
                                              max_depth=j,
                                              random_state=i)
            cgb += pred_cgb
            mart += pred_mart
            Y += y_test

        pred_cgb = cgb / n
        pred_mart = mart / n
        Y = Y/n

        np.savetxt('pred_cgb' + str(j) + '.csv', pred_cgb, delimiter=',')
        np.savetxt('pred_mart' + str(j) + '.csv', pred_mart, delimiter=',')
        np.savetxt('y_test' + str(j) + '.csv', y_test, delimiter=',')

        plt.subplot(2, 2, _+1)
        scatter(Y, pred_cgb, pred_mart)
        plt.suptitle('Comparing C-GB and MART Regressors')
        plt.title('Max depth=' + str(j))

    plt.subplots_adjust(hspace=0.3, wspace=.3)
    plt.savefig('Scatter_regression.jpg', dpi=700)
    plt.close('all')

    fig1, axs1 = plt.subplots(4, 2, figsize=(10, 15))
    fig2, axs2 = plt.subplots(4, 2, figsize=(10, 15))
    for i, j in enumerate(depth):
        pred_cgb = np.loadtxt('pred_cgb'+str(j) +
                              '.csv', delimiter=',')
        pred_mart = np.loadtxt('pred_mart'+str(j) +
                               '.csv', delimiter=',')
        y_test = np.loadtxt('y_test'+str(j) +
                            '.csv', delimiter=',')

        for _ in range(y_test.shape[1]):
            plt.cla()

            axs1[i, _].hexbin(y_test[:, _], pred_cgb[:, _], gridsize=15,
                              mincnt=1, edgecolors="none", cmap="inferno")
            axs1[i, _].scatter(y_test[:, _], pred_cgb[:, _], s=2, c="white")
            axs1[i, _].set_xlabel('real values')
            axs1[i, _].set_ylabel('predicted values')
            axs1[i, _].set_title('\n' + 'Max depth=' +
                                 str(j) + '\n' +
                                 'target=' + str(_))
            plt.cla()

            axs2[i, _].hexbin(y_test[:, _], pred_mart[:, _], gridsize=15,
                              mincnt=1, edgecolors="none", cmap="inferno")
            axs2[i, _].scatter(y_test[:, _], pred_mart[:, _], s=2, c="white")
            axs2[i, _].set_xlabel('real values')
            axs2[i, _].set_ylabel('predicted values')
            axs2[i, _].set_title('\n' + 'Max depth=' +
                                 str(j) + '\n' +
                                 'target=' + str(_))
            fig2.tight_layout()
            fig1.tight_layout()

    fig1.suptitle('C-GB')
    fig2.suptitle('MART')
    fig1.savefig('hexbin_cgb.jpg', dpi=500)
    fig2.savefig('hexbin_mart.jpg', dpi=500)
