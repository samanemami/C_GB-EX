from sklearn.ensemble import GradientBoostingRegressor
from cgb import cgb_reg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings



warnings.simplefilter('ignore')
random_state = 123
np.random.seed(random_state)


def data():
    cl = [
        'relative_compactness', 'surface_area', 'wall_area', 'roof_area',
        'overall_height', 'orientation', 'glazing_area',
        'glazing_area_distribution', 'heating_load', 'cooling_load'
    ]
    data = pd.read_csv('energy.data', names=cl)
    X = data.drop(['heating_load', 'cooling_load'], axis=1).values
    y = (data[['heating_load', 'cooling_load']]).values
    return X, y


def model(max_depth, random_state):

    X, y = data()

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    c_gb = cgb_reg(max_depth=max_depth,
                   subsample=0.75,
                   max_features="sqrt",
                   learning_rate=0.2,
                   random_state=random_state,
                   criterion="squared_error",
                   n_estimators=100)

    c_gb.fit(x_train, y_train)
    pred_cgb = c_gb.predict(x_test)

    gb = GradientBoostingRegressor(max_depth=max_depth,
                                   subsample=0.75,
                                   max_features="sqrt",
                                   learning_rate=0.2,
                                   random_state=random_state,
                                   criterion="squared_error",
                                   n_estimators=100)
    pred_gb = np.zeros_like(y_test)
    for i in range(y_train.shape[1]):
        gb.fit(x_train, y_train[:, i])
        pred_gb[:, i] = gb.predict(x_test)

    return pred_cgb, pred_gb, y_test


def scatter(y, pred_cgb, pred_gb, axs):

    axs.scatter(
        y[:, 0],
        y[:, 1],
        # edgecolor="k",
        c="g",
        s=20,
        marker="s",
        alpha=0.3,
        label="Real values",
    )

    axs.scatter(
        pred_cgb[:, 0],
        pred_cgb[:, 1],
        # edgecolor="k",
        c="royalblue",
        s=20,
        alpha=0.3,
        label='C-GB=%.3f' % r2_score(y, pred_cgb)
    )

    axs.scatter(
        pred_gb[:, 0],
        pred_gb[:, 1],
        # edgecolor="k",
        c="salmon",
        s=20,
        marker="^",
        alpha=0.3,
        label='GB=%.3f' % r2_score(y, pred_gb),
    )

    axs.set_xlabel("Heating")
    axs.set_ylabel("Cooling")
    axs.grid(True)
    # plt.xscale("log")
    axs.legend()


def hexbin(x, y, xlabel, ylabel, title):
    plt.hexbin(x, y, gridsize=15, mincnt=1, edgecolors="none", cmap="inferno")
    plt.scatter(x, y, s=2, c="white")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()


def rmse(y, pred, target):
    error = mean_squared_error(y[:, target], pred[:, target], squared=False)
    return "RMSE: " + str("{:.2f}".format(error))


if __name__ == '__main__':

    n = 10
    size = (10, 7)

    y_test = model(max_depth=5, random_state=1)[2]

    cgb = np.zeros_like(y_test)
    gb = np.zeros_like(y_test)
    y = np.zeros_like(y_test)

    fig1, axs1 = plt.subplots(2, 2, figsize=size)
    for i in range(n):
        pred_cgb, pred_gb, y_test = model(max_depth=5,
                                          random_state=1)
        cgb += pred_cgb
        gb += pred_gb
        y += y_test

    pred_cgb = cgb / n
    pred_gb = gb / n
    y = y/n

    # Plot the Scatter, and histograms
    scatter(y, pred_cgb, pred_gb, axs1[0][0])
    axs1[0][0].set_title('Data point distribution')

    euclidean_gb = np.sqrt(np.power(y - pred_gb, 2).sum(axis=1))
    euclidean_cgb = np.sqrt(np.power(y - pred_cgb, 2).sum(axis=1))

    dist_cgb = np.sqrt(np.power(y - pred_cgb, 2))
    dist_gb = np.sqrt(np.power(y - pred_gb, 2))

    sns.distplot(a=euclidean_gb, hist=True, kde=True, rug=False,
                 label='GB', color='salmon', hist_kws={"alpha": 0.6},
                 kde_kws={"color": "r", "lw": 2, "label": "GB"},
                 ax=axs1[0][1])
    sns.distplot(a=euclidean_cgb, hist=True, kde=True,
                 rug=False, label='C-GB', color='royalblue',
                 hist_kws={"histtype": "step", "linewidth": 3,
                           "alpha": 0.7},
                 kde_kws={"color": "b", "lw": 2, "label": "C-GB"}, ax=axs1[0][1])

    axs1[0][1].set_xlabel("Euclidean Distance")
    axs1[0][1].set_title("Histogram of the euclidean distance")
    axs1[0][1].legend()
    axs1[0][1].grid(True)

    # Maximum Distance
    sns.distplot(a=np.max(dist_gb, axis=0), hist=True, kde=True,
                 rug=False, label='GB', color='salmon', hist_kws={"alpha": 0.6},
                 kde_kws={"color": "r", "lw": 2, "label": "KGB"}, ax=axs1[1][0])
    sns.distplot(a=np.max(dist_cgb, axis=0), hist=True, kde=True,
                 rug=False, label='C-GB', color='royalblue',
                 kde_kws={"color": "b", "lw": 2, "label": "C-GB"},
                 hist_kws={"histtype": "step", "linewidth": 3,
                           "alpha": 0.7}, ax=axs1[1][0])

    # Minimum Distance
    sns.distplot(a=np.min(dist_gb, axis=0), hist=True, kde=True,
                 rug=False, label='GB', color='salmon', hist_kws={"alpha": 0.6},
                 kde_kws={"color": "r", "lw": 2, "label": "GB"}, ax=axs1[1][1])
    sns.distplot(a=np.min(dist_cgb, axis=0), hist=True, kde=True,
                 rug=False, label='C-GB', color='royalblue',
                 kde_kws={"color": "b", "lw": 2, "label": "C-GB"},
                 hist_kws={"histtype": "step", "linewidth": 3,
                           "alpha": 0.7}, ax=axs1[1][1])

    axs1[1][0].set_xlabel("Distance")
    axs1[1][0].set_title("Maximum Distance")
    axs1[1][0].legend()
    axs1[1][0].grid(True)

    axs1[1][1].set_xlabel("Distance")
    axs1[1][1].set_title("Minimum Distance")
    axs1[1][1].legend()
    axs1[1][1].grid(True)

    fig1.suptitle("Comparing C-GB and GB")
    fig1.tight_layout()
    fig1.savefig('regression_plt.eps')
    plt.close('all')

    # Plot the hexbin
    fig2, axs2 = plt.subplots(2, 2, figsize=size)
    fig2.subplots_adjust(hspace=0, wspace=0)

    labels = ["Heating", "Cooling"]
    for target, lb in enumerate(labels):
        plt.cla()

        axs2[target, 0].hexbin(y[:, target], pred_cgb[:, target], gridsize=15,
                               mincnt=1, edgecolors="none", cmap="viridis",
                               label=rmse(y, pred_cgb, target))
        axs2[target, 0].scatter(
            y_test[:, target], pred_cgb[:, target], s=2, c="white")
        axs2[target, 0].set_xlabel('real values')
        axs2[target, 0].text(0.95, 0.1, 'target: ' + lb,
                             verticalalignment='bottom',
                             horizontalalignment='right',
                             transform=axs2[target, 0].transAxes,
                             color='k',
                             )
        axs2[target, 0].legend(loc=2)

        plt.cla()

        axs2[target, 1].hexbin(y_test[:, target], pred_gb[:, target], gridsize=15,
                               mincnt=1, edgecolors="none", cmap="viridis",
                               label=rmse(y, pred_gb, target))
        axs2[target, 1].scatter(
            y_test[:, target], pred_gb[:, target], s=2, c="white")
        axs2[target, 1].set_xlabel('real values')
        axs2[target, 1].text(0.95, 0.1, 'target: ' + lb,
                             verticalalignment='bottom',
                             horizontalalignment='right',
                             transform=axs2[target, 1].transAxes,
                             color='k',
                             )
        axs2[target, 1].legend(loc=2)

    axs2[0, 0].set_ylabel('predicted values')
    axs2[1, 0].set_ylabel('predicted values')

    axs2[0, 1].set_yticks([])
    axs2[1, 1].set_yticks([])

    axs2[0, 0].set_title('C-GB')
    axs2[0, 1].set_title('GB')

    fig2.suptitle('The relationship between predicted and real data points')
    fig2.savefig('Hexbin.eps')
