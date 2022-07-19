import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from cgb import cgb_reg
import matplotlib.pyplot as plt
from scipy.stats import invgauss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


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
                   learning_rate=0.75,
                   random_state=random_state,
                   criterion="squared_error",
                   n_estimators=100)

    c_gb.fit(x_train, y_train)
    pred_cgb = c_gb.predict(x_test)

    gb = GradientBoostingRegressor(max_depth=max_depth,
                                   subsample=0.75,
                                   max_features="sqrt",
                                   learning_rate=0.75,
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
        # alpha=0.3,
        label="Real values",
    )

    axs.scatter(
        pred_cgb[:, 0],
        pred_cgb[:, 1],
        # edgecolor="k",
        c="royalblue",
        s=20,
        # alpha=0.3,
        label='C-GB - R2-Score: %.3f' % r2_score(y, pred_cgb)
    )

    axs.scatter(
        pred_gb[:, 0],
        pred_gb[:, 1],
        # edgecolor="k",
        c="salmon",
        s=20,
        marker="^",
        # alpha=0.3,
        label='GB - R2-Score: %.3f' % r2_score(y, pred_gb),
    )

    axs.set_xlabel("Heating")
    axs.set_ylabel("Cooling")
    axs.grid(True)
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


def distance(y, pred, agg):
    distance_ = np.zeros((y.shape[0], ))
    for i in range(y.shape[0]):
        distance_[i] = agg(np.sqrt(np.power(y[i, :] - pred[i, :], 2)))

    return distance_


if __name__ == '__main__':

    n = 10  # Training times
    size = (10, 7)  # Subplots size
    depth = 5

    y_test = model(max_depth=depth, random_state=random_state)[2]

    cgb = np.zeros_like(y_test)
    gb = np.zeros_like(y_test)
    y = np.zeros_like(y_test)

    fig1, axs1 = plt.subplots(2, 2, figsize=size)

    for i in range(n):
        pred_cgb, pred_gb, y_test = model(max_depth=depth,
                                          random_state=random_state)
        cgb += pred_cgb
        gb += pred_gb
        y += y_test

    pred_cgb = cgb / n
    pred_gb = gb / n
    y = y/n

    # Plot the Scatter, and histograms
    scatter(y, pred_cgb, pred_gb, axs1[0][0])
    axs1[0][0].set_title('Data point distribution')

    # Normalizing the output space
    scale = StandardScaler()
    y_scl = scale.fit_transform(y)
    pred_gb_scl = scale.fit_transform(pred_gb)
    pred_cgb_scl = scale.fit_transform(pred_cgb)

    euclidean_gb = np.sqrt(np.power(y_scl - pred_gb_scl, 2).sum(axis=1))
    euclidean_cgb = np.sqrt(np.power(y_scl - pred_cgb_scl, 2).sum(axis=1))

    sns.distplot(a=euclidean_gb,  kde=False, fit=invgauss, bins=10,
                 label='GB', color='salmon',
                 fit_kws={"color": "r", "lw": 1, "label": "GB"},
                 ax=axs1[0][1])
    sns.distplot(a=euclidean_cgb,  kde=False, fit=invgauss, bins=10,
                 label='C-GB', color='b',
                 hist_kws={"histtype": "step",
                           "linewidth": 1,
                           "linestyle": "--"},
                 fit_kws={"color": "b", "lw": 1,
                          "label": "C-GB", "linestyle": "--"},
                 ax=axs1[0][1])

    axs1[0][1].set_xlabel("Euclidean Distance")
    axs1[0][1].set_title("Histogram of the euclidean distance")
    axs1[0][1].legend()
    axs1[0][1].grid(True)

    # Maximum (Between two targets) Distance
    max_dist_gb = distance(y_scl, pred_gb_scl, np.max)
    sns.distplot(a=max_dist_gb,  kde=False,
                 fit=invgauss, label='GB', color='salmon', bins=10,
                 fit_kws={"color": "r", "lw": 1, "label": "GB"},
                 hist_kws={"histtype": "bar"},
                 ax=axs1[1][0])
    max_dist_cgb = distance(y_scl, pred_cgb_scl, np.max)
    sns.distplot(a=max_dist_cgb,  kde=False,
                 fit=invgauss, label='C-GB', color='b', bins=10,
                 fit_kws={"color": "b", "lw": 1,
                          "label": "C-GB", "linestyle": "--"},
                 hist_kws={"histtype": "step",
                           "linewidth": 1, 'linestyle': '--'},
                 ax=axs1[1][0])

    axs1[1][0].set_xlabel("Distance")
    axs1[1][0].set_title("Maximum Distance (Between targets)")
    axs1[1][0].legend()
    axs1[1][0].grid(True)

    # Minimum (Between two targets) Distance
    min_dist_gb = distance(y_scl, pred_gb_scl, np.min)
    sns.distplot(a=min_dist_gb,  kde=False,
                 fit=invgauss, label='GB', color='salmon', bins=10,
                 fit_kws={"color": "r", "lw": 1, "label": "GB"},
                 ax=axs1[1][1])
    min_dist_cgb = distance(y_scl, pred_cgb_scl, np.min)
    sns.distplot(a=min_dist_cgb,  kde=False,
                 fit=invgauss, label='C-GB', color='b', bins=10,
                 fit_kws={"color": "b", "lw": 1,
                          "label": "C-GB", "linestyle": "--"},
                 hist_kws={"histtype": "step", "linewidth": 1, "linestyle": "--"
                           },
                 ax=axs1[1][1])

    # Compute the average of Maximum and Minimum
    # distance between the predicted and real values
    print('Min_ave_CGB:', np.mean(min_dist_cgb))
    print('Max_ave_CGB:', np.mean(max_dist_cgb))

    print('Min_ave_GB:', np.mean(min_dist_gb))
    print('Max_ave_GB:', np.mean(max_dist_gb))

    # Compute the Pearson Correlation between the pairwise targets
    ((pd.DataFrame(y_scl - pred_cgb_scl)).corr('pearson')).to_csv('corr_CGB.csv')
    ((pd.DataFrame(y_scl - pred_gb_scl)).corr('pearson')).to_csv('corr_GB.csv')

    axs1[1][1].set_xlabel("Distance")
    axs1[1][1].set_title("Minimum Distance (Between targets)")
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
    plt.close('all')
