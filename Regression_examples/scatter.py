from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import invgauss
import matplotlib.pyplot as plt
from cgb import cgb_reg
import seaborn as sns
import pandas as pd
import numpy as np
import warnings


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
        data = pd.read_csv('energy.data', names=cl)
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

    X, y = data(dt)

    kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
    pred_cgb = np.zeros_like(y)
    pred_gb = np.zeros_like(y)

    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
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

        gb = GradientBoostingRegressor(max_depth=max_depth,
                                       subsample=subsample,
                                       max_features=max_features,
                                       learning_rate=learning_rate,
                                       random_state=random_state,
                                       criterion="squared_error",
                                       n_estimators=100)

        gb.fit(x_train, y_train[:, 0])
        pred_gb[test_index, 0] = gb.predict(x_test)

        gb = GradientBoostingRegressor(max_depth=max_depth,
                                       subsample=0.75,
                                       #    max_features="sqrt",
                                       learning_rate=0.1,
                                       random_state=random_state,
                                       criterion="squared_error",
                                       n_estimators=100)

        gb.fit(x_train, y_train[:, 1])
        pred_gb[test_index, 1] = gb.predict(x_test)

    return pred_cgb, pred_gb, y


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

    size = (10, 7)  # Subplots size
    depth = 5

    fig0, axs0 = plt.subplots(1, 1)
    fig1, axs1 = plt.subplots(2, 2, figsize=size)
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    fig3, axs3 = plt.subplots(2, 2, figsize=size)
    fig3.subplots_adjust(hspace=0.1, wspace=0.1)

    pred_cgb, pred_gb, y = model(max_depth=5, learning_rate=0.1,
                                 subsample=0.75, max_features=None, dt='Energy',
                                 random_state=random_state)

    # Plot the Scatter, and histograms
    scatter(y, pred_cgb, pred_gb, axs1[0][0])
    scatter(y, pred_cgb, pred_gb, axs0)
    axs1[0][0].set_title('Data point distribution')
    axs0.set_title('Data point distribution')

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
    diff_gb = y_scl - pred_gb_scl
    diff_cgb = y_scl - pred_cgb_scl
    ((pd.DataFrame(diff_cgb)).corr('pearson')).to_csv('corr_CGB.csv')
    ((pd.DataFrame(diff_gb)).corr('pearson')).to_csv('corr_GB.csv')

    axs2[0].scatter(diff_gb[:, 0], diff_gb[:, 1])
    axs2[1].scatter(diff_cgb[:, 0], diff_cgb[:, 1])
    for i in range(2):
        axs2[i].set_xlabel("target_0_Heating")
        axs2[i].set_ylabel("target_1_Cooling")
    axs2[0].set_title("GB")
    axs2[1].set_title("CGB")
    fig2.suptitle("Difference between real and predicted values")
    fig2.savefig('scatter_corr.eps')

    axs1[1][1].set_xlabel("Distance")
    axs1[1][1].set_title("Minimum Distance (Between targets)")
    axs1[1][1].legend()
    axs1[1][1].grid(True)

    fig1.suptitle("Comparing C-GB and GB")
    fig1.tight_layout()
    # fig1.savefig('regression_plt.eps')

    fig0.tight_layout()
    fig0.savefig('regression_plt_datapoints.eps')


    # Plot the hexbin
    labels = ["Heating", "Cooling"]
    for target, lb in enumerate(labels):
        plt.cla()

        axs3[target, 0].hexbin(y[:, target], pred_cgb[:, target], gridsize=15,
                               mincnt=1, edgecolors="none", cmap="viridis",
                               label=rmse(y, pred_cgb, target))
        axs3[target, 0].scatter(
            y[:, target], pred_cgb[:, target], s=2, c="white")
        axs3[target, 0].set_xlabel('real values')
        axs3[target, 0].text(0.95, 0.1, 'target: ' + lb,
                             verticalalignment='bottom',
                             horizontalalignment='right',
                             transform=axs3[target, 0].transAxes,
                             color='k',
                             )
        axs3[target, 0].legend(loc=2)

        plt.cla()

        axs3[target, 1].hexbin(y[:, target], pred_gb[:, target], gridsize=15,
                               mincnt=1, edgecolors="none", cmap="viridis",
                               label=rmse(y, pred_gb, target))
        axs3[target, 1].scatter(
            y[:, target], pred_gb[:, target], s=2, c="white")
        axs3[target, 1].set_xlabel('real values')
        axs3[target, 1].text(0.95, 0.1, 'target: ' + lb,
                             verticalalignment='bottom',
                             horizontalalignment='right',
                             transform=axs3[target, 1].transAxes,
                             color='k',
                             )
        axs3[target, 1].legend(loc=2)

    axs3[0, 0].set_ylabel('predicted values')
    axs3[1, 0].set_ylabel('predicted values')

    axs3[0, 0].set_title('C-GB')
    axs3[0, 1].set_title('GB')

    fig3.suptitle('The relationship between predicted and real data points')
    fig3.savefig('Hexbin.eps')
