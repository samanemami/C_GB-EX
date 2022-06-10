import warnings
import numpy as np
import seaborn as sns
from cgb import cgb_reg
import scipy.stats as st
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


warnings.simplefilter('ignore')
random_state = 123
np.random.seed(random_state)

X, y = dts.make_regression(n_samples=1500,
                           n_targets=2,
                           random_state=random_state)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state)


pred_mart = np.zeros_like(y_test)
max_depth = 2


c_gb = cgb_reg(max_depth=max_depth,
               subsample=0.75,
               max_features="sqrt",
               learning_rate=0.1,
               random_state=1,
               criterion="squared_error",
               n_estimators=100)

c_gb.fit(x_train, y_train)
pred_cgb = c_gb.predict(x_test)
c_gb_score = r2_score(y_test, pred_cgb)

mart = GradientBoostingRegressor(max_depth=max_depth,
                                 subsample=0.75,
                                 max_features="sqrt",
                                 learning_rate=0.1,
                                 random_state=1,
                                 criterion="mse",
                                 n_estimators=100)
for i in range(y.shape[1]):
    mart.fit(x_train, y_train[:, i])
    pred_mart[:, i] = mart.predict(x_test)


mart_score = r2_score(y_test, pred_mart)


def three_d(x, y, xlabel, ylabel, title):
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10

    xmin = min(x) - deltaX
    xmax = max(x) + deltaX

    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1,
                           cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('predicted values')
    ax.set_ylabel('real values')
    ax.set_zlabel('PDF')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(60, 35)
    plt, plt.tight_layout()
    plt.show()
    plt.close('all')

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    w = ax.plot_wireframe(xx, yy, f)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('PDF')
    plt.tight_layout()
    ax.set_title(title)
    plt, plt.tight_layout()


def two_d(x, y, xlabel, ylabel, title):
    xy = np.vstack((x, y))
    kernel = st.gaussian_kde(xy)
    xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    concat = np.dstack((xx, yy))
    z = np.apply_along_axis(kernel, 2, concat)
    z = z.reshape(100, 100)
    plt.imshow(z, aspect=xx.ptp() / yy.ptp())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def hexbin(x, y, xlabel, ylabel, title):
    sns.set_theme(style="ticks")
    sns.jointplot(x=x, y=y, kind="hex", color="#4CB391")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()


if __name__ == '__main__':
    x = pred_cgb[:, 0]
    y = pred_mart[:, 0]
    two_d(x=x, y=y, xlabel='predicted C-GB',
          ylabel='pred MART', title='Target 0')
    three_d(x=x, y=y, xlabel='predicted C-GB',
            ylabel='pred MART', title='Target 0')
    hexbin(x=x, y=y, xlabel='predicted C-GB',
           ylabel='pred MART', title='Target 0')
