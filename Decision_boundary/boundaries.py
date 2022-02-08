import numpy as np
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from wrapper import classification
from sklearn.model_selection import train_test_split
from Scikit_CGB import C_GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier


def plotModel_MultiClass(X, y, clf, title=None):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)
    plt.grid(True)
    plt.gca().set_xlim(xx.min(), xx.max())
    plt.gca().set_ylim(yy.min(), yy.max())


X, y = dts.make_classification(n_features=2,
                               n_redundant=0,
                               n_informative=2,
                               random_state=2,
                               n_clusters_per_class=1,
                               n_classes=3,
                               n_samples=1300,
                               flip_y=0.15)

plt.scatter(X[:, 0], X[:, 1], c=y)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

path = '/home/user/.local/lib/python~/site-packages/gbdtmo/build/gbdtmo.so'
tfbt_path = "Path_to_remove_tf_logs"

c_gb = C_GradientBoostingClassifier(max_depth=20,
                                    subsample=0.75,
                                    max_features="sqrt",
                                    learning_rate=0.1,
                                    random_state=1,
                                    criterion="mse",
                                    n_estimators=100)

c_gb.fit(x_train, y_train)


mart = GradientBoostingClassifier(max_depth=20,
                                  subsample=0.75,
                                  max_features="sqrt",
                                  learning_rate=0.1,
                                  random_state=1,
                                  criterion="mse",
                                  n_estimators=100)

mart.fit(x_train, y_train)


model_gbdtmo = classification(max_depth=10,
                              learning_rate=0.1,
                              random_state=1,
                              num_boosters=100,
                              lib=path,
                              subsample=1.0,
                              verbose=False,
                              num_eval=0
                              )
model_gbdtmo.fit(x_train, y_train)




plt.figure(num=1, figsize=(20, 7))
plt.subplot(1, 3, 1)
plotModel_MultiClass(x_train, y_train, c_gb, title="C-GB")

plt.subplot(1, 3, 2)
plotModel_MultiClass(x_train, y_train, mart, title="MART")

plt.subplot(1, 3, 3)
plotModel_MultiClass(x_train, y_train, model_gbdtmo, title="GBDT-MO")
