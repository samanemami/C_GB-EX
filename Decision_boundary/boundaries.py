import numpy as np
from cgb import cgb_clf
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from wrapper import classification
from sklearn.model_selection import train_test_split
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
                               n_samples=1040,
                               flip_y=0.15)

plt.scatter(X[:, 0], X[:, 1], c=y)


path = '/home/user/.local/lib/python~/site-packages/gbdtmo/build/gbdtmo.so'
tfbt_path = "Path_to_remove_tf_logs"


cgb_ = cgb_clf(max_depth=5,
               subsample=0.75,
               max_features="sqrt",
               learning_rate=0.1,
               random_state=1,
               n_estimators=100,
               criterion='squared_error')

cgb_.fit(X, y)


mart = GradientBoostingClassifier(max_depth=5,
                                  subsample=0.75,
                                  max_features="sqrt",
                                  learning_rate=0.1,
                                  random_state=1,
                                  criterion="squared_error",
                                  n_estimators=100)

mart.fit(X, y)


model_gbdtmo = classification(max_depth=5,
                              learning_rate=0.1,
                              random_state=1,
                              num_boosters=100,
                              lib=path,
                              subsample=0.75,
                              verbose=False,
                              num_eval=0)

model_gbdtmo.fit(X, y)


plt.figure(num=1, figsize=(20, 7))
plt.subplot(1, 3, 1)
plotModel_MultiClass(X, y, cgb_, title="C-GB")

plt.subplot(1, 3, 2)
plotModel_MultiClass(X, y, mart, title="MART")

plt.subplot(1, 3, 3)
plotModel_MultiClass(X, y, model_gbdtmo, title="GBDT-MO")
