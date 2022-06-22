import numpy as np
from cgb import cgb_clf
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from wrapper import classification
from sklearn.ensemble import GradientBoostingClassifier


def plotModel_MultiClass(X, y, clf, axs):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axs.contourf(xx, yy, Z, alpha=0.4)
    axs.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.gca().set_xlim(xx.min(), xx.max())
    plt.gca().set_ylim(yy.min(), yy.max())


X, y = dts.make_classification(n_features=2,
                               n_redundant=0,
                               n_informative=2,
                               random_state=2,
                               n_clusters_per_class=1,
                               n_classes=3,
                               n_samples=1200,
                               flip_y=0.15)

plt.scatter(X[:, 0], X[:, 1], c=y)


path = '/lustre/home/user/.local/lib/python~/site-packages/gbdtmo/build/gbdtmo.so'


cgb_ = cgb_clf(max_depth=5,
               subsample=0.75,
               max_features="sqrt",
               learning_rate=0.1,
               random_state=1,
               n_estimators=100)

cgb_.fit(X, y)


mart = GradientBoostingClassifier(max_depth=5,
                                  subsample=0.75,
                                  max_features="sqrt",
                                  learning_rate=0.1,
                                  random_state=1,
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


fig, axs = plt.subplots(1, 3, figsize=(15, 4), facecolor='w', edgecolor='k')
axs = axs.ravel()


plotModel_MultiClass(X, y, cgb_, axs=axs[0], title='C-GB')
axs[0].set_title('C-GB')
axs[0].grid(True)

plotModel_MultiClass(X, y, mart, axs=axs[1], title='GB')
axs[1].set_title('GB')
axs[1].grid(True)

plotModel_MultiClass(X, y, model_gbdtmo, axs=axs[2], title='GBDT-MO')
axs[2].set_title('GBDT-MO')
axs[2].grid(True)
plt.tight_layout()
