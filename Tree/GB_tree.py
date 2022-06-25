import warnings
import numpy as np
from cgb import cgb_clf
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.interpolate import make_interp_spline
from sklearn.ensemble import GradientBoostingClassifier



warnings.simplefilter("ignore")
np.random.seed(1)
n_classes = 3

X, y = dts.make_classification(n_features=2,
                               n_redundant=0,
                               n_informative=2,
                               random_state=2,
                               n_clusters_per_class=1,
                               n_classes=n_classes,
                               n_samples=100,
                               flip_y=0.15)


def plot(tree, axs):

    plot_tree(tree, filled=True, rounded=True,
              precision=2, ax=axs, label="root")


def boundaries(X, model, tree, class_, axs):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    if model.estimators_.shape[1] > 2:
        tree_ = model.estimators_[tree][class_]
        Z = tree_.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

    else:
        tree_ = model.estimators_[tree][0]
        Z = tree_.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, class_].reshape(xx.shape)

    axs.contourf(xx, yy, Z, alpha=0.4)
    axs.set_title("class " + str(class_))
    plt.gca().set_xlim(xx.min(), xx.max())
    plt.gca().set_ylim(yy.min(), yy.max())
    axs.grid(True)


def model(random_state=1):

    cgb = cgb_clf(max_depth=5,
                  subsample=0.75,
                  max_features="sqrt",
                  learning_rate=0.1,
                  random_state=random_state,
                  n_estimators=100,
                  criterion="squared_error")
    cgb.fit(X, y)

    gb = GradientBoostingClassifier(max_depth=5,
                                    subsample=0.75,
                                    max_features="sqrt",
                                    learning_rate=0.1,
                                    random_state=random_state,
                                    criterion="squared_error",
                                    n_estimators=100)

    gb.fit(X, y)

    tree_cgb = cgb.estimators_.reshape(-1)[0]

    fig1, axs1 = plt.subplots(1, 1, figsize=(10, 3), facecolor="w")
    fig2, axs2 = plt.subplots(1, 2, figsize=(30, 7), facecolor="w")
    fig3, axs3 = plt.subplots(1, 3, figsize=(30, 7), facecolor="w")
    fig4, axs4 = plt.subplots(1, 3, figsize=(30, 7), facecolor="w")

    for i in range(1, 3):
        exec(f'fig{i}.subplots_adjust(hspace=-0.5, wspace=-0.15)')

    plot(tree_cgb, axs=axs1)

    for i in range(n_classes):

        tree_gb = gb.estimators_[0][i]
        boundaries(X=X, model=cgb, tree=0, class_=i, axs=axs3[i])
        boundaries(X=X, model=gb, tree=0, class_=i, axs=axs4[i])
        j = i
        if j < 2:
            plot(tree_gb, axs2[j])
        j += 1

    fig3.suptitle("CGB Boundaries")
    fig4.suptitle("GB Boundaries")

    fig1.savefig("C_GB_Tree.jpg", dpi=1000)
    fig2.savefig("GB_Tree.jpg", dpi=1000)
    fig3.savefig("CGB_boundaries.jpg", dpi=1000)
    fig4.savefig("GB_boundaries.jpg", dpi=1000)


if __name__ == "__main__":
    model()
