from sklearn.ensemble import GradientBoostingClassifier
from scipy.special import logsumexp
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import sklearn.datasets as dts
from cgb import cgb_clf
import numpy as np
import warnings


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


def boundaries(X=np.array,
               y=np.array,
               model=np.array,
               tree=-1,
               class_=-1,
               regression=True,
               title='title',
               axs=plt.axes):

    level = 11
    cm = plt.cm.viridis

    n_classes = len(np.unique(y))
    n_estimator = model.get_params()['n_estimators']
    learning_rate = model.get_params()['learning_rate']

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    pred = np.zeros((xx.ravel().shape[0], n_classes))

    def pred_gb(n_classes, tree):
        pred_ = np.zeros_like(pred)
        for i in range(n_classes):
            tree_ = model.estimators_[tree][i]
            pred_[:, i] = (learning_rate *
                           tree_.predict(np.c_[xx.ravel(),
                                               yy.ravel()]))
        return pred_

    if model.estimators_.shape[1] > 2:
        for i in range(n_estimator):
            pred_ = pred_gb(n_classes=n_classes, tree=i)
            pred += pred_

            if i == tree:
                print('break')
                pred = pred_
                break

    else:
        for i in range(n_estimator):
            tree_ = model.estimators_[i][0]
            pred_ = (learning_rate *
                     tree_.predict(np.c_[xx.ravel(),
                                         yy.ravel()]))
            pred += pred_
            if i == tree:
                pred = pred_
                break
            else:
                continue

    if regression:
        # Return Decision boundary for all tree and predict (Regression)
        Z = pred[:, class_].reshape(xx.shape)
        axs.contourf(xx, yy, Z, level, alpha=1, cmap=cm)
        CS = axs.contourf(xx, yy, Z, level, cmap=cm, shrink=0.9)
    else:
        # Return Decision boundary for one tree and predict_proba (classification).
        proba = np.nan_to_num(
            np.exp(pred - (logsumexp(pred, axis=1)[:, np.newaxis])))
        Z = np.argmax(proba, axis=1)
        Z = Z.reshape(xx.shape)

        axs.contourf(xx, yy, Z, level, alpha=1, cmap=cm)
        CS = axs.contourf(xx, yy, Z, level, cmap=cm, shrink=0.9)
        axs.contour(xx, yy, Z, [0.0], linewidths=2, colors='k')
        axs.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k')

    axs.set_title(title)

    plt.gca().set_xlim(xx.min(), xx.max())
    plt.gca().set_ylim(yy.min(), yy.max())

    axs.grid(True)

    return CS


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
    fig3, axs3 = plt.subplots(1, 2, figsize=(10, 4), facecolor="w")
    fig4, axs4 = plt.subplots(2, 3, figsize=(20, 7), facecolor="w")
    fig5, axs5 = plt.subplots(2, 3, figsize=(20, 7), facecolor="w")

    for i in range(1, 3):
        exec(f'fig{i}.subplots_adjust(hspace=-0.5, wspace=-0.15)')

    plot(tree_cgb, axs=axs1)

    # Plot (Tree) two first class only
    for i in range(n_classes):
        tree_gb = gb.estimators_[0][i]
        j = i
        if j < 2:
            plot(tree_gb, axs2[j])
        j += 1

        # Boundaries for all the tress
        CS = boundaries(X=X, y=y, model=cgb, tree=-1, class_=i, regression=True,
                        title='C-GB-class:' + str(i), axs=axs4[0][i])
        fig4.colorbar(CS, ax=axs4[0][i])
        CS = boundaries(X=X, y=y, model=gb, tree=-1, class_=i, regression=True,
                        title='GB-class:' + str(i), axs=axs4[1][i])
        fig4.colorbar(CS, ax=axs4[1][i])

        # Boundaries for one tree
        CS = boundaries(X=X, y=y, model=cgb, tree=0, class_=i, regression=True,
                        title='C-GB-class:' + str(i), axs=axs5[0][i])
        fig5.colorbar(CS, ax=axs5[0][i])
        CS = boundaries(X=X, y=y, model=gb, tree=0, class_=i, regression=True,
                        title='GB-class:' + str(i), axs=axs5[1][i])
        fig5.colorbar(CS, ax=axs5[1][i])

    # Plot Decision Boundaries
    for i, m in enumerate([cgb, gb]):
        CS = boundaries(X=X, y=y, model=m, tree=0, class_=-1, regression=False,
                        title='C-GB' if i == 0 else 'GB', axs=axs3[i])
        fig3.colorbar(CS, ax=axs3[i])

    fig3.suptitle(
        "Decision Boundaries for the first Decision Tree Regressor (Predicted values)")
    fig4.suptitle(
        "Decision Boundaries of each class (Terminal values of regression trees)")
    fig5.suptitle(
        "Decision Boundaries of each class (Terminal values of first regression trees)")
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()


if __name__ == "__main__":
    model()
