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


def terminal_leaves(model, tree, class_):
    # Return the terminal regions values
    if model.estimators_.shape[1] > 2:
        est = model.estimators_[tree][class_]
    else:
        est = model.estimators_[tree][0]
    children_left = est.tree_.children_left
    children_right = est.tree_.children_right
    n_nodes = est.tree_.node_count
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]

        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    terminal_leave = est.tree_.value[np.where(is_leaves == True)]
    if model.estimators_.shape[1] > 2:
        terminal_leave = terminal_leave.reshape(-1, 1)[:, 0]
    else:
        terminal_leave = terminal_leave[:, :, 0]
    return terminal_leave


def interpolating(data):
    y = [i for i in range(data.shape[0])]
    idx = range(len(y))
    x = np.linspace(min(idx), max(idx), 300)

    spl = make_interp_spline(idx, data, k=2)
    smooth = spl(x)

    return x, smooth


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

    for i in range(1, 3):
        exec(f'fig{i}.subplots_adjust(hspace=-0.5, wspace=-0.15)')

    plot(tree_cgb, axs=axs1)

    terminal_leave_CGB = terminal_leaves(cgb, 0, 0)

    for i in range(n_classes):

        tree_gb = gb.estimators_[0][i]
        j = i
        if j < 2:
            plot(tree_gb, axs2[j])
        j += 1

        new_x, smooth = interpolating(terminal_leave_CGB[:, i])

        axs3[i].plot(new_x, smooth, color='b',
                     label="C-GB", linestyle="--", linewidth=3)

        terminal_leave_GB = terminal_leaves(gb, 0, i)

        new_x, smooth = interpolating(terminal_leave_GB)

        axs3[i].plot(new_x, smooth, color='r',
                     label="GB")

        axs3[i].set_title("class_" + str(i))
        axs3[i].set_xlabel("Leave number")
        axs3[i].grid(True)

        axs3[i].text(0.02, 0.2, 'Number of leaves (GB)='+str(tree_gb.tree_.n_leaves),
                     verticalalignment='bottom',
                     horizontalalignment='left',
                     transform=axs3[i].transAxes,
                     color='r',
                     fontsize=10,
                     rotation=90
                     )

        axs3[i].text(.98, 0.2, 'Number of leaves (C-GB)='+str(tree_cgb.tree_.n_leaves),
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     transform=axs3[i].transAxes,
                     color='b',
                     fontsize=10,
                     rotation=90
                     )

    axs3[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs3[0].set_ylabel("Leave value")

    fig3.suptitle("Terminal leaves values")
    fig3.tight_layout()

if __name__ == "__main__":
    model()
