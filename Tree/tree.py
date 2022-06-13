import warnings
import numpy as np
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from cgb import cgb_clf
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
              precision=2, ax=axs, label='root')


def terminal_leaves(model, tree, class_):
    # Return the terminal regions values
    if model.estimators_.shape[1]>2:
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
    if model.estimators_.shape[1]>2: 
        terminal_leave = terminal_leave.reshape(-1, 1)[:, 0]
    else:
        terminal_leave = terminal_leave[:, :, 0]
    return terminal_leave


def c_gb(tree_id=0, max_depth=5, random_state=1):

    model = cgb_clf(max_depth=max_depth,
                    subsample=0.75,
                    max_features="sqrt",
                    learning_rate=0.1,
                    random_state=random_state,
                    n_estimators=100,
                    criterion='squared_error')
    model.fit(X, y)
    tree = model.estimators_.reshape(-1)[tree_id]
    print("number of leaves:", tree.tree_.n_leaves)

    fig1, axs1 = plt.subplots(1, 1, figsize=(10, 3), facecolor='w')
    fig2, axs2 = plt.subplots(1, 3, figsize=(10, 3), facecolor='w')
    fig1.subplots_adjust(hspace=0, wspace=0)

    plot(tree, axs=axs1)
    plt.tight_layout()


    terminal_leave = terminal_leaves(model, 0, 0)
    for i in range(n_classes):
        axs2[i].plot(terminal_leave[:, i])

    fig2.savefig('leaves_CGB.jpg',  dpi=700)
    fig1.savefig('C_GB_Tree.jpg',  dpi=700)
    fig1.savefig('C_GB_Tree.eps')


def GB(max_depth=5, random_state=1):

    model = GradientBoostingClassifier(max_depth=max_depth,
                                       subsample=0.75,
                                       max_features="sqrt",
                                       learning_rate=0.1,
                                       random_state=random_state,
                                       criterion="squared_error",
                                       n_estimators=100)

    model.fit(X, y)
    fig1, axs1 = plt.subplots(1, 2, figsize=(30, 7), facecolor='w')
    fig2, axs2 = plt.subplots(1, 3, figsize=(30, 7), facecolor='w')
    fig1.subplots_adjust(hspace=0, wspace=0)
    leaves = []
    for i in range(n_classes-1):
        tree = model.estimators_[0][i]
        print("number of leaves_Tree1:", tree.tree_.n_leaves)

        plot(tree, axs1[i])

        # Find terminal region's values
        #  (For the first tree and 3 classes)
        axs2[i].plot(terminal_leaves(model, 0, i))

    fig1.tight_layout()
    fig1.savefig('GB_Tree.jpg',  dpi=700)
    fig1.savefig('GB_Tree.eps')
    plt.close('all')

    tree = model.estimators_[0][2]
    print("number of leaves_Tree1:", tree.tree_.n_leaves)

    plot_tree(tree)

    axs2[2].plot(terminal_leaves(model, 0, 2))

    plt.tight_layout()
    plt.savefig('GB_Tree2.jpg',  dpi=700)

    fig2.savefig('leaves,jpg')

if __name__ == '__main__':

    c_gb()
    GB()
