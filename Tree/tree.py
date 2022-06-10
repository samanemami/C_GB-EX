import warnings
import numpy as np
from cgb import cgb_clf
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from gbdtmo import GBDTMulti, load_lib, create_graph
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
                               n_samples=200,
                               flip_y=0.15)

def plot(tree, axs):

    return plot_tree(tree, filled=True, rounded=True, precision=2, ax=axs)


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

    fig, axs = plt.subplots(1, 1, figsize=(7, 2), facecolor='w')
    fig.subplots_adjust(hspace=0, wspace=0)

    plot(tree, axs=axs)
    plt.tight_layout()
    
    plt.savefig('CGB_Tree.jpg',  dpi=700)
    plt.savefig('CGB_Tree.eps')


def mart(max_depth=5, random_state=1):

    model = GradientBoostingClassifier(max_depth=max_depth,
                                       subsample=0.75,
                                       max_features="sqrt",
                                       learning_rate=0.1,
                                       random_state=random_state,
                                       criterion="squared_error",
                                       n_estimators=100)

    model.fit(X, y)
    fig, axs = plt.subplots(1, 2, figsize=(9, 2), facecolor='w')
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(n_classes-1):
        tree = model.estimators_[0][i]
        print("number of leaves_Tree1:", tree.tree_.n_leaves)

        plot(tree, axs[i])
        axs[i].set_title('class '+str(i))

    plt.tight_layout()
    plt.savefig('MART_Tree.jpg',  dpi=700)
    plt.savefig('MART_Tree.eps')


def gbdtmo(tree_id=0, max_depth=3, random_state=1):
  
    params = {"max_depth": max_depth, "lr": 0.1,
              'loss': b"ce", 'verbose': False, 'seed': random_state}

    X, y = np.ascontiguousarray(
        X, dtype=np.float64), y.astype(np.int32)

    path = '/home/user/.local/lib/python~/site-packages/gbdtmo/build/gbdtmo.so'  # Path to lib
    lib = load_lib(path)

    booster = GBDTMulti(lib, out_dim=len(np.unique(y)), params=params)
    booster.set_data((X, y))
    booster.train(100)

    booster.dump(b"dumpmodel.txt")
    graph = create_graph("dumpmodel.txt", tree_id, [
                         0, len(np.unique(y))-1])
    graph.render("GBDTMO_Tree", format='jpg')

    nodes = []
    dumped_model = "dumpmodel.txt"

    with open(dumped_model, "r") as f:
        tree = f.read().split("Booster")
        tree.pop(0)
        tree = tree[tree_id]
        tree = tree.split("\n")[1:]
        for line in tree:
            line = line.strip().split(",")
            if len(line) <= 1:
                continue
            node = int(line.pop(0))
            if node > 0:
                nodes.append(node)

    print("number of leaves:", len(nodes))


if __name__ == "__main__":
    c_gb(tree_id=0, max_depth=2, random_state=1)
    mart(max_depth=2, random_state=1)
    gbdtmo(tree_id=0, max_depth=2, random_state=1)
