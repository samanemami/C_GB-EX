import warnings
import numpy as np
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from Scikit_CGB import C_GradientBoostingClassifier
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
                               n_samples=500,
                               flip_y=0.15)


def cgb(tree_id=0, max_depth=3, random_state=1):
    model = C_GradientBoostingClassifier(max_depth=max_depth,
                                         subsample=0.75,
                                         max_features="sqrt",
                                         learning_rate=0.1,
                                         random_state=random_state,
                                         n_estimators=100)
    model.fit(X, y)
    tree = model.estimators_.reshape(-1)[tree_id]
    print("number of leaves:", tree.tree_.n_leaves)
    # Storing class labels for further applications
    class_names = []
    [class_names.append(str(i)) for i in model.classes_]
    plot_tree(tree)
    plt.savefig('C_GB_Tree.jpg', dpi=500)
    plt.close("all")


def mart(tree_id=0, max_depth=3, random_state=1):
    model = GradientBoostingClassifier(max_depth=max_depth,
                                       subsample=0.75,
                                       max_features="sqrt",
                                       learning_rate=0.1,
                                       random_state=random_state,
                                       criterion="mse",
                                       n_estimators=100)

    model.fit(X, y)
    # Plot decision tree regressor for each class
    for i in range(n_classes):
        tree = model.estimators_.reshape(-1)[tree_id + i]

        print("number of leaves_Tree1:", tree.tree_.n_leaves)
        plot_tree(tree)
        plt.savefig('MART_Tree' + str(i) + '.jpg', dpi=500)
        plt.close("all")


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
    cgb(tree_id=0, max_depth=3, random_state=1)
    mart(tree_id=0, max_depth=3, random_state=1)
    gbdtmo(tree_id=0, max_depth=3, random_state=1)
