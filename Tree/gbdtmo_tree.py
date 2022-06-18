import warnings
import numpy as np
import sklearn.datasets as dts
from sklearn.model_selection import train_test_split
from gbdtmo import GBDTMulti, load_lib, create_graph

warnings.simplefilter("ignore")
np.random.seed(1)

X, y = dts.make_classification(n_features=2,
                               n_redundant=0,
                               n_informative=2,
                               random_state=2,
                               n_clusters_per_class=1,
                               n_classes=2,
                               n_samples=500,
                               flip_y=0.15)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

def gbdtmo(tree_id=0, max_depth=3, random_state=1):
    params = {"max_depth": max_depth, "lr": 0.1,
              'loss': b"ce", 'verbose': False, 'seed': random_state}

    X, y = np.ascontiguousarray(
        x_train, dtype=np.float64), y_train.astype(np.int32)

    path = '/lustre/home/samanema/.local/lib/python3.6/site-packages/gbdtmo/build/gbdtmo.so'
    lib = load_lib(path)

    booster = GBDTMulti(lib, out_dim=len(np.unique(y_train)), params=params)
    booster.set_data((X, y))
    booster.train(100)

    booster.dump(b"dumpmodel.txt")
    graph = create_graph("dumpmodel.txt", tree_id, [
                         0, len(np.unique(y_train))-1])
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
    gbdtmo(tree_id=0, max_depth=3, random_state=1)
