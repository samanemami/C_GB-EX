import numpy as np
import sklearn.datasets as dt
from gbdtmo import GBDTMulti, load_lib
from Scikit_CGB import C_GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


X, y = dt.load_digits(return_X_y=True)
n_class = len(np.unique(y))

max_depth = 5
random_state = 1
path = '/home/user/.local/lib/python~/site-packages/gbdtmo/build/gbdtmo.so' # Path to the dynamic library of gbdtmo
lib = load_lib(path)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state)


cgb = C_GradientBoostingClassifier(max_depth=max_depth,
                                   subsample=1,
                                   max_features="sqrt",
                                   learning_rate=0.1,
                                   random_state=random_state,
                                   criterion="mse",
                                   loss="deviance",
                                   n_estimators=1)


cgb.fit(x_train, y_train)
print('C-GB Leaves')
print(sum(tree.tree_.n_leaves for tree in cgb.estimators_.reshape(-1)))


mart = GradientBoostingClassifier(max_depth=max_depth,
                                  subsample=1,
                                  max_features="sqrt",
                                  learning_rate=0.1,
                                  random_state=random_state,
                                  criterion="mse",
                                  n_estimators=1)

mart.fit(x_train, y_train)
print('MART Leaves')
print(sum(tree.tree_.n_leaves for tree in mart.estimators_.reshape(-1)))


params = {"max_depth": max_depth,
          "lr": 0.1,
          'loss': b"ce",
          'verbose': False,
          'seed': random_state}


x_train, y_train = np.ascontiguousarray(
    x_train, dtype=np.float64), y_train.astype(np.int32)
x_test, y_test = np.ascontiguousarray(
    x_test, dtype=np.float64), y_test.astype(np.int32)


booster = GBDTMulti(lib, out_dim=n_class, params=params)
booster.set_data((x_train, y_train), (x_test, y_test))
booster.train(100)
booster.dump(b"digits.txt")

tree_index = 100
nodes = []
dumped_model = "digits.txt"
gbdtmo_nodes = []
for i in range(tree_index):
    with open(dumped_model, "r") as f:
        tree = f.read().split("Booster")
        tree.pop(0)
        tree = tree[i]
        tree = tree.split("\n")[1:]
        for line in tree:
            line = line.strip().split(",")
            if len(line) <= 1:
                continue
            node = int(line.pop(0))
            if node > 0:
                nodes.append(node)
    gbdtmo_nodes.append(nodes)

print('GBDT-MO Leaves')
print(len(gbdtmo_nodes[tree_index-1]))
