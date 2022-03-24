import TFBT
import warnings
import Scikit_CGB
import Gridsearch as grid
import sklearn.datasets as dt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

warnings.simplefilter("ignore")

X, y = dt.load_iris(return_X_y=True)

random_state = 1
cv_intrain = 2


cgb = Scikit_CGB.C_GradientBoostingClassifier(max_depth=10,
                                              subsample=0.75,
                                              max_features=1,
                                              learning_rate=0.25,
                                              random_state=random_state,
                                              criterion="mse",
                                              loss="deviance",
                                              n_estimators=100)


mart = GradientBoostingClassifier(max_depth=10,
                                  subsample=0.75,
                                  max_features=1,
                                  learning_rate=0.1,
                                  random_state=random_state,
                                  criterion="mse")


tfbt = TFBT.BoostedTreesClassifier(label_vocabulary=None,
                                   n_batches_per_layer=1,
                                   n_trees=100,
                                   max_depth=2,
                                   learning_rate=0.1,
                                   max_steps=None,
                                   steps=1000,
                                   model_dir=None)


param_grid = {"clf__max_depth": [-1, 2, 5, 10, 20],
              "clf__learning_rate": [0.025, 0.05, 0.1, 0.5, 1],
              "clf__max_features": ["sqrt", None],
              "clf__subsample": [0.75, 0.5, 1]}

param_grid_tfbt = {"clf__max_depth": [2, 5, 10],
                   "clf__learning_rate": [0.025, 0.05, 0.1, 0.5, 1]}

if __name__ == "__main__":

    grid.gridsearch(X=X,
                    y=y,
                    model=cgb,
                    grid=param_grid,
                    scoring_functions='accuracy',
                    pipeline=('scaler', StandardScaler()),
                    best_scoring=True,
                    random_state=random_state,
                    n_cvntrain=cv_intrain,
                    title='iris_CGB_(stratified_sampling)_')

    grid.gridsearch(X=X,
                    y=y,
                    model=mart,
                    grid=param_grid,
                    scoring_functions='accuracy',
                    pipeline=('scaler', StandardScaler()),
                    best_scoring=True,
                    random_state=random_state,
                    n_cvntrain=cv_intrain,
                    title='iris_MART_(stratified_sampling)_')

    grid.gridsearch(X=X,
                    y=y,
                    model=tfbt,
                    grid=param_grid_tfbt,
                    scoring_functions=None,
                    pipeline=('scaler', StandardScaler()),
                    best_scoring=True,
                    random_state=random_state,
                    n_cvntrain=cv_intrain,
                    title='iris_TFBT_(stratified_sampling)_')
