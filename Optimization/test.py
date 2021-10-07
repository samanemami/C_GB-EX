# %%
from itertools import product
import sklearn.datasets as dts
from sklearn.ensemble import GradientBoostingClassifier
from opt_gbdtmo import grid

X, y = dts.load_iris(return_X_y=True)
model = GradientBoostingClassifier()
param_grid = {'learning_rate': [0.1, 0.2, 3], 'max_depth': [2, 5, 10]}

# grid(estimator=GradientBoostingClassifier, X=X,
#      y=y, cv=2,
#      param_grid=param_grid,
#      random_state=1)


# %%
grid = [dict(zip(param_grid, v))
                for v in product(*param_grid.values())]
grid
# %%
import warnings
warnings.simplefilter('ignore')
for param in grid:
    model = GradientBoostingClassifier(param)
    print(model.get_params())
