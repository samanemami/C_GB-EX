# Tuning the hyperparameters



We defined an optimization class including cross-validation for each model and problem.


## C-GB, MART, TFBT
The general [optimization](optimization.py) method for the `Condensed Gradient Boosted`, `MART`, `TFBT`, and `GBDT-MO` is the same. To use tune the hyperparameters for these models, simply run the following method. Note that the `scoring` function is different for classification and regression. This method serves for these three models and binary, multi-class classification, regression, and multi-output regression.

```Python
gridsearch(X, y, model, grid,
               scoring_functions=None,
               pipeline=None,
               best_scoring=True,
               random_state=None,
               n_cv_general=10,
               n_cv_intrain=10,
               clf=False)
```
The `clf` defines the model that we considered to be optimized, is classification (`True`) or regression (`False`).

## GBDT-MO
Due to the high memory usage of the `GBDT-MO` model for some datasets, we defined a customized gridsearch only for this model, which consumes less memory than other methods such as gridsearchCV.
The [opt_gbdtmo](https://github.com/samanemami/C_GB-EX/blob/main/Optimization/opt_gbdtmo.py) method, manage the gridsearchCV for the GBDT-MO model with the following param_grid;
```Python
param_grid = {"max_depth": [2, 5, 10, 20],
              "learning_rate": [0.025, 0.05, 0.1, 0.5, 1],
              "subsample": [0.75, 0.5, 1]}
```
To tune the hyperparameters of the GBDT-MO, one can use this method as follows;
```Python
gridsearch(X, y, cv, random_state, path, param_grid)
```

## Scoring function
The scoring we used for different problems to find a better grid are as follows;
<ul>
  <li> multi-class classification: accuracy </li>
  <li> regression: RMSE</li>
  <li> Multioutput-Regression: r2, RMSE</li>
</ul>
