# Tuning the hyperparameters



We defined an optimization class including cross-validation for each model and problem.


## General (one gridsearch for all models)
The general [Optimization_universal](Optimization_universal.py) method for the `Condensed Gradient Boosted`, `MART`, `TFBT`, and `GBDT-MO` is the same. To use tune the hyperparameters for these models, simply run the following method. Note that the `scoring` function is different for classification, regression, and Multi-label classification. This method serves for these three models and binary, multi-class classification, regression, and multi-output regression.

```Python
gridsearch(X=X, y=y,
           model=model,
           grid=param_grid,
           scoring_functions='accuracy',
           pipeline=None,
           best_scoring=True,
           random_state=1,
           n_cv_general=2,
           n_cv_intrain=2,
           verbose=True,
           clf=True,
           title='Dataset')
```
The `clf` defines the model that we considered to be optimized, is classification (`True`) or regression (`False`).

Note that the `scoring_functions` for `Multi_output regressio`n and `Multi-label classification` must be `r2` for ranking purposes. The model will return the `RMSE` for each output

## GBDT-MO
Due to the high memory usage of the `GBDT-MO` model for some datasets, we defined a customized gridsearch only for this model, which consumes less memory than other methods such as gridsearchCV.
The [Optimize_gbdtmo_wrapper](Optimize_gbdtmo_wrapper.py) method, manage the gridsearchCV for the GBDT-MO model with the following param_grid;
```Python
param_grid = {"max_depth": [2, 5, 10, 20],
              "learning_rate": [0.025, 0.05, 0.1, 0.5, 1],
              "subsample": [0.75, 0.5, 1]}
```
To tune the hyperparameters of the GBDT-MO, one can use this method as follows;
```Python
gridsearch(X=X, y=y,
           cv=2
           random_state=1,
           path='path.so',
           param_grid=param_grid,
           verbose=True,
           verbose=True,
           clf=True)
```
The `clf` defines the model that we considered to be optimized, is classification (`True`) or regression (`False`).
Note this class ([Optimize_gbdtmo_wrapper](Optimize_gbdtmo_wrapper.py)) `GBDT-MO` wrapper with the mentioned grid. 

This gridsearch is designed to work only for the wrapper of `GBDT-MO`, if you want to do the grid search without using the wrapper, please refer to the [optimize_gbdtmo](optimize_gbdtmo).


## Scoring function
The scoring we used for different problems to find a better grid are as follows;
<ul>
  <li> multi-class classification: accuracy </li>
  <li> regression: RMSE</li>
  <li> Multioutput-Regression: r2, RMSE</li>
</ul>
