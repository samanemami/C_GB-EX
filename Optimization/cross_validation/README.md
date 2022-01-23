
### Cross-validation only
We also defined a customized Cross-validation only for `GBDT-MO`, which consumes less memory.

<h5> Note that this part only considers the train and validation and does not have the final test training. </h5>

>  Please refer to the **<span style='color:red'> [bash](https://github.com/samanemami/C_GB-EX/tree/main/Optimization/bash_gbdtmo) optimization </span>** for tuning the `GBDT-MO` hyperparameters. 


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

This gridsearch is designed to work only for the wrapper of `GBDT-MO`, if you want to do the grid search without using the wrapper, please refer to the [optimize_gbdtmo](optimize_gbdtmo.py).
