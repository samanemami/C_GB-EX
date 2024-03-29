# Tuning the hyperparameters

We defined an optimization class, including cross-validation and stratified sampling for each model and problem.

## Stratified sampling

The number of instances for all the experiments and models is the same, and the results are comparable. We considered the stratified approach to split the datasets and cv_intrain = 2 for in train epoch of finding the best hyperparameters. You can find the related illustrations [here](https://github.com/samanemami/C_GB-EX/tree/main/Optimization/stratified_sampling). For the C-GB, MART, and TFBT, the example provided in pure [python](https://github.com/samanemami/C_GB-EX/tree/main/Optimization/stratified_sampling/py), and for the GBDT-MO, you need to run the bash [file](https://github.com/samanemami/C_GB-EX/tree/main/Optimization/stratified_sampling/bash_gbdtmo) first.

### Optimization with bash

Due to the high memory usage of the `GBDT-MO` model we provided a script in bash to do the gridsearch out of the python. This approach consumes less memory than python.
For this matter, you have to run your python file by calling it with the bash script. To access this approach, please refer to the [bash](bash_gbdtmo) directory.
With this approach, we include the grid inside the bash file, and by calling the python file, it trains the model with new arguments and stores the validation score in a separate file. For the test part, it trains the best_estimator with the optimum arguments and returns the generalization score.

The grid that considered for this optimization is as follows;
As the subsample does not influence the mode, we skip it for the tunning.

```bash
for lr in 0.025 0.05 0.1 0.5 1
do
    for dp in 2 5 10 20
    do
    
    python3 opt.py $lr $dp train1
    
    done
done
```

## General (one gridsearch for all models)
The general [Optimization_universal](Optimization_universal.py) method for the `Condensed Gradient Boosted`, `MART`, `TFBT`, and `GBDT-MO` is the same. To use tune the hyperparameters for these models, simply run the following method. Note that the `scoring` function is different for classification, regression, and Multi-label classification. This method serves for these three models and binary, multi-class classification, regression, and multi-output regression.

```Python
gridsearch(X=X, y=y,
           model=model,
           grid=param_grid,
           scoring_functions='accuracy',
           pipeline=None,
           random_state=1,
           n_cv_general=2,
           n_cv_intrain=2,
           verbose=True,
           clf=True,
           metric=None,
           title='Method1')


'''Python
        scoring_functions: str, callable, list, tuple or dict, default=None
                            Strategy for ranking the splits of the cross-validated model

        best_scoring: bool , default=None
                            using the best-found parameters

        n_cv_general: int, default=10
                            n-folds of the cross-validation 

        n_cv_intrain: int, default=10
                            int, to specify the number of folds in a `(Stratified)KFold`

        verbose: bool, default=False
                            if verbose then retrun the progress of the search

        clf: bool, default=True
                            if verbose then, return the progress of the search

        metric: str, {'euclidean', 'rmse'} default=None
                            if clf is Flase:
                                       if metric is 'euclidean': score = euclidean distance
                                                       (Only works for multi-outputs regression)
                                       if metric is 'rmse': score = rmse
                            

'''
```

The `clf` defines the model that we considered to be optimized, is classification (`True`) or regression (`False`).

Note that the `scoring_functions` for `Multi_output regression` and `Multi-label classification` must be `r2` for ranking purposes. The model will return the `RMSE` for each output

The metric `euclidean` was added to consider the noisy and extreme value for multivariate regression problems.

Note the fact that the `metric` is different from the `scoring_functions` in this method.
This `scoring_functions` is used to rank the model performance of each split in the cross-validation of the gridsearch. In contrast, the metric returns the score of the model performance based on the ranked splits of the gridsearch. The `metric` applies the predicted values of the highest-ranked split for each fold of the cross-validation to state the wanted score (RMSE or Euclidean distance).

## Scoring function
The scoring we used for different problems to find a better grid are as follows;
<ul>
  <li> multi-class classification: accuracy </li>
  <li> regression: RMSE</li>
  <li> Multioutput-Regression: r2, RMSE</li>
</ul>

<hr>

* There are extra examples in this [directory](https://github.com/samanemami/C_GB-EX/tree/main/Optimization/cross_validation) as well.
