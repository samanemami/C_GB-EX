import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

""" ProgressBar """


def ProgressBar(percent, barLen=20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()


""" Cross-validation method """


def gridsearch(X, y, model, grid,
               scoring_functions=None,
               pipeline=None,
               best_scoring=True,
               random_state=None,
               n_cv_general=10,
               n_cv_intrain=10,
               verbose=False,
               clf=True,
               title=None):

    cv_results_test = np.zeros((n_cv_general, 1))
    cv_results_generalization = np.zeros((n_cv_general, 1))
    best_index_time = np.zeros((n_cv_general, 2))
    pred = np.zeros_like(y)
    bestparams = []
    cv_results = []
    if not clf:
        try:
            if y.shape[1] > 1:
                err = np.zeros((n_cv_general, y.shape[1]))
        except:
            pass

    if clf:
        kfold_gen = StratifiedKFold(n_splits=n_cv_general,
                                    shuffle=True,
                                    random_state=random_state)
    else:
        kfold_gen = KFold(n_splits=n_cv_general,
                          random_state=random_state,
                          shuffle=True)

    # k Fold cross-validation

    for cv_i, (train_index, test_index) in enumerate(kfold_gen.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if clf:
            kfold = StratifiedKFold(n_splits=n_cv_intrain,
                                    shuffle=True,
                                    random_state=random_state)
        else:
            kfold = KFold(n_splits=n_cv_intrain,
                          random_state=random_state,
                          shuffle=True)

        estimator = model if pipeline is None else Pipeline(
            [pipeline, ('clf', model)])

        # Finding optimum hyper-parameter

        grid_search = GridSearchCV(estimator, grid,
                                   cv=kfold,
                                   scoring=scoring_functions,
                                   refit=best_scoring,
                                   return_train_score=False,
                                   )

        grid_search.fit(x_train, y_train)

        if clf is False:
            try:
                if y.shape[1] > 1:
                    pred[test_index] = grid_search.predict(x_test)
                    err[cv_i, :] = np.sqrt(np.average(
                        (y_test - pred[test_index])**2, axis=0))
            except:
                pass

        bestparams.append(grid_search.best_params_)

        grid_search.cv_results_[
            'final_test_error'] = grid_search.score(x_test, y_test)

        cv_results.append(grid_search.cv_results_)

        cv_results_test[cv_i, 0] = grid_search.cv_results_[
            'mean_test_score'][grid_search.best_index_]
        cv_results_generalization[cv_i, 0] = grid_search.cv_results_[
            'final_test_error']
        best_index_time[cv_i, 0] = grid_search.cv_results_[
            'mean_fit_time'][grid_search.best_index_]
        best_index_time[cv_i, 1] = grid_search.cv_results_[
            'mean_score_time'][grid_search.best_index_]

        if verbose:
            ProgressBar(cv_i/abs((n_cv_general)-1), barLen=n_cv_general)

    results = {}
    results['Metric'] = [
        'Score' if scoring_functions is None else scoring_functions]
    results['Mean_test_score'] = np.mean(
        cv_results_test, axis=0)
    results['Std_test_score'] = np.std(
        cv_results_test, axis=0)
    results['Mean_generalization_score'] = np.mean(
        cv_results_generalization, axis=0)
    results['Std_generalization_score'] = np.std(
        cv_results_generalization, axis=0)
    results['mean_fit_time'] = np.mean(
        best_index_time[:, 0], axis=0)
    results['std_fit_time'] = np.std(
        best_index_time[:, 0], axis=0)
    results['mean_score_time'] = np.mean(
        best_index_time[:, 1], axis=0)
    results['std_scoret_time'] = np.std(
        best_index_time[:, 1], axis=0)

    pd.DataFrame(results).to_csv(title + '- Summary.csv')
    pd.DataFrame(cv_results).to_csv(title + '- CV_results.csv')
    pd.DataFrame(bestparams).to_csv(title + '- Best_Parameters.csv')
    pd.DataFrame(best_index_time, columns=["Fit_time", "Score_time"]).to_csv(
        title + '- Best_Index_time.csv')

    try:
        rmse = {}
        rmse['mean-RMSE'] = np.mean(err, axis=0)
        rmse['std-RMSE'] = np.std(err, axis=0)
        pd.DataFrame(rmse).to_csv(title + '- RMSE.csv')
    except:
        print('\n', 'The search has finished successfully')