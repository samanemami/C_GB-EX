import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold


def gridsearch(X, y, model, grid,
               scoring_functions=None,
               pipeline=None,
               best_scoring=True,
               random_state=None,
               n_cvntrain=10,
               title='none'):

    bestparams = []
    cv_results = []

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, test_size=0.2)

    kfold = StratifiedKFold(n_splits=n_cvntrain,
                            shuffle=True,
                            random_state=random_state)

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

    bestparams.append(grid_search.best_params_)

    grid_search.cv_results_[
        'final_test_error'] = grid_search.score(x_test, y_test)

    cv_results.append(grid_search.cv_results_)

    results = {}
    results['Metric'] = [scoring_functions]
    results['test_score'] = grid_search.cv_results_[
        'mean_test_score'][grid_search.best_index_]
    results['Mean_generalization_score'] = grid_search.cv_results_[
        'final_test_error']
    results['fit_time'] = grid_search.cv_results_[
        'mean_fit_time'][grid_search.best_index_]
    results['score_time'] = grid_search.cv_results_[
        'mean_score_time'][grid_search.best_index_]

    pd.DataFrame(results).to_csv(title +'_Summary.csv')
    pd.DataFrame(cv_results).to_csv(title + '_CV_results.csv')
    pd.DataFrame(bestparams).to_csv(title + '_Best_Parameters.csv')
