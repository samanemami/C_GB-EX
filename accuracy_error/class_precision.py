from sklearn.ensemble import GradientBoostingClassifier
from Scikit_CGB import C_GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import warnings


# Dataset Entry
waveform = np.loadtxt('waveform.data', delimiter=',')
X = waveform[:, :-1]
y = waveform[:, -1]
n_class = len(np.unique(y))


max_depth = 5
T = 100


def model_training(X, y, max_depth, T):
    """ Training the C-GB and MART models.

    Trains two multi-class classification models.

    Parameters
    -------
    X : input. array-like of shape (n_samples, n_features)
    y : output. array-like of shape (n_samples,)
    max_depth : Maximum depth of decision tree regressor
    T : number of estimators

    Returns
    -------
    precision_c_gb: ndarray of shape (n_boosting_iterations, n_classes) 
                    the precision of each class for n boosting iterations.
    precision_mart: ndarray of shape (n_boosting_iterations, n_classes) 
                    the precision of each class for n boosting iterations.

    """

    warnings.simplefilter('ignore')

    n_class = len(np.unique(y))

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    # training the Mart
    mart = GradientBoostingClassifier(max_depth=max_depth,
                                      subsample=0.75,
                                      max_features="sqrt",
                                      learning_rate=0.1,
                                      random_state=1,
                                      criterion="mse",
                                      n_estimators=T)
    mart.fit(x_train, y_train)

    #  Training the proposed condensed method
    c_gb = C_GradientBoostingClassifier(max_depth=max_depth,
                                        subsample=0.75,
                                        max_features="sqrt",
                                        learning_rate=0.1,
                                        random_state=1,
                                        criterion="mse",
                                        n_estimators=T)
    c_gb.fit(x_train, y_train)

    precision_c_gb = np.zeros((T, n_class))
    precision_mart = np.zeros((T, n_class))

    for _, pred in enumerate(c_gb.staged_predict(x_test)):
        report = classification_report(y_test, pred, output_dict=True, target_names=[
            'class 0', 'class 1', 'class 2'])
        for i in range(n_class):
            precision_c_gb[_, i] = report['class ' + str(i)]['precision']

    for _, pred in enumerate(mart.staged_predict(x_test)):
        report = classification_report(y_test, pred, output_dict=True, target_names=[
            'class 0', 'class 1', 'class 2'])
        for i in range(n_class):
            precision_mart[_, i] = report['class ' + str(i)]['precision']

    return precision_c_gb, precision_mart

# jet = plt.get_cmap('bwr')
# colors = iter(jet(np.linspace(0, 2, n_class+1)))
# color =next(colors)



if __name__ == '__main__':
    for _, j in enumerate([2, 5, 10, 20]):
        fig, axes = plt.subplots(4, 3, figsize=(15, 3))
        plt.tight_layout()
        precision_c_gb, precision_mart = model_training(X=X,
                                                        y=y,
                                                        max_depth=j,
                                                        T=100)
        for i in range(n_class):
            plt.subplot(1, 3, i+1)
            plt.plot(precision_mart[:, i],
                     label='MART', color='r')
            plt.plot(precision_c_gb[:, i],
                     label='C-GB', color='b')
            plt.grid(True, alpha=0.7)
            plt.xlabel('Boosting iteration')
            plt.ylabel('class Precision')
            plt.yticks(np.arange(0.0, 0.9, step=0.05))
            plt.autoscale(enable=True, axis='both')
            plt.suptitle('Max depth=' + str(j))
            plt.title('class ' + str(i))
            plt.tight_layout()
            plt.legend()
        plt.savefig('depth' + str(j)+'.jpg', dpi=700)
