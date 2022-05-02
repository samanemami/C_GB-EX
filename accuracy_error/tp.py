from sklearn.ensemble import GradientBoostingClassifier
from Scikit_CGB import C_GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import warnings


# Dataset Entry
waveform = np.loadtxt('
   'waveform.data',
    delimiter=',')
X = waveform[:, :-1]
y = waveform[:, -1]
n_class = len(np.unique(y))

warnings.simplefilter('ignore')


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
    acc_c_gb: ndarray of shape (n_boosting_iterations, n_classes) 
                    the true positive of each class for n boosting iterations.
    acc_mart: ndarray of shape (n_boosting_iterations, n_classes) 
                    the true positive of each class for n boosting iterations.

    """

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
    pred = mart.predict(x_test)

    #  Training the proposed condensed method
    c_gb = C_GradientBoostingClassifier(max_depth=max_depth,
                                        subsample=0.75,
                                        max_features="sqrt",
                                        learning_rate=0.1,
                                        random_state=1,
                                        criterion="mse",
                                        n_estimators=T)
    c_gb.fit(x_train, y_train)
    pred = c_gb.predict(x_test)

    acc_cgb = np.zeros((T, n_class))
    acc_mart = np.zeros((T, n_class))

    for _, pred in enumerate(c_gb.staged_predict(x_test)):
        cf = confusion_matrix(y_test, pred)
        acc_cgb[_, :] = np.diag(cf)

    for _, pred in enumerate(mart.staged_predict(x_test)):
        cf = confusion_matrix(y_test, pred)
        acc_mart[_, :] = np.diag(cf)

    return acc_cgb, acc_mart


if __name__ == '__main__':
    for _, j in enumerate([2, 5, 10, 20]):
        fig, axes = plt.subplots(4, 3, figsize=(15, 3))
        plt.tight_layout()
        acc_cgb, acc_mart = model_training(X=X,
                                           y=y,
                                           max_depth=j,
                                           T=100)
        for i in range(n_class):
            plt.subplot(1, 3, i+1)
            plt.plot(acc_mart[:, i],
                     label='MART', color='r')
            plt.plot(acc_cgb[:, i],
                     label='C-GB', color='b')
            plt.grid(True, alpha=0.7)
            plt.xlabel('Boosting iteration')
            plt.ylabel('True positive')
            plt.yticks(np.arange(0.0, 0.9, step=0.05))
            plt.autoscale(enable=True, axis='both')
            # plt.suptitle('Max depth=' + str(j))
            plt.title('class ' + str(i))
            plt.tight_layout()
            plt.legend()
        plt.savefig('tp' + str(j)+'.eps')
