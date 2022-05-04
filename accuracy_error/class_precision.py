import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from Scikit_CGB import C_GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
sys.path.append(r'D:\Academic\Ph.D\Programming\Py\PhD Thesis\Scikit_CGB')


def model_training(X, y, max_depth, T, random_state):
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

    np.random.seed(random_state)
    n_class = len(np.unique(y))

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    # training the Mart
    mart = GradientBoostingClassifier(max_depth=max_depth,
                                      subsample=0.75,
                                      max_features="sqrt",
                                      learning_rate=0.1,
                                      random_state=random_state,
                                      criterion="mse",
                                      n_estimators=T
                                      )
    mart.fit(x_train, y_train)

    #  Training the proposed condensed method
    c_gb = C_GradientBoostingClassifier(max_depth=max_depth,
                                        subsample=0.75,
                                        max_features="sqrt",
                                        learning_rate=0.1,
                                        random_state=random_state,
                                        criterion="mse",
                                        n_estimators=T
                                        )
    c_gb.fit(x_train, y_train)

    precision_c_gb = np.zeros((T, n_class))
    precision_mart = np.zeros((T, n_class))
    target_names = ['class '+str(i) for i in range(n_class)]

    for _, pred in enumerate(c_gb.staged_predict(x_test)):
        report = classification_report(
            y_test, pred, output_dict=True, target_names=target_names)
        for i in range(n_class):
            precision_c_gb[_, i] = report['class ' + str(i)]['precision']

    for _, pred in enumerate(mart.staged_predict(x_test)):
        report = classification_report(
            y_test, pred, output_dict=True, target_names=target_names)
        for i in range(n_class):
            precision_mart[_, i] = report['class ' + str(i)]['precision']

    return precision_c_gb, precision_mart

# jet = plt.get_cmap('bwr')
# colors = iter(jet(np.linspace(0, 2, n_class+1)))
# color =next(colors)


if __name__ == '__main__':
    warnings.simplefilter('ignore')

    # Dataset Entry
    vehicle = np.loadtxt(
        'vehicle.data',
        delimiter=',')
    X = vehicle[:, :-1]
    y = vehicle[:, -1]

    # Number of class labels
    n_class = len(np.unique(y))

    # Number of base learners
    T = 100

    # define how much it trains the model
    n = 10

    # Train the models for different values of the maximum depth
    for _, j in enumerate([2, 5, 10, 20]):
        c_gb = np.zeros((100, n_class))
        mart = np.zeros((100, n_class))

        for i in range(n):
            precision_c_gb, precision_mart = model_training(X=X,
                                                            y=y,
                                                            max_depth=j,
                                                            T=T,
                                                            random_state=i)
            c_gb += precision_c_gb
            mart += precision_mart
        c_gb = c_gb/n
        mart = mart/n

        fig, axes = plt.subplots(4, n_class, figsize=(15, 3))
        plt.tight_layout()

        for i in range(n_class):
            plt.subplot(1, n_class, i+1)
            plt.plot(mart[:, i],
                     label='MART', color='r')
            plt.plot(c_gb[:, i],
                     label='C-GB', color='b')
            plt.grid(True, alpha=0.7)
            plt.xlabel('Boosting iteration')
            plt.ylabel('class Precision')
            plt.yticks(np.arange(0.0, 0.9, step=0.05))
            plt.autoscale(enable=True, axis='both')
            # plt.suptitle('Max depth=' + str(j))
            plt.title('class ' + str(i))
            plt.tight_layout()
            plt.legend()
        plt.savefig('precision' + str(j)+'.eps')
        print('*', end='')
