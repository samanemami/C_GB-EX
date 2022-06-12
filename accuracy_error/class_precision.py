from sklearn.ensemble import GradientBoostingClassifier
from cgb import cgb_clf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import warnings


warnings.simplefilter('ignore')


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
                                      criterion="squared_error",
                                      n_estimators=T
                                      )
    mart.fit(x_train, y_train)

    #  Training the proposed condensed method
    c_gb = cgb_clf(max_depth=max_depth,
                   subsample=0.75,
                   max_features="sqrt",
                   learning_rate=0.1,
                   random_state=random_state,
                   criterion="squared_error",
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


if __name__ == '__main__':

    # Dataset
    waveform = np.loadtxt(
        r'D:\Academic\Ph.D\Programming\Datasets\Classification\waveform.data',
        delimiter=',')
    X = waveform[:, :-1]
    y = waveform[:, -1]

    # Number of class labels
    n_class = len(np.unique(y))
    # define how much it trains the model
    n = 10

    fig, axes = plt.subplots(4, n_class, figsize=(20, 15))
    fig.subplots_adjust(hspace=.6, wspace=0)

    # Train the models for different values of the maximum depth
    yticks = np.arange(0.7, 0.9, step=0.05)
    for i, depth in enumerate([2, 5, 10, 20]):
        c_gb = np.zeros((100, n_class))
        mart = np.zeros((100, n_class))

        for seed in range(n):
            precision_c_gb, precision_mart = model_training(X=X,
                                                            y=y,
                                                            max_depth=depth,
                                                            T=100,
                                                            random_state=seed)
            c_gb += precision_c_gb
            mart += precision_mart

        c_gb = c_gb/n
        mart = mart/n

        for k in range(n_class):

            axes[i][k].plot(mart[:, k],
                            label='GB', color='r')
            axes[i][k].plot(c_gb[:, k],
                            label='C-GB', color='b')

            axes[i][k].set_xlabel('Boosting iteration')
            axes[i][k].set_title('class ' + str(k))
            axes[i][k].text(0.95, 0.1, 'Max Depth='+str(depth),
                            verticalalignment='bottom',
                            horizontalalignment='right',
                            transform=axes[i][k].transAxes,
                            color='k',
                            # fontsize=10
                            )
            axes[i][k].set_yticks(yticks, labels=[" " for _ in yticks])
            axes[i][k].grid(visible=True, axis='both')

        axes[i][0].set_yticks(
            yticks, labels=["{:.2f}".format(tick) for tick in yticks])
        axes[i][0].set_ylabel('Precision')

        print('*', end='')

plt.legend(loc='upper center', bbox_to_anchor=(-0.5, -0.27),
           fancybox=False, shadow=False, ncol=2)
plt.savefig('precision.jpg')
