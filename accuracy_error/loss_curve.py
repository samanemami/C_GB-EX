from sklearn.ensemble import GradientBoostingClassifier
from cgb import cgb_clf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.simplefilter('ignore')

def loss_curve(max_depth=2):

    # Defining the toy dataset
    waveform = np.loadtxt('waveform.data', delimiter=',')
    X = waveform[:, :-1]
    y = waveform[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    #  Training the proposed condensed method
    c_gb = cgb_clf(max_depth=max_depth,
                   subsample=0.75,
                   max_features="sqrt",
                   learning_rate=0.1,
                   random_state=1,
                   criterion="squared_error",
                   n_estimators=100)

    c_gb.fit(x_train, y_train)

    # training the Mart
    mart = GradientBoostingClassifier(max_depth=max_depth,
                                      subsample=0.75,
                                      max_features="sqrt",
                                      learning_rate=0.1,
                                      random_state=1,
                                      criterion="squared_error",
                                      n_estimators=100)
    mart.fit(x_train, y_train)

    # Plotting the training loss curve
    plt.plot(mart.train_score_, color='r',
             alpha=1, label='GB', linestyle='-')
    plt.plot(c_gb.train_score_, color='b', linestyle='--',
             alpha=0.4, label='C-GB')
    plt.title('Max depth=' + str(max_depth))
    plt.xlabel('Boosting iteration')
    plt.ylabel('Training loss')
    plt.grid(visible=True, alpha=0.3)


# Plotting the loss curve for different depth
if __name__ == '__main__':
    plt.figure(figsize=(10, 5))
    for i, j in enumerate([2, 5, 10, 20]):
        plt.subplot(2, 2, i+1)
        loss_curve(max_depth=j)
        print('*', end='')

    plt.legend(loc='upper center', bbox_to_anchor=(-.15, -0.13),
               fancybox=False, shadow=False, ncol=2)
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    plt.savefig('loss_curve.jpg', dpi=700)
    plt.savefig('loss_curve.eps',rasterized=True,dpi=700)
