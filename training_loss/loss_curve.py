# %%
import warnings
import numpy as np
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Scikit_CGB import C_GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

warnings.simplefilter('ignore')


def loss_curve(max_depth=2, n_classes=3):

    # Defining the toy dataset
    X, y = dts.make_classification(n_features=2,
                                   n_redundant=0,
                                   n_informative=2,
                                   random_state=2,
                                   n_clusters_per_class=1,
                                   n_classes=n_classes,
                                   n_samples=1040,
                                   flip_y=0.15)

    #  Training the proposed condensed method
    c_gb = C_GradientBoostingClassifier(max_depth=max_depth,
                                        subsample=0.75,
                                        max_features="sqrt",
                                        learning_rate=0.1,
                                        random_state=1,
                                        criterion="mse",
                                        n_estimators=100)

    c_gb.fit(X, y)

    # training the Mart
    mart = GradientBoostingClassifier(max_depth=max_depth,
                                      subsample=0.75,
                                      max_features="sqrt",
                                      learning_rate=0.1,
                                      random_state=1,
                                      criterion="mse",
                                      n_estimators=100)
    mart.fit(X, y)

    # Plotting the training loss curve
    plt.plot(mart.train_score_, color='r',
             alpha=1, label='MART', linestyle='--')
    plt.plot(c_gb.train_score_, color='b', alpha=0.4, label='C-GB')
    plt.title('Max depth=' + str(max_depth))
    plt.xlabel('Boosting iteration')
    plt.ylabel('Training error')
    plt.grid(visible=True, alpha=0.3)


# Plotting the loss curve for different depth
if __name__ == '__main__':
    plt.figure(figsize=(10, 5))
    for i, j in enumerate([2, 5, 10, 20]):
        plt.subplot(2, 2, i+1)
        loss_curve(max_depth=j)
    plt.legend(loc='upper center', bbox_to_anchor=(-.15, -0.13),
               fancybox=False, shadow=False, ncol=2)
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    # plt.tight_layout()
    plt.savefig('loss_curve.jpg', dpi=700)
    plt.savefig('loss_curve.eps')
