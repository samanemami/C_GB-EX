import sklearn.datasets as dts
from gridsearch import gridsearchcv

X, y = dts.load_digits(return_X_y=True)

if __name__ == '__main__':
    gridsearchcv(X=X,
                 y=y,
                 num_train=20,
                 num_test=100,
                 loss=b"ce",
                 random_state=1,
                 verbose=True)
