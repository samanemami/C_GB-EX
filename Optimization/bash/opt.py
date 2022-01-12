import sklearn.datasets as dts
from gridsearch import gridsearchcv


X, y = dts.load_digits(return_X_y=True)


if __name__ == '__main__':
    gridsearchcv(X=X,
                 y=y,
                 num=100,
                 random_state=1,
                 loss=b"ce")
