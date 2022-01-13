import sklearn.datasets as dts
from gridsearch import gridsearchcv

X, y = dts.load_digits(return_X_y=True)

if __name__ == '__main__':
    gridsearchcv(X, y, 20, 100, b"ce", 1, False)
