import numpy as np
import tracemalloc
from time import process_time
from gbdtmo import GBDTMulti, load_lib
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


class gbdt_mo(BaseEstimator):
    def __init__(self,
                 max_depth=5,
                 learning_rate=0.1,
                 random_state=1,
                 lib=None,
                 num_boosters=30,
                 subsample=1.0,
                 verbose=False,
                 num_eval=0,
                 max_bins=8,
                 min_samples=16,
                 gamma=1e-3
                 ):

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.lib = lib
        self.num_boosters = num_boosters
        self.subsample = subsample
        self.verbose = verbose
        self.num_eval = num_eval
        self.max_bins = max_bins
        self.min_samples = min_samples
        self.gamma = gamma

    def _model_complexity(self):
        return self.training_time, self.memory


class classification(gbdt_mo):

    def fit(self, X, y):

        if X.flags['C_CONTIGUOUS'] is False:
            X = np.ascontiguousarray(X, dtype=np.float64)

        X = X.astype('float64')
        y = y.astype('int32')
        n_class = len(np.unique(y))
        loss = b"ce"

        LIB = load_lib(self.lib)
        params = {"max_depth": self.max_depth,
                  "lr": self.learning_rate,
                  'loss': loss,
                  "seed": self.random_state,
                  "subsample": self.subsample,
                  "verbose": self.verbose,
                  "max_bins": self.max_bins,
                  "min_samples": self.min_samples,
                  "gamma": self.gamma}

        self.booster = GBDTMulti(LIB, out_dim=n_class, params=params)
        self.booster.set_data((X, y))

        tracemalloc.start()
        t0 = process_time()
        self.booster.train(self.num_boosters)
        self.training_time = process_time() - t0
        self.memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.clear_traces()
        return self

    def predict_proba(self, X):
        return self.booster.predict(X, self.num_eval)

    def predict(self, X):
        if X.flags['C_CONTIGUOUS'] is False:
            X = np.ascontiguousarray(X, dtype=np.float64)
        X = X.astype('float64')

        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        if X.flags['C_CONTIGUOUS'] is False:
            X = np.ascontiguousarray(X, dtype=np.float64)
        X = X.astype('float64')
        y = y.astype('int32')

        return accuracy_score(y, self.predict(X))


class regression(gbdt_mo):

    def fit(self, X, y):

        if X.flags['C_CONTIGUOUS'] is False:
            X = np.ascontiguousarray(X, dtype=np.float64)

        X = X.astype('float64')
        y = y.astype('float64')
        n_class = y.shape[1]
        loss = b"mse"

        LIB = load_lib(self.lib)
        params = {"max_depth": self.max_depth,
                  "lr": self.learning_rate,
                  'loss': loss,
                  "seed": self.random_state,
                  "subsample": self.subsample,
                  "verbose": self.verbose,
                  "max_bins": self.max_bins,
                  "min_samples": self.min_samples,
                  "gamma": self.gamma}

        self.booster = GBDTMulti(LIB, out_dim=n_class, params=params)
        self.booster.set_data((X, y))

        tracemalloc.start()
        t0 = process_time()
        self.booster.train(self.num_boosters)
        self.training_time = process_time() - t0
        self.memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.clear_traces()
        return self

    def predict(self, X):
        if X.flags['C_CONTIGUOUS'] is False:
            X = np.ascontiguousarray(X, dtype=np.float64)
        X = X.astype('float64')

        return self.booster.predict(X, self.num_eval)

    def score(self, X, y):
        if X.flags['C_CONTIGUOUS'] is False:
            X = np.ascontiguousarray(X, dtype=np.float64)
        X = X.astype('float64')
        y = y.astype('float64')

        return np.sqrt(np.average((y - self.predict(X)) ** 2, axis=0))
