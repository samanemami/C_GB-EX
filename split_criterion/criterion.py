# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets as dts
import numpy as np
from numba import jit
from multiprocessing import Pool
from functools import partial
import ctypes


@jit(forceobj=True)
def construct_bin_column(x: np.array, max_bins: int) -> np.array:
    x, cnt = np.unique(x, return_counts=True)
    sum_cnt = np.sum(cnt)
    if len(x) == 1:
        return np.array([], 'float64')
    elif len(x) == 2:
        bins = (x[0]*cnt[0] + x[1]*cnt[1]) / sum_cnt
        return np.array([bins], 'float64')
    elif len(x) <= max_bins:
        bins = np.zeros(len(x)-1, 'float64')
        for i in range(len(x)-1):
            bins[i] = (x[i] + x[i+1]) / 2.0
        return bins
    elif len(x) > max_bins:
        cnt = np.cumsum(cnt)
        t, p = 0, len(x) / float(max_bins)
        bins = np.zeros(max_bins-1, 'float64')
        for i in range(len(x)):
            if cnt[i] >= p:
                bins[t] = x[i]
                t += 1
                p = cnt[i] + (sum_cnt - cnt[i]) / float(max_bins-t)
            if t == max_bins-1:
                break
        return bins


def map_bin_column(x, bins):
    bins = np.insert(bins, 0, -np.inf)
    bins = np.insert(bins, len(bins), np.inf)

    return np.searchsorted(bins, x, side='left').astype('uint16') - 1


def _get_bins_maps(x_column: np.array, max_bins: int) -> tuple:
    bins = construct_bin_column(x_column, max_bins)
    maps = map_bin_column(x_column, bins)

    return (bins, maps)


def get_bins_maps(x: np.array, max_bins: int, threads: int = 1) -> (list, np.array):
    out = []
    if threads == 1:
        for i in range(x.shape[-1]):
            out.append(_get_bins_maps(x[:, i], max_bins))
    else:
        x = list(np.transpose(x))
        pool = Pool(threads)
        f = partial(_get_bins_maps, max_bins=max_bins)
        out = pool.map(f, x)
        pool.close()

    bins, maps = [], []
    while out:
        _bin, _map = out.pop(0)
        bins.append(_bin)
        maps.append(_map)
    return bins, np.stack(maps, axis=1)


if __name__ == '__main__':
    x = np.random.rand(10000, 10)
    bins, maps = get_bins_maps(x, 8, 2)
    bin = bins[0]
    print(bin)

# %%


class mse():
    def __init__(self,
                 sq_sum_total=0.0,
                 weighted_n_node_samples=0,
                 n_outputs=0,
                 sum_total=None):
       self.sq_sum_total = sq_sum_total
       self.weighted_n_node_samples = weighted_n_node_samples
       self.n_outputs = n_outputs
       self.sum_total = sum_total

    def impurity(self):
        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    def proxy_impurity_improvement(self):
        proxy_impurity_left = 0
        proxy_impurity_right = 0

        for k in range(self.n_outputs):
            proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
            proxy_impurity_right += self.sum_right[k] * self.sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    def children_impurity(self, impurity_left, impurity_right):

        sq_sum_left = 0.0
        w = 1.0

        for p in range(self.start, self.pos):
            i = self.samples[p]

            if self.sample_weight != None:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (self.sum_left[k] /
                                 self.weighted_n_left) ** 2.0
            impurity_right[0] -= (self.sum_right[k] /
                                  self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


#%%
X, y = dts.load_boston(return_X_y=True)
np.unique(y, return_counts=True)
bins, maps = get_bins_maps(X, 21, 2)
model = DecisionTreeRegressor()
model.fit(X, y)
model.score(X, y)
