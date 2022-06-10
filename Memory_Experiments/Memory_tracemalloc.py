import numpy as np
import tracemalloc
from cgb import cgb_clf
from gbdtmo import load_lib
import sklearn.datasets as dts
from time import process_time
import matplotlib.pyplot as plt
from TFBT import BoostedTreesClassifier
from IPython.display import clear_output
from wrapper import regression, classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


random_seed = 1
n_class = 10
path = '/home/oem/.local/lib/python3.8/site-packages/gbdtmo/build/gbdtmo.so'
LIB = load_lib(path)

X, y = dts.make_classification(
    n_samples=100, n_classes=3, n_clusters_per_class=2,
    random_state=random_seed, n_informative=4)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_seed)

T = 100

gbdtmo_complexity = np.zeros((T, 2))
tf_complexity = np.zeros((T, 2))
MemCGB = np.zeros((T,))
MemMart = np.zeros((T,))
time_cgb = np.zeros((T))
time_mart = np.zeros((T))

for i in range(1, T+1):
    model = classification(max_depth=5,
                           learning_rate=0.1,
                           random_state=1,
                           num_boosters=i,
                           lib=path,
                           subsample=1.0,
                           verbose=False,
                           num_eval=0
                           )
    model.fit(x_train, y_train)
    gbdtmo_complexity[i-1, 0] = model._model_complexity()[0]
    gbdtmo_complexity[i-1, 1] = model._model_complexity()[1]

    tracemalloc.start()
    tracemalloc.clear_traces()
    cgb_ = cgb_clf(max_depth=max_depth,
                    subsample=1,
                    max_features="sqrt",
                    learning_rate=0.1,
                    random_state=1,
                    n_estimators=i,
                    criterion='squared_error')
    
    t0 = process_time()
    cgb_.fit(x_train, y_train)
    MemCGB[i-1] = (tracemalloc.get_traced_memory()[0])
    time_cgb[i-1] = process_time()-t0

    tracemalloc.start()
    tracemalloc.clear_traces()
    mart = GradientBoostingClassifier(max_depth=5,
                                      subsample=1,
                                      max_features='sqrt',
                                      learning_rate=0.1,
                                      random_state=1,
                                      criterion="mse",
                                      loss="deviance",
                                      n_estimators=i)
    t0 = process_time()
    mart.fit(x_train, y_train)
    MemMart[i-1] = tracemalloc.get_traced_memory()[0]
    time_mart[i-1] = process_time()-t0

    tf = BoostedTreesClassifier(label_vocabulary=None,
                                     n_trees=i,
                                     max_depth=5,
                                     learning_rate=0.1,
                                     steps=20,
                                     model_dir='/home/oem/Desktop/temp'
                                     )

    tf.fit(x_train, y_train)
    tf_complexity[i-1, 0] = tf._model_complexity()[0]
    tf_complexity[i-1, 1] = tf._model_complexity()[1]
    clear_output()


plt.plot(time_cgb, color='blue', label='cgb')
plt.plot(time_mart, color='r', label = 'mart')
plt.plot(gbdtmo_complexity[:, 0], color='black', label = 'gbdtmo')
plt.plot(tf_complexity[:, 0], color='green', label = 'tfbt')
plt.legend()
plt.title('Training time')
plt.xlabel('Boosting epochs')
plt.ylabel('Time in seconds')
plt.close('all')

plt.plot(MemCGB[1:], color='blue', label='cgb')
plt.plot(MemMart[1:], color='r', label = 'mart')
plt.plot(gbdtmo_complexity[1:, 1], color='black', label = 'gbdtmo')
plt.plot(tf_complexity[:, 1], color='green', label = 'tfbt')
plt.legend()
plt.title('Memory usage')
plt.xlabel('Boosting epochs')
plt.ylabel('Used memory in MB')
plt.close('all')


