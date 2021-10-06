from sklearn.ensemble import GradientBoostingClassifier
from Scikit_CGB import C_GradientBoostingClassifier
from wrapper import regression, classification
from gbdtmo import GBDTMulti, load_lib
from TFBT import BoostedTreesClassifier
from sklearn import datasets as dts
import memory_profiler


X, y = dts.load_digits(return_X_y=True)
path = 'path to so lib'

@profile
def cgb():
  model = C_GradientBoostingClassifier(max_depth=10,
                                       subsample=1,
                                       max_features='sqrt',
                                       learning_rate=0.1,
                                       random_state=1,
                                       criterion="mse",
                                       loss="deviance",
                                       n_estimators=100)

  model.fit(X, y)

@profile
def gbdtmo():
    model = classification(max_depth=10,
                           learning_rate=0.1,
                           random_state=1,
                           num_boosters=100,
                           lib=path,
                           subsample=1.0,
                           verbose=False,
                           num_eval=0
                           )
    model.fit(X, y)

@profile
def mart():
  model = GradientBoostingClassifier(max_depth=10,
                                     subsample=1,
                                     max_features='sqrt',
                                     learning_rate=0.1,
                                     random_state=1,
                                     criterion="mse",
                                     loss="deviance",
                                     n_estimators=100)
  model.fit(X, y)

@profile
def tfbt():
  model = BoostedTreesClassifier(label_vocabulary=None,
                                 n_trees=i,
                                 max_depth=10,
                                 learning_rate=0.1,
                                 steps=100,
                                 model_dir=None
                                 )
  model.fit(X, y)


if __name__ == '__main__':
  cgb()
  gbdtmo()
  mart()
  tfbt()
