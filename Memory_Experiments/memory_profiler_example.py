from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from wrapper import regression, classification
from gbdtmo import GBDTMulti, load_lib
from TFBT import BoostedTreesClassifier
from sklearn import datasets as dts
from memory_profiler import profile
from cgb import cgb_clf
import warnings


warnings.simplefilter("ignore")


X, y = dts.load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


path = 'path to so lib'

@profile
def cgb():
  model = cgb_clf(max_depth=10,
                subsample=1,
                max_features="sqrt",
                learning_rate=0.1,
                random_state=1,
                n_estimators=100,
                criterion='squared_error')

  model.fit(x_train, y_train)

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
    model.fit(x_train, y_train)

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
  model.fit(x_train, y_train)

@profile
def tfbt():
  model = BoostedTreesClassifier(label_vocabulary=None,
                                 n_trees=100,
                                 max_depth=10,
                                 learning_rate=0.1,
                                 steps=100,
                                 model_dir=None
                                 )
  model.fit(x_train, y_train)


if __name__ == '__main__':
  cgb()
  gbdtmo()
  mart()
  tfbt()
