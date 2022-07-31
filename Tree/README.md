The [GB_tree](GB_tree.py) and [gbdtmo_tree](gbdtmo_tree.py) are designed to return a tree for the multi-class classification problem, you can add more classes by modifying the plots.

The [GB_tree](GB_tree.py), plot a tree for C-GB, GB, and [gbdtmo_tree](gbdtmo_tree.py) plot a tree for GBDT-MO models. 
The [GB_tree](GB_tree.py), also returns the number of leaves of each tree and each class and the values of the terminal nodes. Regarding the terminal values, the predicted values are plotted on a 2-D scheme to indicate the performance of the Decision Regressor Tree of each C-GB and GB model.

All the useful information about the base learner are added in the plots, including the number of leaves, Terminal regions, etc.

Moreover, the class has the ability to produce different Decision boundaries as well for the Classifier Ensemble, and Decision Tree Regressors for [1, 100] trees.
