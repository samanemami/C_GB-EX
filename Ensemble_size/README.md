In this part, we are trying to estimate the size of the ensemble and have a comparison. The goal is to build a smaller ensemble with better performance and higher training speed.

One approach besides the memory consumption is to count the number of leaves in the ensemble. It is easy to see that a tree with fewer leaves is shorter and leed to a smaller ensemble.

By counting the leaves of each tree and summing them, we will have the total number of leaves in the ensemble. [ensemble_size](ensemble_size.py)
The proposed C_GB method, MART, and GBDT-MO can count the leaves of each tree. The example includes the digits dataset to calculate the entire leaves.

Note that various settings of the model led to the different decision trees. For this matter, we considered the same values for all models' hyper-parameters and ran the model for different tree depths.

[ensemble_size](ensemble_size.py) is just an example of a digits dataset. It extends for different datasets and methods as well.
