# C_GB-EX
Condensed Gradient Boosting Decision Tree - Examples
![![Tree](https://github.com/samanemami/C_GB-EX/blob/main/docs/C_GB_Tree.jpg)](https://github.com/samanemami/C_GB-EX/blob/main/docs/C_GB_Tree.jpg)

# About
This project has four main purposes:
<ul>
<li> provides real examples to use C-GB </li>
<li> provides codes for reproduction the paper experiments </li>
<li> Provides additional comparisons </li>
<li> Fixes the compared models' bugs </li>
</ul>

Moreover, in this project, one may find additional experiments which they are not in the paper.

# Usage
First, the following packages should be installed. 

* [Condensed Gradient Boosting Decision Tree](https://github.com/samanemami/C-GB)
* [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
* [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
* [BoostedTreesEstimator](https://www.tensorflow.org/api_docs/python/tf/estimator/BoostedTreesEstimator)
* [Gradient Boosted Decision Tree for Multiple Outputs](https://github.com/zzd1992/GBDTMO)

For some of the experiments, it would be easier to use related wrappers. For this purpose, the wrappers have designed as the following;

* [GBDTM-O-wrapper](https://github.com/samanemami/GBDTMO/blob/master/gbdtmo/wrapper.py)
* [TFBT-wrapper](https://github.com/samanemami/TFBoostedTree)


# Models used for comparison

* [C-GB](https://github.com/samanemami/C-GB)
* [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
* [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
* [GBDT-MO](https://arxiv.org/abs/1909.04373)
* [TFBT](https://www.tensorflow.org/api_docs/python/tf/estimator/BoostedTreesClassifier)

# The included examples
There are related codes for different experiments. The experiments are as follows;
<ul>
  <li> Training the C-GB model </li>
  <li> Optimization class for all compared models </li>
  <li> Training time calculations </li>
  <li> Measuring the usage of the memory </li>
  <li> Measuring the accuracy </li>
  <li> Measuring the RMSE </li>
  <li> Measuring the loss curve </li>
</ul>


## Visualization

### Decision boundary
In the following, you will find an example of decision boundary for three studied models, including our Condensed Gradient Boosting model.

![![classification](https://github.com/samanemami/C_GB-EX/blob/main/docs/example.jpg)](https://github.com/samanemami/C_GB-EX/blob/main/docs/example.jpg)


### Regression example
Here, the C-GB model is trained at the same time for a multi-output regression problem with two outputs in one training procedure. As the plots show, the model works perfectly for all of the outputs.

![![regression](https://raw.githubusercontent.com/samanemami/C_GB-EX/main/docs/Scatter_regression.jpg?token=GHSAT0AAAAAABSTP7JH6T4V5OI5VVXWKND6YTNC2UQ)](https://github.com/samanemami/C_GB-EX/blob/main/docs/Scatter_regression.jpg)



# Requirements
<ul>
  <li> Numpy </li>
  <li> Pandas </li>
  <li> Matplotlib </li>
  <li> ctypes </li>
  <li> SciPy </li>
  <li> Numba </li>
</ul>
