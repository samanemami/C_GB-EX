# C_GB-EX
Condensed Gradient Boosting Decision Tree - Examples

# About
This project has five main purposes:
<ul>
<li> Provides additional comparisons </li>
<li> Fixes the compared models' bugs </li>
<li> provides real examples to use C-GB </li>
<li> Provide the wrapper of different compared models </li>
  <li> provides codes for reproduction the paper experiments </li>
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

* [C-GB - Last version](https://github.com/samanemami/C-GB)
* [GradientBoostingClassifier - Last version](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
* [GradientBoostingRegressor - Last version](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
* [GBDT-MO - Version 0.0.1](https://arxiv.org/abs/1909.04373)
* [TFBT - Version 2.4.1](https://git.kot.tools/nk2/syntaxnet_rus/-/tree/caae66a144f1237eb6b5c19fa00c317ca3bed09c/tensorflow/tensorflow/contrib/boosted_trees)

> In the latest updates of the TensorFlow, the TFBT was marked as deprecated and replaced with the RF model. But you can find the base repository in the following [link](https://git.kot.tools/nk2/syntaxnet_rus/-/tree/caae66a144f1237eb6b5c19fa00c317ca3bed09c/tensorflow/tensorflow/contrib/boosted_trees).

# The included examples
There are related codes for different experiments. The experiments are as follows;
<ul>
  <li> Precision analysis </li>
  <li> Measuring the RMSE </li>
  <li> Measuring the accuracy </li>
  <li> Training the C-GB model </li>
  <li> Measuring the loss curve </li>
  <li> Training time calculations </li>
  <li> Measuring the usage of the memory </li>
  <li> Optimization class for all compared models </li>
  <li> Decision boundaries (For the ensembles and the base learner) </li>
  <li> Related scatter plots for regression (including different experiments) </li>
  <li> Drawing base classifiers (Decision Tree regressors) for different ensembles </li>
  
</ul>

## Sample of some of the experiments
In the following, two samples of the included experiments are revealed. Of course, the [Decision Boudry](https://github.com/samanemami/C_GB-EX/tree/main/Decision_boundary) includes more samples.
* Decision boundary example
* Regression example

### Visualization

### Decision boundary
In the following, you will find an example of decision boundary for three studied models, including our Condensed Gradient Boosting model.

![![classification](https://github.com/samanemami/C_GB-EX/blob/main/docs/example.jpg)](https://github.com/samanemami/C_GB-EX/blob/main/docs/example.jpg)


### Regression example
Here, the C-GB model is trained at the same time for a multi-output regression problem with two outputs in one training procedure. As the plots show, the model works perfectly for all of the outputs.

![![regression](https://raw.githubusercontent.com/samanemami/C_GB-EX/main/docs/Scatter_regression.jpg?token=GHSAT0AAAAAABSTP7JH6T4V5OI5VVXWKND6YTNC2UQ)](https://github.com/samanemami/C_GB-EX/blob/main/docs/Scatter_regression.jpg)



# Requirements
To run the related experiments of the paper, the following libraries would be required.  These libraries are only for related experiments. For the C-GB model, you do not need to install any library as it handles the dependencies.
 
<ul>
  <li> Numba </li>
  <li> Numpy </li>
  <li> SciPy </li>
  <li> Pandas </li>
  <li> ctypes </li>
  <li> Matplotlib </li>
  <li> tracemalloc </li>
  <li> memory_profiler </li>
</ul>
