# C_GB-EX
Condensed Gradient Boosting Decision Tree - Examples

# About this rep
This project has two purposes:
<ul>
<li> provides real examples to use C-GB </li>
<li> provides codes for reproduction the paper experiments </li>
</ul>

Moreover, in this project, one may find additional experiments which they are not in the paper.

# Usage
First, the C-GB package should be installed. 
To use TFBT and GBDTMO, first, you have to install the related packages and then use their wrapper.
To install the C-GB, please refer to C-GB.
To download the related wrappers, use the following links;


* [GBDTM-O-wrapper](https://github.com/samanemami/GBDTMO/blob/master/gbdtmo/wrapper.py)
* [TFBT-wrapper](https://github.com/samanemami/TFBoostedTree)


# Models used for comparison

* C-GB 
* [GBDTM-O](https://github.com/zzd1992/GBDTMO)
* [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
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

To make the comparison more user-friendly, we developed related wrappers with the access link to clone/install.

![![alt text](/home/oem/Desktop/C_GB-EX/Decision boundary/docs/example.jpg)](/home/oem/Desktop/C_GB-EX/Decision boundary/docs/example.jpg)


# Requirements
<ul>
  <li> Numpy </li>
  <li> Pandas </li>
  <li> Matplotlib </li>
  <li> ctypes </li>
  <li> SciPy </li>
</ul>
