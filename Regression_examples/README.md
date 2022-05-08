# Regression examples
The related examples for the multi-output regression problem are included here. 

## Scatter plot

The [target_wise](target_wise.py) illustrates the scatter plot regarding the predicted values of two C-GB and MART models against the real values. 
It shows the data distribution and model R2 score regarding the different depths for the decision tree regressors. 

Moreover, to have a clear idea about the relationship between the predicted and real values of each method and target, the hexbin plot within scatter plot has included.

# KDE

As an extra experiment, the related codes to reproduce the density analysis are included in the [kde](kde.py). The estimation of the probability density function (pdf) with the kernel function for the predicted and real values are included. This file produces the report of the relationship between two pair random variables, which could be the predicted values of each target or the real values. Due to the decision, you could select different variables for the method. 


![![Regression](https://github.com/samanemami/C_GB-EX/blob/main/docs/Scatter_regression.jpg)](https://github.com/samanemami/C_GB-EX/blob/main/docs/Scatter_regression.jpg)

