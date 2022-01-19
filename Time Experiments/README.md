# Training time experiments

For the training time, there is one example of calculating this metric. We calculated the time in seconds for the training method of each model. Instead of `time.time()`, we used `time.process_time()`, which returns the sum of the system and user CPU time of the current process and guarantees that the return would be correct even if the hardware has put the jobs on hold.

The hyperparameters for all of the models are the same with one random seed. As the` max_depth` has a significant effect on training speed, the experiments had done for different depths.

To split the dataset, we used the ShuffleSplit cross-validation method with 10-folds to return the train and test indexes.

For the `TFBT` and `GBDT-MO` models, the `time.process_time()` defined inside of their wrapper in the `_model_complexity` method, which returns a tuple of training time and consumed memory.


If you prefer not to use the wrapper for the `TFBT` of the `GBDT-MO` models, you may have the [`gbdtmo_time.py`](gbdtmo_time.py) file for each of them. This file only works for the GBDT-MO. Likewise, for the TFBT, you have to use the model without the wrapper.
