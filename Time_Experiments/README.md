# Training time experiments

For the training time, you may use a [Training_time](Training_time.py) method.
The [Training_time](Training_time.py) also returns the training time for each classifier.

Instead of `time.time()`, we used `time.process_time()`, which returns the sum of the system and user CPU time of the current process and guarantees that the return would be correct even if the hardware has put the jobs on hold.

The values of the hyperparameters for all of the models are the same with one random seed. As the `max_depth` has a significant effect on training speed, the experiments had done for different depths. The examples here are only for depth 5.

# GB

We use Sklearn.GradientBoostingClassifier for the MART model. To have a fair comparison, we modify the prediction part for the [GB](GB.py) only.

# Wrappers

For the `TFBT` and `GBDT-MO` models, the `time.process_time()` defined inside of their wrapper in the `_model_complexity` method, which returns a tuple of training time and consumed memory.

If you prefer not to use the wrapper for the `TFBT` or the `GBDT-MO` models, you may use the [`gbdtmo_time.py`](gbdtmo_time.py) for each of them. This file only works for the `GBDT-MO`. Likewise, for the `TFBT`, you have to use the `TensorFlow Boosted Tree` without the wrapper.
