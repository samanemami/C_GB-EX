![![memory](https://github.com/samanemami/C_GB-EX/blob/main/docs/letter-all.jpg)](https://github.com/samanemami/C_GB-EX/blob/main/docs/letter-all.jpg)

# Memory experiments

For memory consumptions, we considered different profilers and libraries to have robust and reliable results.
The methods used to calculate the consumed memory are as follows;
<ul>
  <li> tracemalloc </li>
  <li> memory_profiler </li>
  <li> psutil </li>



There is an example of each method in this directory.

We extracted the memory of the datasets for the experiments and only considered the train methods.

## tracemalloc

Regarding the tracemalloc experiments, for the `GBDT-MO` and `TFBT`, the `_model_complexity()`, defined in the related wrapper to return a tuple containing the time and consumed memory, which the second item shows the memory usage.

Also, the examples contain the plotting part.

## memory_profiler

After installing memory_profiler, you may run `python -m memory_profiler_example.py` in the python interpreter.
To install and plot a memory usage with the profiler, refer to this [library](https://pypi.org/project/memory-profiler/).
Also, if you run `mprof run memory_profiler_example.py` and then `mprof plot`, you will have a memory usage trend.
After running `mprof run memory_profiler_example.py`, it gives you a [dat](https://github.com/samanemami/C_GB-EX/blob/main/Memory%20Experiments/mprofile_20211005101227.dat) file regarding the consumed memory.
