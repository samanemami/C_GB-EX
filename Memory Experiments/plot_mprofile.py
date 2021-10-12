import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mprofile(path: 'path to mprofile_.da'):
  # To work with this file, you have to
  # consider the outputs of mprof run scripy.py as
  # an input of this method.

  # To plot your recorded memory *dat files,
  # define your path to record memory (*dat),
  # and it will plot all models.

  # Reading the files
  def files():
    name = []
    for dirname, _, filenames in os.walk(path):
      for file in filenames:
        name.append(file)
    return name

  # Extract the measured memory from each file
  def dataframe(item):
    data = pd.read_csv(os.path.join(path, files()[item]), sep=' ')
    data.drop(columns=['CMDLINE'], inplace=True)
    return data.iloc[:, 0]

  #To have a clean plot, we define an axis to plot a minimum
  # range of time for each model, as each model has
  # different complexity.
  Len = []
  for i, j in enumerate(files()):
    Len.append(dataframe(i).shape[0])
  axis = np.min(Len)

  plt.figure(figsize=(6, 5))
  for i, j in enumerate(files()):
    plt.subplot(2, 2, i+1)
    plt.plot(np.arange(start=0, stop=dataframe(i)[
        :axis].shape[0]/10, step=0.1), dataframe(i)[:axis], label=j[:4], linewidth=2)
    plt.title('Recorded memory - ' + j[:4])
    plt.ylabel('Memory used (in MiB)')
    plt.xlabel('time (in secons)')
    plt.legend()
    plt.grid(True)
  plt.tight_layout()
  plt.savefig('subplot.jpg', dpi=700)
  plt.close('all')

  plt.figure(figsize=(8, 6))
  for i, j in enumerate(files()):
    plt.plot(np.arange(start=0, stop=dataframe(i)[:axis].shape[0]/10, step=0.1), dataframe(
        i)[:axis], label=j[:4], linewidth=2)
    plt.title('Recorded memory')
    plt.ylabel('Memory used (in MiB)')
    plt.xlabel('time (in secons)')
    plt.legend()
    plt.grid(True)
  plt.tight_layout()
  plt.savefig('all.jpg', dpi=700)
  plt.close('all')
