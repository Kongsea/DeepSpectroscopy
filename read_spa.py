import sys
import os
import numpy as np
import matplotlib.pyplot as plt

work_dir = os.getcwd()
spa_name = 'data_001.spa'

with open(os.path.join(work_dir, spa_name), 'r') as f:
  tmp = f.read().split('\n')[1:57145]
  spa_array = np.empty((len(tmp), 2))
  for i, t in enumerate(tmp):
    spa_array[i, :] = np.array(t.split(), float)

plt.plot(spa_array)
plt.show()
