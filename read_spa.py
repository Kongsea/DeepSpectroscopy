import sys
import os
import numpy as np
import matplotlib.pyplot as plt

spa_name = 'data_001.spa'

with open(spa_name, 'r') as f:
  spa_array = np.loadtxt(f, skiprows=1)

plt.plot(spa_array)
plt.show()
