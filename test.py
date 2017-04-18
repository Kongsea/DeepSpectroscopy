import os
import numpy as np

work_dir = '/home/kong/4T/DeepSpectroscopy'
spe_name = 'data_001.spe'
spa_name = 'data_001.spa'

with open(os.path.join(work_dir, spe_name), 'r') as f:
  for line in f:
    print(line)
