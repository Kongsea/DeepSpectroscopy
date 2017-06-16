import numpy as np
import matplotlib.pyplot as plt

from gen_bin import read_spa
from spectroscopy import FLAGS

path = 'original.bin'
length = (FLAGS.NUM_LABEL + FLAGS.NUM_SPEC) * FLAGS.PIXEL_LENGTH
with open(path, 'rb') as f:
  f.seek(length * 0)
  tmp = f.read(length)
  label = np.frombuffer(tmp[:4], dtype=np.float32).astype(np.int64)[0]
  data = (np.frombuffer(tmp[4:], dtype=np.float32)).reshape(-1)
  x = np.arange(data.shape[-1])
  
  plt.plot(x, data)
  plt.show()
  
spa_array = read_spa('data/1/1/data_001.spa')
x = np.arange(spa_array.shape[-1]
plt.plot(x, spa_array)
plt.show()

print(np.all(np.round(data, 4) == np.round(spa_array.astype('float32'), 4)))
