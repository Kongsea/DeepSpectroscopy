import numpy as np
import matplotlib.pyplot as plt

from gen_bin import read_spa

PLOT = False

path = 'data/original.bin'
length = (10 + 1473) * 4
with open(path, 'rb') as f:
  f.seek(length * 0)
  tmp = f.read(length)
  label = np.frombuffer(tmp[:4], dtype=np.float32).astype(np.int64)[0]
  elements = np.frombuffer(tmp[4:4 * 10], dtype=np.float32)
  data = (np.frombuffer(tmp[4 * 10:], dtype=np.float32)).reshape(-1)
  x = np.arange(data.shape[-1])
  print(label)
  print(elements)
  if PLOT:
    plt.plot(x, data)
    plt.show()
  
spa_array = read_spa('data/1/1/data_001.spa')
x = np.arange(spa_array.shape[-1]
if PLOT:
  plt.plot(x, spa_array)
  plt.show()

print(np.all(np.round(data, 4) == np.round(spa_array.astype('float32'), 4)))
