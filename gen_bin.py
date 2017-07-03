import os
from os import walk
import csv
import numpy as np
import cPickle as pickle
import struct
import pandas

from utils import readCSV, normalize

path = 'data'
CUT_LEFT = 0.2963
CUT_RIGHT = 0.3069


def gen_path_dict(path):
  path_dict = {}
  for dirpath, dirnames, filenames in walk(path):
    dirnames.sort()
    filenames.sort()
    if len(filenames) >= 90:
      cls = int(dirpath.split('/')[-2])
      no = int(dirpath.split('/')[-1])
      path_dict[cls] = path_dict.get(cls, {})
      for fn in filenames:
        if fn.endswith('.spa'):
          path_dict[cls].setdefault(no, []).append(os.path.join(dirpath, fn))
  with open('path_dict.pkl', 'wb') as f:
    pickle.dump(path_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
def get_path_dict(path):
  with open(path, 'rb') as f:
    return pickle.load(f)
    
    
def read_spa(path):
  with open(path, 'r') as f:
    spa_array = np.loadtxt(f, skiprows=1)
  tmp = spa_array[spa_array[:, 0] >= CUT_LEFT]
  spa_intensity = tmp[tmp[:, 0] <= CUT_RIGHT][:, -1]
  return normalize(spa_intensity)
  
  
def gen_label_dict():
  folder_length = [6, 10, 5, 6]
  dct = {}
  for i in range(len(folder_length)):
    dct[str(i+1)] = [str(j) for j in range(1, folder_length[i] + 1)]
  label_dict = {}
  for i in range(1, len(folder_length)+1):
    label_dict[i] = label_dict.get(i, {})
    for j in dct[str(i)]:
      label_dict[i][int(j)] = i
  return label_dict
  
  
if __name__ == '__main__':
  folder_length = [6, 10, 5, 6]
  label_dict = gen_label_dict()
  path_dict = get_path_dict('path_dict.pkl')
  elements = [l[1:] for l in readCSV('data/elements.csv')[1:]]
  with open('original.bin', 'wb') as f, open('original.csv', 'wb') as fcsv:
    cw = csv.writer(fcsv)
    cw.writerow(['Class', 'SubNO', 'Category'] + 'Mn Si Ni Cr V Mo Ti Cu Fe'.split())
    for cls, subfolders in path_dict.items():
      for subno, filenames in subfolders.items():
        no = sum(folder_length[:cls - 1]) + subno
        label = label_dict[cls][subno]
        csv_list = [cls, subno, label] + elements[no - 1]
        for fn in filenames:
          cw.writerow(csv_list)
          spa_array = read_spa(fn)
          f.write(struct.pack('<f', label))
          spa_array.astype('float32').tofile(f)
