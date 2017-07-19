import os
import csv
import time
import numpy as np
from tqdm import tqdm

from utils import readCSV


def shuffle_bin(spec_name, spec_shuffle_name, shuffle_seed, spec_length, bit_length=4):
  length = (10 + spec_length) * bit_length
  with open(spec_name, 'rb') as f, open(spec_shuffle_name, 'wb') as fw:
    buf = ''
    for i, ss in enumerate(tqdm(shuffle_seed)):
      f.seek(ss * length)
      buf += f.read(length)
      if (i + 1) % 300 == 0:
        fw.write(buf)
        buf = ''
    else:
      fw.write(buf)


def shuffle_csv(csv_lines, writename, shuffle_seed):
  with open(writename, 'wb') as fw:
    cw = csv.writer(fw)
    cw.writerow(csv_lines[0])
    for i in shuffle_seed:
      cw.writerow(csv_lines[i + 1])


def main():
  csvName = 'data/original.csv'
  csv_lines = readCSV(csvName)
  spec_num = len(csv_lines) - 1
  shuffle_seed_name = 'data/shuffle_seed.npy'
  shuffle_seed = np.random.permutation(spec_num)
  with open(shuffle_seed_name, 'wb') as frs:
    np.save(frs, shuffle_seed)

  shuffle_csv(csv_lines, 'data/shuffle.csv', shuffle_seed)

  spec_length = 1473
  spec_name = 'data/original.bin'
  spec_shuffle_name = 'data/shuffle.bin'
  shuffle_bin(spec_name, spec_shuffle_name, shuffle_seed, spec_length=spec_length)


if __name__ == '__main__':
  main()
