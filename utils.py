import csv
import numpy as np


def readCSV(filename):
  return [line for line in csv.reader(open(filename, 'rb'))]
  
  
def normalize(data):
  _min = np.min(data)
  _max = np.max(data)
  return (data - _min) / (_max - _min)
