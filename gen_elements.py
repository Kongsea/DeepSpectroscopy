import os
import csv

path = 'data/elements.txt'

with open(path) as f, open(path.replace('txt', 'csv'), 'wb') as fcsv:
  cw = csv.writer(fcsv)
  lines = f.readlines()
  header = lines[0]
  lines = lines[1:]
  header[0] = 'No'
  header.insert(1, 'Class')
  header.insert(2, 'SubClass')
  cw.writerow(header)
  for i, line in enumerate(lines):
    line = line.strip().split('\t')
    if len(line) < 10:
      break
    line[0] = i
    line.insert(1, )
    line.insert(2, )
    cw.writerow(line)
