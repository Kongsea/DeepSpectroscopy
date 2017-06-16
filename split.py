import os
import csv
import random
from tqdm import tqdm

from utils import readCSV


def split_train_test():
  bin_name = 'shuffle.bin'
  csv_name = 'shuffle_new.csv'
  test_no = ['3', '9', '19', '24']
  
  csv_train = 'train_tmp.csv'
  csv_test = 'test.csv'
  bin_train = 'train_tmp.bin'
  bin_test = 'test.bin'
  
  length = (1 + 57144) * 4
  csv_lines = readCSV(csv_name)
  csv_header = csv_lines[0]
  csv_lines = csv_lines[1:]
  
  with open(csv_train, 'wb') as fcsvtrain, open(bin_train, 'wb') as fbintrain,\
       open(csv_test, 'wb') as fcsvtest, open(bin_test, 'wb') as fbintest,\
       open(bin_name, 'rb') as fbin:
    trainwriter = csv.writer(fcsvtrain)
    testwriter = csv.writer(fcsvtest)
    trainwriter.writerow(csv_header)
    testwriter.writerow(csv_header)
    buf_train = buf_test = ''
    for i, line in enumerate(tqdm(csv_lines)):
      buf = fbin.read(length)
      if line[0] in test_no:
        testwriter.writerow(line)
        buf_test += buf
        if i%100 == 0 and i != 0:
          fbintest.write(buf_test)
          buf_test = ''
      else:
        trainwriter.writerow(line)
        buf_train += buf
        if i%100 == 0 and i!=0:
         fbintrain.write(buf_train)
         buf_train = ''
    else:
      fbintrain.write(buf_train)
      fbintest.write(buf_test)
      
      
def split_train_val():
  bin_name = 'train_tmp.bin'
  csv_name = 'train_tmp.csv'
  
  csv_train = 'train.csv'
  csv_val = 'val.csv'
  bin_train = 'train.bin'
  bin_val = 'val.bin'
  
  length = (1+57144)*4
  csv_lines = readCSV(csv_name)
  csv_header = csv_lines[0]
  csv_lines = csv_lines[1:]
  train_length = int(len(csv_lines)*0.9)
  
  with open(csv_train, 'wb') as fcsvtrain, open(bin_train, 'wb') as fbintrain,\
       open(csv_val, 'wb') as fcsvval, open(bin_val, 'wb') as fbinval,\
       open(bin_name, 'rb') as fbin:
    trainwriter = csv.writer(fcsvtrain)
    valwriter = csv.writer(fcsvval)
    trainwriter.writerow(csv_header)
    valwriter.writerow(csv_header)
    buf_train = buf_val = ''
    for i, line in enumerate(tqdm(csv_lines)):
      buf = fbin.read(length)
      if i < train_length:
        trainwriter.writerow(line)
        buf_train += buf
        if i%100 == 0 and i!=0:
          fbintrain.write(buf_train)
          buf_train = ''
      else:
        valwriter.writerow(line)
        buf_val += buf
        if i%100 == 0 and i!=0:
          fbinval.write(buf_val)
          buf_val = ''
    else:
      fbintrain.write(buf_train)
      fbinval.write(buf_val)
      
      
if __name__ == '__main__':
  split_train_test()
  split_train_val()
  os.system('rm train_tmp.bin train_tmp.csv')
