#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''Spectroscopy basic functions.

   Author: Haiyang Kong
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import csv
import SimpleITK as sitk
import cv2
from tensorflow.contrib.layers import batch_norm
from utils import readCSV


class FLAGS():
  NUM_EPOCHS = 100
  CHANNEL_NUMBER = 1
  LABEL_NUMBER = 9
  BATCH_SIZE = 32
  EVAL_BATCH_SIZE = 32
  SEED = 66478
  NUM_GPU = 1
  NUM_PREPROCESS_THREADS = 12
  NUM_LABEL = 9
  NUM_SPEC = 1473
  PIXEL_LENGTH = 4
  VIEW_PATH = 'quantitative/models'
  MODEL_PATH = None
  SAVE_MODEL = True

XAVIER_INIT = tf.contrib.layers.xavier_initializer(seed=FLAGS.SEED)


def accuracy(predictions, labels):
  """Return the accuracy based on dense predictions and sparse labels."""
  return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def init_bin_file(path):
  bin_file_name = [path]
  for f in bin_file_name:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  fqb = tf.train.string_input_producer(bin_file_name)
  record_bytes = (10 + FLAGS.NUM_SPEC) * FLAGS.PIXEL_LENGTH
  rb = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  return fqb, rb


def get_train_data(fqb, rb):
  key, value = rb.read(fqb)
  record_bytes = tf.decode_raw(value, tf.float32)
  label = tf.cast(tf.slice(record_bytes, [1], [FLAGS.NUM_LABEL]), tf.int64)
  image = tf.reshape(tf.slice(record_bytes, [10], [FLAGS.NUM_SPEC]),
                     shape=[FLAGS.NUM_SPEC, 1])
  min_queue_examples = FLAGS.BATCH_SIZE * 100
  labels, images = tf.train.batch(
      [label, image],
      batch_size=FLAGS.BATCH_SIZE,
      num_threads=FLAGS.NUM_PREPROCESS_THREADS,
      capacity=min_queue_examples + 3 * FLAGS.BATCH_SIZE) 
  labels = tf.reshape(labels, [-1, 9])
  return labels, images


def get_size():
  return len(readCSV('data/train.csv')[1:]),\
         len(readCSV('data/test.csv')[1:]),\
         len(readCSV('data/val.csv')[1:])
