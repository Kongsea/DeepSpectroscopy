#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''Test function using trained models.
   Author: Haiyang Kong
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
from os import walk
import numpy as np
import csv
import time
from scipy.ndimage.interpolation import zoom

from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from spectroscopy import FLAGS
from spectroscopy import readCSV, save3DSlice
from spectroscopy import normalizePlanes, worldToVoxelCoord
from spectroscopy import interpolatefilter, createImageBorder

FLAGS.USE_OFFICIAL = False
XAVIER_INIT = tf.contrib.layers.xavier_initializer(seed=FLAGS.SEED)

Wb = {
    'W1': tf.get_variable('W1', [3, 3, 3, FLAGS.CHANNEL_NUMBER, 16], tf.float32, XAVIER_INIT),
    'b1': tf.Variable(tf.zeros([16])),
    'W2': tf.get_variable('W2', [3, 3, 3, 16, 32], tf.float32, XAVIER_INIT),
    'b2': tf.Variable(tf.zeros([32])),
    'W3': tf.get_variable('W3', [3, 3, 3, 32, 64], tf.float32, XAVIER_INIT),
    'b3': tf.Variable(tf.zeros([64])),
    'W4': tf.get_variable('W4', [3, 3, 3, 64, 128], tf.float32, XAVIER_INIT),
    'b4': tf.Variable(tf.zeros([128])),
    'W5': tf.get_variable('W5', [3, 3, 3, 128, 256], tf.float32, XAVIER_INIT),
    'b5': tf.Variable(tf.zeros([256])),
    'fcw1': tf.get_variable('fcw1', [2**3 * 256, 32], tf.float32, XAVIER_INIT),
    'fcb1': tf.Variable(tf.zeros([32])),
    'fcw2': tf.get_variable('fcw2', [32, FLAGS.LABEL_NUMBER], tf.float32, XAVIER_INIT),
    'fcb2': tf.Variable(tf.zeros([FLAGS.LABEL_NUMBER]))
}


def model(data, keep_prob):
  with tf.variable_scope('conv1') as scope:
    conv = tf.nn.conv3d(data, Wb['W1'], strides=[1, 1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b1']))
  with tf.variable_scope('conv2') as scope:
    conv = tf.nn.conv3d(relu, Wb['W2'], strides=[1, 1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b2']))
    pool = tf.nn.max_pool3d(relu, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv3') as scope:
    conv = tf.nn.conv3d(pool, Wb['W3'], strides=[1, 1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b3']))
    pool = tf.nn.max_pool3d(relu, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv4') as scope:
    conv = tf.nn.conv3d(pool, Wb['W4'], strides=[1, 1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b4']))
    pool = tf.nn.max_pool3d(relu, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv5') as scope:
    conv = tf.nn.conv3d(pool, Wb['W5'], strides=[1, 1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b5']))
    pool = tf.nn.max_pool3d(relu, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('reshape'):
    ps = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [-1, ps[1] * ps[2] * ps[3] * ps[4]])
  with tf.variable_scope('fc1'):
    hidden = tf.nn.relu(tf.matmul(reshape, Wb['fcw1']) + Wb['fcb1'])
  with tf.variable_scope('dropout'):
    hidden = tf.nn.dropout(hidden, keep_prob, seed=FLAGS.SEED)
  with tf.variable_scope('fc2'):
    out = tf.matmul(hidden, Wb['fcw2']) + Wb['fcb2']
  return out


def readSpecialLine(csvLines, uid):
  coor = []
  oriAxis = []
  label = []
  for line in csvLines:
    if line[0] == uid:
      coor.append(np.array(line[-2:0:-1], np.float))
      oriAxis.append(line[1:-1])
      label.append(int(line[-1]))
  return coor, label, oriAxis


def readSpecialLine_own(csvLines, uid):
  coor = []
  oriAxis = []
  label = []
  for line in csvLines:
    if line[0] == uid:
      coor.append(np.array(line[3:0:-1], np.float))
      oriAxis.append(line[1:-1])
      label.append(1 if float(line[-1]) > 0.5 else 0)
  return coor, label, oriAxis


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) /
                  predictions.shape[0])


def LUNAtest(image, sess, test_prediction, test_data_node):
  def test_in_batches(data):
    size = data.shape[0]
    predictions = np.ndarray(shape=(size, FLAGS.LABEL_NUMBER), dtype=np.float32)
    for begin in xrange(0, size, FLAGS.EVAL_BATCH_SIZE):
      end = begin + FLAGS.EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            test_prediction,
            feed_dict={test_data_node: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            test_prediction,
            feed_dict={test_data_node: data[-FLAGS.EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  predictions = test_in_batches(image)
  return predictions


def test_official(mhdOriginalPath, uid, csvLines, count, sess, test_prediction, test_data_node):
  print('Processing No. {}...'.format(count))
  axis, label, oriAxis = readSpecialLine(csvLines, uid)
  interpolatedImage, outputsize, _, outputspacing, origin = interpolatefilter(
      mhdOriginalPath)
  BackImage = createImageBorder(interpolatedImage, outputsize)
  ccList = []
  for a in axis:
    cutCenter = worldToVoxelCoord(a, origin[::-1], outputspacing)
    cutCenter += (FLAGS.BACK_SIZE / 2 - np.array(outputsize[::-1]) / 2)
    ccList.append(cutCenter)
  ccList = np.round(ccList).astype(np.int)
  image = np.empty([len(ccList), FLAGS.SAVE_SIZE, FLAGS.SAVE_SIZE, FLAGS.SAVE_SIZE, 1])
  for index, cc in enumerate(ccList):
    cutTemp = BackImage[cc[0] - int(FLAGS.CUT_SIZE / 2):cc[0] + int(FLAGS.CUT_SIZE / 2),
                        cc[1] - int(FLAGS.CUT_SIZE / 2):cc[1] + int(FLAGS.CUT_SIZE / 2),
                        cc[2] - int(FLAGS.CUT_SIZE / 2):cc[2] + int(FLAGS.CUT_SIZE / 2)]
    # save3DSlice(BackImage, cc, 'test1/')
    cutTemp = zoom(cutTemp, float(FLAGS.SAVE_SIZE) / FLAGS.CUT_SIZE)
    image[index, ..., 0] = cutTemp
  results = LUNAtest(image, sess, test_prediction, test_data_node)
  return results, oriAxis, label


def test_own(mhdOriginalPath, uid, csvLines, count, sess, test_prediction, test_data_node):
  print('Processing No. {}...'.format(count))
  axis, label, oriAxis = readSpecialLine_own(csvLines, uid)
  interpolatedImage, outputsize, spacing, _, _ = interpolatefilter(mhdOriginalPath)
  BackImage = createImageBorder(interpolatedImage, outputsize)
  ccList = []
  for a in axis:
    cutCenter = np.array(a, np.float) * spacing[::-1] / spacing[0]
    cutCenter += (FLAGS.BACK_SIZE / 2 - np.array(outputsize[::-1]) / 2)
    ccList.append(cutCenter)
  ccList = np.round(ccList).astype(np.int)
  image = np.empty([len(ccList), FLAGS.SAVE_SIZE, FLAGS.SAVE_SIZE, FLAGS.SAVE_SIZE, 1])
  for index, cc in enumerate(ccList):
    cutTemp = BackImage[cc[0] - int(FLAGS.CUT_SIZE / 2):cc[0] + int(FLAGS.CUT_SIZE / 2),
                        cc[1] - int(FLAGS.CUT_SIZE / 2):cc[1] + int(FLAGS.CUT_SIZE / 2),
                        cc[2] - int(FLAGS.CUT_SIZE / 2):cc[2] + int(FLAGS.CUT_SIZE / 2)]
    # save3DSlice(BackImage, cc, 'test1/')
    cutTemp = zoom(cutTemp, float(FLAGS.SAVE_SIZE) / FLAGS.CUT_SIZE)
    image[index, ..., 0] = cutTemp
  results = LUNAtest(image, sess, test_prediction, test_data_node)
  return results, oriAxis, label


def genPath(originalPath):
  '''generate a dict with the uid as keys
     and the subset number as values.
  '''
  pathDict = {}
  for (dirpath, dirnames, filenames) in walk(originalPath):
    dirnames.sort()
    filenames.sort()
    if(len(filenames) < 20):
      continue
    else:
      for filename in filenames:
        if filename.endswith('.mhd'):
          pathDict.setdefault(filename[:-4], dirpath[-1])
  return pathDict


def main(_):
  filePath = '/home/kong/4T/nodule_project'
  pathDict = genPath(filePath)
  csvHeader = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
  csvName = '/home/kong/4T/official_extended/submissionCSV-model-012345678.csv'
  csvLines = readCSV(csvName)[1:]

  with open('results_submit_voxel.csv', 'wb') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(csvHeader)
    count = 0
    test_data_node = tf.placeholder(
        tf.float32, (None, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.CHANNEL_NUMBER))
    test_prediction = tf.nn.softmax(model(test_data_node, 1))
    saver = tf.train.Saver()
    for ssNo in range(10):
      modelPath = '/home/kong/4T/official_extended/Cross{}'.format(ssNo)
      with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(modelPath)
        ckpt.model_checkpoint_path = os.path.join(modelPath, ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
          print('Model of Subset {} Restored Successfully...'.format(ssNo))
        first = True
        for uid in pathDict:
          if str(ssNo) != pathDict[uid]:
            continue
          mhdFileName = '{}/subset{}/{}.mhd'.format(filePath, ssNo, uid)
          if FLAGS.USE_OFFICIAL:
            result, axis, label = test_official(mhdFileName, uid, csvLines, count,
                                                sess, test_prediction, test_data_node)
          else:
            result, axis, label = test_own(mhdFileName, uid, csvLines, count,
                                           sess, test_prediction, test_data_node)
          if first:
            allResult = result
            allLabel = label
            first = False
          else:
            allResult = np.append(allResult, result, 0)
            allLabel = np.append(allLabel, label, 0)
          for i in range(result.shape[0]):
            writeContent = [uid]
            writeContent.extend(axis[i])
            writeContent.append(result[i, 1])
            csvwriter.writerow(writeContent)
          count += 1
        print('Error Rate of Model {} is: {}'.format(
            ssNo, error_rate(allResult, allLabel)))


if __name__ == '__main__':
  tf.app.run()
