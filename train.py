#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''3D convolutional neural network trained
   to analyze the spectral datasets.
   The spectral datasets are organized in the CIFAR architecture.

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
import matplotlib
from six.moves import xrange
import tensorflow as tf
from spectroscopy import FLAGS
from spectroscopy import error_rate, readCSV
from spectroscopy import get_size
from spectroscopy import init_bin_file, init_csv_file
from spectroscopy import get_train_data
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import l2_regularizer, apply_regularization
from tensorflow.contrib.losses import log_loss
from tensorflow.contrib.metrics import accuracy
from tensorflow.contrib.layers import batch_norm

FLAGS.NUM_EPOCHS = 50
FLAGS.LABEL_NUMBER = 2
XAVIER_INIT = tf.contrib.layers.xavier_initializer(seed=FLAGS.SEED)
RELU_INIT = variance_scaling_initializer()

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
    pool = tf.nn.max_pool3d(relu, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='VALID')
  with tf.variable_scope('conv2') as scope:
    conv = tf.nn.conv3d(pool, Wb['W2'], strides=[1, 1, 1, 1, 1], padding='SAME')
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
  with tf.variable_scope('reshape'):
    ps = relu.get_shape().as_list()
    reshape = tf.reshape(relu, [-1, ps[1] * ps[2] * ps[3] * ps[4]])
  with tf.variable_scope('fc1'):
    hidden = tf.nn.relu(tf.matmul(reshape, Wb['fcw1']) + Wb['fcb1'])
  with tf.variable_scope('dropout'):
    hidden = tf.nn.dropout(hidden, keep_prob, seed=FLAGS.SEED)
  with tf.variable_scope('fc2'):
    out = tf.matmul(hidden, Wb['fcw2']) + Wb['fcb2']
    out = tf.nn.sigmoid(out)
  return out


def eval_in_batches(data, sess, eval_prediction, eval_data, keep_hidden):
  size = data.shape[0]
  if size < FLAGS.EVAL_BATCH_SIZE:
    raise ValueError("batch size for evals larger than dataset: %d" % size)
  predictions = np.ndarray(shape=(size, FLAGS.LABEL_NUMBER), dtype=np.float32)
  for begin in xrange(0, size, FLAGS.EVAL_BATCH_SIZE):
    end = begin + FLAGS.EVAL_BATCH_SIZE
    if end <= size:
      predictions[begin:end, :] = sess.run(eval_prediction, feed_dict={
          eval_data: data[begin:end, ...], keep_hidden: 1})
    else:
      batch_predictions = sess.run(eval_prediction, feed_dict={
          eval_data: data[-FLAGS.EVAL_BATCH_SIZE:, ...], keep_hidden: 1})
      predictions[begin:, :] = batch_predictions[begin - size:, :]
  return predictions


def lunaTrain(train_size, test_size):
  sssstttt = time.time()
  WORK_DIRECTORY = os.path.join(FLAGS.VIEW_PATH, 'models')
  st = time.time()
  fqbt, rbt = init_bin_file()

  length = 4 + 40 * 40 * 40 * 4
  imgBuf = labelBuf = ''
  with open('/home/kong/4T/lung_cancer_merged41/test_shuffle.bin', 'rb') as ftest:
    buf = ftest.read(length)
    while buf:
      imgBuf += buf[4:]
      labelBuf += buf[:4]
      buf = ftest.read(length)
  test_data = (np.frombuffer(imgBuf, np.float32)).reshape((-1, 40, 40, 40, 1))
  test_label = np.frombuffer(labelBuf, np.float32).astype(np.int64)

  data_node = tf.placeholder(tf.float32, shape=(
      None, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.CHANNEL_NUMBER))
  labels_node = tf.placeholder(tf.int64, shape=(None, 2))
  keep_hidden = tf.placeholder(tf.float32)
  logits = model(data_node, keep_hidden)
  # tf.nn.log_softmax()
  # tf.nn.log_poisson_loss()
  loss = tf.abs(tf.reduce_mean(log_loss(labels_node, logits)))
  loss += apply_regularization(l2_regularizer(5e-4), tf.trainable_variables())

  batch = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(0.01, batch * FLAGS.BATCH_SIZE,
                                             train_size // 5, 0.95, staircase=True)
  optimizer = tf.train.MomentumOptimizer(
      learning_rate, 0.9).minimize(loss, global_step=batch)
  eval_predictions = tf.nn.softmax(logits)

  train_label_node, train_data_node = get_train_data(fqbt, rbt)

  saver = tf.train.Saver(tf.global_variables())

  TRAIN_FREQUENCY = train_size // FLAGS.BATCH_SIZE // 20
  TEST_FREQUENCY = train_size // FLAGS.BATCH_SIZE // 20
  SAVE_FREQUENCY = 10 * train_size // FLAGS.BATCH_SIZE

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(WORK_DIRECTORY, sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():
        start_time = time.time()
        for step in xrange(int(FLAGS.NUM_EPOCHS * train_size) // FLAGS.BATCH_SIZE):
          train_data, train_label = sess.run([train_data_node, train_label_node])
          train_label = np.where(train_label == [0], np.array([1, 0]), np.array([0, 1]))
          feed_dict = {data_node: train_data,
                       labels_node: train_label, keep_hidden: 0.5}
          _, l, lr = sess.run(
              [optimizer, loss, learning_rate], feed_dict=feed_dict)
          if step != 0 and step % TRAIN_FREQUENCY == 0:
            et = time.time() - start_time
            print('Step %d (epoch %.2f), %.1f ms' %
                  (step, float(step) * FLAGS.BATCH_SIZE / train_size, 1000 * et / TRAIN_FREQUENCY))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            start_time = time.time()
          if step != 0 and step % TEST_FREQUENCY == 0:
            st = time.time()
            test_label_total = np.zeros(
                (test_size // FLAGS.BATCH_SIZE * FLAGS.BATCH_SIZE))
            prediction_total = np.zeros(
                (test_size // FLAGS.BATCH_SIZE * FLAGS.BATCH_SIZE, 2))
            for ti in xrange(test_size // FLAGS.BATCH_SIZE):
              offset = ti * FLAGS.BATCH_SIZE
              batch_data = test_data[offset:(offset + FLAGS.BATCH_SIZE), ...]
              batch_labels = test_label[offset:(offset + FLAGS.BATCH_SIZE)]
              predictions = eval_in_batches(
                  batch_data, sess, eval_predictions, data_node, keep_hidden)
              prediction_total[offset:offset + FLAGS.BATCH_SIZE, :] = predictions
              test_label_total[offset:offset + FLAGS.BATCH_SIZE] = batch_labels
            test_error = error_rate(prediction_total, test_label_total)
            print('Test error: %.3f%%' % test_error)
            print('Test costs {:.2f} seconds.'.format(time.time() - st))
          if step % SAVE_FREQUENCY == 0 and step != 0:
            if FLAGS.SAVE_MODEL:
              checkpoint_path = os.path.join(WORK_DIRECTORY, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)
        else:
          if FLAGS.SAVE_MODEL:
            checkpoint_path = os.path.join(WORK_DIRECTORY, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
          coord.request_stop()
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      pass
    coord.join(threads)
  print('All costs {:.2f} seconds...'.format(time.time() - sssstttt))
  train_data = train_labels = 0


def main(_):
  train_size = 38344
  test_size = 1392
  lunaTrain(train_size, test_size)

if __name__ == '__main__':
  tf.app.run()
