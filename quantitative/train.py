#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''Convolutional neural networks trained
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
from spectroscopy import accuracy
from spectroscopy import get_train_data, get_size
from spectroscopy import init_bin_file
from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer
from tensorflow.contrib.layers import l2_regularizer, apply_regularization
from tensorflow.contrib.layers import batch_norm
from utils import readCSV

FLAGS.NUM_EPOCHS = 100
FLAGS.LABEL_NUMBER = 9
XAVIER_INIT = xavier_initializer(seed=FLAGS.SEED)
# RELU_INIT = variance_scaling_initializer()

Wb = {
    'W1': tf.get_variable('W1', [7, FLAGS.CHANNEL_NUMBER, 32], tf.float32, XAVIER_INIT),
    'b1': tf.Variable(tf.zeros([32])),
    'W2': tf.get_variable('W2', [5, 32, 32], tf.float32, XAVIER_INIT),
    'b2': tf.Variable(tf.zeros([32])),
    'W3': tf.get_variable('W3', [3, 32, 64], tf.float32, XAVIER_INIT),
    'b3': tf.Variable(tf.zeros([64])),
    'W4': tf.get_variable('W4', [3, 64, 64], tf.float32, XAVIER_INIT),
    'b4': tf.Variable(tf.zeros([64])),
    'W5': tf.get_variable('W5', [3, 64, 128], tf.float32, XAVIER_INIT),
    'b5': tf.Variable(tf.zeros([128])),
    'W6': tf.get_variable('W6', [3, 128, 128], tf.float32, XAVIER_INIT),
    'b6': tf.Variable(tf.zeros([128])),
    'W7': tf.get_variable('W7', [3, 128, 256], tf.float32, XAVIER_INIT),
    'b7': tf.Variable(tf.zeros([256])),
    'W8': tf.get_variable('W8', [3, 256, 256], tf.float32, XAVIER_INIT),
    'b8': tf.Variable(tf.zeros([256])),
    'fcw1': tf.get_variable('fcw1', [5 * 256, 32], tf.float32, XAVIER_INIT),
    'fcb1': tf.Variable(tf.zeros([32])),
    'fcw2': tf.get_variable('fcw2', [32, FLAGS.LABEL_NUMBER], tf.float32, XAVIER_INIT),
    'fcb2': tf.Variable(tf.zeros([FLAGS.LABEL_NUMBER]))
}


def model(data, keep_prob, reuse=None):
  with tf.variable_scope('conv1', reuse=reuse) as scope:
    conv = tf.nn.conv1d(data, Wb['W1'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b1']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv2', reuse=reuse) as scope:
    conv = tf.nn.conv1d(pool, Wb['W2'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b2']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv3', reuse=reuse) as scope:
    conv = tf.nn.conv1d(pool, Wb['W3'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b3']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv4', reuse=reuse) as scope:
    conv = tf.nn.conv1d(pool, Wb['W4'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b4']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv5', reuse=reuse) as scope:
    conv = tf.nn.conv1d(pool, Wb['W5'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b5']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv6', reuse=reuse) as scope:
    conv = tf.nn.conv1d(pool, Wb['W6'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b6']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv7', reuse=reuse) as scope:
    conv = tf.nn.conv1d(pool, Wb['W7'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b7']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv8', reuse=reuse) as scope:
    conv = tf.nn.conv1d(pool, Wb['W8'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b8']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('reshape', reuse=reuse) as scope:
    reshape = tf.reshape(pool, [-1, np.prod(pool.get_shape().as_list()[1:])])
  with tf.variable_scope('fc1', reuse=reuse) as scope:
    hidden = tf.nn.relu(tf.matmul(reshape, Wb['fcw1']) + Wb['fcb1'])
  with tf.variable_scope('dropout', reuse=reuse) as scope:
    hidden = tf.nn.dropout(hidden, keep_prob, seed=FLAGS.SEED)
  with tf.variable_scope('fc2', reuse=reuse) as scope:
    out = tf.matmul(hidden, Wb['fcw2']) + Wb['fcb2']
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


def train():
  start_time_first = time.time()
  WORK_DIRECTORY = FLAGS.VIEW_PATH
  train_size, test_size, val_size = get_size()
  fqbt, rbt = init_bin_file('data/train.bin')
  fqbv, rbv = init_bin_file('data/val.bin')
  fqbe, rbe = init_bin_file('data/test.bin')

  data_node = tf.placeholder(tf.float32, shape=(None, FLAGS.NUM_SPEC, FLAGS.CHANNEL_NUMBER))
  labels_node = tf.placeholder(tf.float32, shape=(None, FLAGS.LABEL_NUMBER))
  keep_hidden = tf.placeholder(tf.float32)
  logits = model(data_node, keep_hidden)
  loss = tf.losses.mean_squared_error(labels=labels_node, predictions=logits)
  loss += apply_regularization(l2_regularizer(5e-4), tf.trainable_variables())

  batch = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(0.01, batch * FLAGS.BATCH_SIZE,
                                             train_size, 0.95, staircase=True)
  optimizer = tf.train.MomentumOptimizer(
      learning_rate, 0.9).minimize(loss, global_step=batch)
  eval_predictions = model(data_node, keep_hidden, reuse=True)

  train_label_node, train_data_node = get_train_data(fqbt, rbt)
  val_label_node, val_data_node = get_train_data(fqbv, rbv)
  test_label_node, test_data_node = get_train_data(fqbe, rbe)

  saver = tf.train.Saver(tf.global_variables())

  TRAIN_FREQUENCY = train_size // FLAGS.BATCH_SIZE * 2
  TEST_FREQUENCY = TRAIN_FREQUENCY
  VAL_FREQUENCY = TRAIN_FREQUENCY
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
          feed_dict = {data_node: train_data,
                       labels_node: train_label, keep_hidden: 0.5}
          _, l, lr, logit = sess.run(
              [optimizer, loss, learning_rate, logits], feed_dict=feed_dict)
          if step != 0 and step % TRAIN_FREQUENCY == 0:
            et = time.time() - start_time
            print('Step %d (epoch %.2f), %.1f ms' %
                  (step, float(step) * FLAGS.BATCH_SIZE / train_size, 1000 * et / TRAIN_FREQUENCY))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            print('Train accuracy: {:.3f}'.format(accuracy(logit, train_label)))
            start_time = time.time()
          if step != 0 and step % VAL_FREQUENCY == 0:
            val_label_total = np.zeros(
                (val_size // FLAGS.BATCH_SIZE * FLAGS.BATCH_SIZE, FLAGS.LABEL_NUMBER))
            prediction_total = np.zeros(
                (val_size // FLAGS.BATCH_SIZE * FLAGS.BATCH_SIZE, FLAGS.LABEL_NUMBER))
            for ti in xrange(val_size // FLAGS.BATCH_SIZE):
              offset = ti * FLAGS.BATCH_SIZE
              val_data, val_label = sess.run([val_data_node, val_label_node])
              predictions = eval_in_batches(
                  val_data, sess, eval_predictions, data_node, keep_hidden)
              prediction_total[offset:offset + FLAGS.BATCH_SIZE, :] = predictions
              val_label_total[offset:offset + FLAGS.BATCH_SIZE] = val_label
            acc = accuracy(prediction_total, val_label_total)
            print('Accuracy of validation: {:.3f}'.format(acc))
            start_time = time.time()
          if step != 0 and step % TEST_FREQUENCY == 0:
            test_label_total = np.zeros(
                (test_size // FLAGS.BATCH_SIZE * FLAGS.BATCH_SIZE, FLAGS.LABEL_NUMBER))
            prediction_total = np.zeros(
                (test_size // FLAGS.BATCH_SIZE * FLAGS.BATCH_SIZE, FLAGS.LABEL_NUMBER))
            for ti in xrange(test_size // FLAGS.BATCH_SIZE):
              offset = ti * FLAGS.BATCH_SIZE
              test_data, test_label = sess.run([test_data_node, test_label_node])
              predictions = eval_in_batches(
                  test_data, sess, eval_predictions, data_node, keep_hidden)
              prediction_total[offset:offset + FLAGS.BATCH_SIZE, :] = predictions
              test_label_total[offset:offset + FLAGS.BATCH_SIZE] = test_label
            acc = accuracy(prediction_total, test_label_total)
            print('Accuracy of test: {:3.f}'.format(acc))
            start_time = time.time()
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
  print('All training process costs {:.2f} seconds...'.format(time.time() - start_time_first))


if __name__ == '__main__':
  train()
