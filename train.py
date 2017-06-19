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
from spectroscopy import get_data, get_size
from spectroscopy import init_bin_file
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import l2_regularizer, apply_regularization
from tensorflow.contrib.layers import batch_norm
from utils import readCSV

FLAGS.NUM_EPOCHS = 50
FLAGS.LABEL_NUMBER = 4
XAVIER_INIT = tf.contrib.layers.xavier_initializer(seed=FLAGS.SEED)
RELU_INIT = variance_scaling_initializer()

Wb = {
    'W1': tf.get_variable('W1', [128, FLAGS.CHANNEL_NUMBER, 16], tf.float32, XAVIER_INIT),
    'b1': tf.Variable(tf.zeros([16])),
    'W2': tf.get_variable('W2', [64, 16, 32], tf.float32, XAVIER_INIT),
    'b2': tf.Variable(tf.zeros([32])),
    'W3': tf.get_variable('W3', [32, 32, 64], tf.float32, XAVIER_INIT),
    'b3': tf.Variable(tf.zeros([64])),
    'W4': tf.get_variable('W4', [16, 64, 64], tf.float32, XAVIER_INIT),
    'b4': tf.Variable(tf.zeros([64])),
    'W5': tf.get_variable('W5', [8, 64, 128], tf.float32, XAVIER_INIT),
    'b5': tf.Variable(tf.zeros([128])),
    'W6': tf.get_variable('W6', [3, 128, 128], tf.float32, XAVIER_INIT),
    'b6': tf.Variable(tf.zeros([128])),
    'W7': tf.get_variable('W7', [3, 128, 256], tf.float32, XAVIER_INIT),
    'b7': tf.Variable(tf.zeros([256])),
    'W8': tf.get_variable('W8', [3, 256, 256], tf.float32, XAVIER_INIT),
    'b8': tf.Variable(tf.zeros([256])),
    'W9': tf.get_variable('W9', [3, 256, 256], tf.float32, XAVIER_INIT),
    'b9': tf.Variable(tf.zeros([256])),
    'W10': tf.get_variable('W10', [3, 256, 256], tf.float32, XAVIER_INIT),
    'b10': tf.Variable(tf.zeros([256])),   
    'fcw1': tf.get_variable('fcw1', [55 * 256, 128], tf.float32, XAVIER_INIT),
    'fcb1': tf.Variable(tf.zeros([128])),
    'fcw2': tf.get_variable('fcw2', [128, FLAGS.LABEL_NUMBER], tf.float32, XAVIER_INIT),
    'fcb2': tf.Variable(tf.zeros([FLAGS.LABEL_NUMBER]))
}


def model(data, keep_prob):
  with tf.variable_scope('conv1') as scope:
    conv = tf.nn.conv1d(data, Wb['W1'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b1']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv2') as scope:
    conv = tf.nn.conv1d(pool, Wb['W2'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b2']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv3') as scope:
    conv = tf.nn.conv1d(pool, Wb['W3'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b3']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv4') as scope:
    conv = tf.nn.conv1d(pool, Wb['W4'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b4']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')  
  with tf.variable_scope('conv5') as scope:
    conv = tf.nn.conv1d(pool, Wb['W5'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b5']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv6') as scope:
    conv = tf.nn.conv1d(pool, Wb['W6'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b6']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv7') as scope:
    conv = tf.nn.conv1d(pool, Wb['W7'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b7']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv8') as scope:
    conv = tf.nn.conv1d(pool, Wb['W8'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b8']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv9') as scope:
    conv = tf.nn.conv1d(pool, Wb['W9'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b9']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('conv10') as scope:
    conv = tf.nn.conv1d(pool, Wb['W10'], strides=1, padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, Wb['b10']))
    pool = tf.nn.max_pool1d(relu, pool_size=2,
                            strides=(2), padding='VALID')
  with tf.variable_scope('reshape'):
    reshape = tf.reshape(pool, [-1, np.prod(pool.get_shape().as_list()[1:])])
  with tf.variable_scope('fc1'):
    hidden = tf.nn.relu(tf.matmul(reshape, Wb['fcw1']) + Wb['fcb1'])
  with tf.variable_scope('dropout'):
    hidden = tf.nn.dropout(hidden, keep_prob, seed=FLAGS.SEED)
  with tf.variable_scope('fc2'):
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


def lunaTrain(train_size, test_size):
  sssstttt = time.time()
  WORK_DIRECTORY = FLAGS.VIEW_PATH
  train_size, test_size, val_size = get_size()
  fqbt, rbt = init_bin_file('train.bin')
  fqbv, rbv = init_bin_file('val.bin')
  fqbe, rbe = init_bin_file('test.bin')

  data_node = tf.placeholder(tf.float32, shape=(
      None, FLAGS.NUM_SPEC, FLAGS.CHANNEL_NUMBER))
  labels_node = tf.placeholder(tf.int64, shape=(None, FLAGS.LABEL_NUMBER))
  keep_hidden = tf.placeholder(tf.float32)
  logits = model(data_node, keep_hidden)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_node))
  loss += apply_regularization(l2_regularizer(5e-4), tf.trainable_variables())

  batch = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(0.005, batch * FLAGS.BATCH_SIZE,
                                             train_size // 5, 0.95, staircase=True)
  optimizer = tf.train.MomentumOptimizer(
      learning_rate, 0.9).minimize(loss, global_step=batch)
  eval_predictions = tf.nn.softmax(logits)

  train_label_node, train_data_node = get_train_data(fqbt, rbt)
  val_label_node, val_data_node = get_train_data(fqbv, rbv)
  test_label_node, test_data_node = get_train_data(fqbe, rbe)

  saver = tf.train.Saver(tf.global_variables())

  TRAIN_FREQUENCY = train_size // FLAGS.BATCH_SIZE // 20
  TEST_FREQUENCY = TRAIN_FREQUENCY
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
