#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 qingze <qingze@localhost.localdomain>
#
# Distributed under terms of the MIT license.

"""

"""
import sys
sys.path.append('..')
# from utils.reader.read_comment.reader import Reader
import tensorflow as tf

from datetime import datetime
import numpy as np
import os

from reader import Reader
from fcn8_vgg import FCN8VGG

train_dir = './train'
batch_size = 1
max_len = 260
max_steps = 5000000
test_steps = 100
# restore_path = './train/'
restore_path = './train/'
test_dir = './test'

reader = Reader('/export/wangqingze/fcn.berkeleyvision.org/data/card', batch_size = batch_size)
model = FCN8VGG('./vgg16.npy')


def evaluate():
    feed_dict  = reader.next_test()
    images = feed_dict['images']
    labels = feed_dict['labels']
    prob, _ = model.build(images)
    correct = tf.nn.in_top_k(prob, labels, 1)

    return tf.reduce_sum(tf.cast(correct, tf.int32))

def precision(num_correct):
    precision = num_correct * 1. / (test_steps * batch_size)
    print('%s: testing output' %(datetime.now()))
    print('     Num examples: %d Num corrcet: %d Precision @ 1: %0.04f' %
            (test_steps * batch_size, num_correct, precision))


def train():
    with tf.Graph().as_default():
        feed_dict  = reader.next_train()
        images = feed_dict['images']
        labels = feed_dict['labels']

        with tf.device('/gpu:1'):
            logits = model.build(images)
            total_loss = model.loss(logits, labels)

        losses = tf.get_collection('losses')
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name = 'avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.scalar_summary(l.op.name +' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)

        # global_step = tf.get_variable('global_step', [],
        #                             initializer = tf.constant_initializer(0), trainable = False)
        global_step = tf.Variable(0, trainable = False)

        lr = tf.train.exponential_decay(0.01,
                            global_step,
                            1000,
                            0.9999,
                            staircase = True)

        opt = tf.train.GradientDescentOptimizer(lr)
        grads_and_vars = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step = global_step)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.scalar_summary('learning_rate', lr))

        for grad, var in grads_and_vars:
            summaries.append(tf.histogram_summary(var.op.name + '/gradients', grad))
            summaries.append(tf.histogram_summary(var.op.name, var))


        variable_averages = tf.train.ExponentialMovingAverage(0.9, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        summary_op = tf.merge_summary(summaries)

        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # variables = tf.all_variables()
        # for var in variables:
        #     print(var.name)

        # with tf.variable_scope(scope, reuse = True):
        #     evaluate_op = evaluate()

        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
                                                      log_device_placement = False))


        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        sess.run(init)


        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep = 25)

        if restore_path is not None:
            ckpt = tf.train.get_checkpoint_state(restore_path)
            if ckpt and ckpt.model_checkpoint_path:
                print('load from ' + ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

        # Start the queue runners.
        tf.train.start_queue_runners(sess = sess)

        for step in range(max_steps):
            _, loss = sess.run([train_op, total_loss])

            assert not np.isnan(loss), 'Model diverged with loss = NaN'

            if step != 0 and step % 100 == 0:
                format_str = ('%s: step %d, loss = %.2f')
                print(format_str % (datetime.now(), step, loss))

            if step != 0 and step % 1000 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # if step % 100 == 0:
            #     cnt = 0
            #     for i in xrange(test_steps):
            #         cnt += sess.run(evaluate_op)

            #     precision(cnt);

            # Save the model checkpoint periodically.
            if (step != 0 and step % 1000 == 0) :
                path = os.path.join(train_dir, 'comment')
                saver.save(sess, path, global_step = step)



if __name__ == '__main__':
    train()
