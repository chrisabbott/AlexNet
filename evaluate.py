import os
import copy
import numpy as np
import tensorflow as tf

from tools import utils
from models.vanilla import alexnet
from models.slim import model
import tensorflow.contrib.slim.nets as nets

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim
metrics = tf.contrib.metrics

# Define os and dataset flags
tf.app.flags.DEFINE_string('data_dir', '/home/christian/Data/ILSVRC/tfrecords/', 'Path to data directory')
tf.app.flags.DEFINE_string('train_dir', '/home/christian/AlexNet/datasets/tiny-imagenet-200/cached/train.tfrecords', 'Path to training data')
tf.app.flags.DEFINE_string('val_dir', '/home/christian/AlexNet/datasets/tiny-imagenet-200/cached/test.tfrecords', 'Path to validation data')
tf.app.flags.DEFINE_string('log_dir', '/home/christian/AlexNet/logs', 'Path to the log folder')
tf.app.flags.DEFINE_string('trainlog_dir', '/home/christian/AlexNet/logs/train', 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_dir', '/home/christian/AlexNet/logs/eval', 'Path to the evaluation log folder')
tf.app.flags.DEFINE_string('trainlog_momentum_dir', '/home/christian/AlexNet/logs/train/train_momentum/checkpoint.ckpt', 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_momentum_dir', '/home/christian/AlexNet/logs/eval/eval_momentum', 'Path to the evaluation log folder')
tf.app.flags.DEFINE_string('trainlog_gradient_descent_dir', '/home/christian/AlexNet/logs/train/train_gradient_descent/checkpoint.ckpt', 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_gradient_descent_dir', '/home/christian/AlexNet/logs/eval/eval_gradient_descent', 'Path to the evaluation log folder')
tf.app.flags.DEFINE_string('trainlog_adadelta_dir', '/home/christian/AlexNet/logs/train/train_adadelta/checkpoint.ckpt', 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_adadelta_dir', '/home/christian/AlexNet/logs/eval/eval_adadelta', 'Path to the evaluation log folder')
tf.app.flags.DEFINE_string('trainlog_adam_dir', '/home/christian/AlexNet/logs/train/train_adam/checkpoint.ckpt', 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_adam_dir', '/home/christian/AlexNet/logs/eval/eval_adam', 'Path to the evaluation log folder')
tf.app.flags.DEFINE_string('trainlog_rmsprop_dir', '/home/christian/AlexNet/logs/train/train_rmsprop/checkpoint.ckpt', 'Path to the training log folder')
tf.app.flags.DEFINE_string('evallog_rmsprop_dir', '/home/christian/AlexNet/logs/eval/eval_rmsprop', 'Path to the evaluation log folder')
tf.app.flags.DEFINE_integer('num_classes', 200, 'Number of classes in Tiny ImageNet')

# Define training flags
tf.app.flags.DEFINE_integer('batch_size', 256, 'Batch size')
tf.app.flags.DEFINE_integer('image_size', 64, 'Image size')
tf.app.flags.DEFINE_integer('max_steps', 1, 'Maximum number of steps before termination')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Total number of epochs')
tf.app.flags.DEFINE_integer('num_evals', 1, 'Number of batches to evaluate')

# Directories containing the training and validation tfrecords, respectively
TRAIN_SHARDS = FLAGS.train_dir
VAL_SHARDS = FLAGS.val_dir

def evaluate():
  with tf.Graph().as_default():
    config = tf.ConfigProto(device_count={'GPU':0})

    images, labels = utils.load_batch(shards=VAL_SHARDS,
                                      batch_size=FLAGS.batch_size,
                                      train=False,
                                      crop=False,
                                      flip=False)

    predictions = alexnet.AlexNet(images, batch_size=FLAGS.batch_size).model
    prediction = tf.to_int64(tf.argmax(predictions, 1)) # Returns index of largest

    metrics_to_values, metrics_to_updates = metrics.aggregate_metric_map({
        'mse': metrics.streaming_mean_squared_error(prediction, labels),
        'rmse': metrics.streaming_root_mean_squared_error(prediction, labels),
        'accuracy': metrics.streaming_accuracy(prediction, labels),
        'precision': metrics.streaming_precision(prediction, labels),
    })

    for metric_name, metric_value in metrics_to_values.items():
        tf.summary.scalar(metric_name, metric_value)

    slim.evaluation.evaluation_loop(
        '',
        FLAGS.trainlog_dir,
        FLAGS.evallog_dir,
        num_evals=FLAGS.num_evals,
        eval_op = list(metrics_to_updates.values()),
        eval_interval_secs=5,
        session_config=config)

    '''checkpoint_list = [FLAGS.trainlog_momentum_dir,
                       FLAGS.trainlog_gradient_descent_dir,
                       FLAGS.trainlog_adadelta_dir,
                       FLAGS.trainlog_adam_dir,
                       FLAGS.trainlog_rmsprop_dir]

    evallog_list = [FLAGS.evallog_momentum_dir,
                    FLAGS.evallog_gradient_descent_dir,
                    FLAGS.evallog_adadelta_dir,
                    FLAGS.evallog_adam_dir,
                    FLAGS.evallog_rmsprop_dir]

    while (1):
        for i in range(5):
            slim.evaluation.evaluate_once(
                '',
                checkpoint_list[i],
                evallog_list[i],
                num_evals=FLAGS.num_evals,
                eval_op = list(metrics_to_updates.values()),
                session_config=config)'''

evaluate()