import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tools import utils
from models.slim import model
from models.vanilla import alexnet

FLAGS = tf.app.flags.FLAGS
cwd = os.path.dirname(os.path.realpath(__file__))

# Define os and dataset flags
tf.app.flags.DEFINE_string('train_dir', os.path.join(cwd, 'dataset/train.tfrecords'), 'Training data')
tf.app.flags.DEFINE_string('val_dir', os.path.join(cwd, 'dataset/test.tfrecords'), 'Validation data')
tf.app.flags.DEFINE_string('trainlog_dir', os.path.join(cwd, 'logs/train/'), 'Training logs')
tf.app.flags.DEFINE_string('evallog_dir', os.path.join(cwd, 'logs/eval/'), 'Evaluation logs')

# Define training flags
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, 'Initial learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay', 0.4, 'Learning rate decay')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum optimizer')
tf.app.flags.DEFINE_float('adam_epsilon', 0.1, 'Stability value for adam optimizer')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('image_size', 64, 'Image size')
tf.app.flags.DEFINE_integer('max_steps', 25000, 'Maximum number of steps before termination')
tf.app.flags.DEFINE_integer('num_classes', 200, 'Number of classes in Tiny ImageNet')

# Directories containing the training and validation tfrecords, respectively
TRAIN_SHARDS = FLAGS.train_dir
VAL_SHARDS = FLAGS.val_dir

# config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction=0.5 # don't hog all vRAM


# Momentum optimizer with log loss (using nesterov and a lower initial momentum)
def train_momentum_cross_entropy():
    with tf.Graph().as_default():
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.inverse_time_decay(learning_rate=FLAGS.initial_learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=5000,
                                                    decay_rate=FLAGS.learning_rate_decay)

        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True,
                                          flip=False,
                                          crop=False)

        labels = tf.one_hot(labels, depth=200)

        # TF-Slim model
        # predictions = model.AlexNet(images)

        # Vanilla TensorFlow model
        predictions = alexnet.AlexNet(images, batch_size=FLAGS.batch_size).model

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, 
                                               logits=predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                               momentum=FLAGS.momentum,
                                               use_nesterov=True)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Setup the saver to save only one checkpoint and not destroy disk space
        saver = tf.train.Saver(max_to_keep=1, filename='checkpoint')

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30,
                            saver=saver)


# Gradient descent optimizer
def train_gradient_descent_cross_entropy():
    with tf.Graph().as_default():
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.inverse_time_decay(learning_rate=FLAGS.initial_learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=5000,
                                                    decay_rate=FLAGS.learning_rate_decay)

        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True,
                                          flip=False,
                                          crop=False)

        labels = tf.one_hot(labels, depth=200)

        # TF-Slim model
        # predictions = model.AlexNet(images)

        # Vanilla TensorFlow model
        predictions = alexnet.AlexNet(images, batch_size=FLAGS.batch_size).model

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, 
                                               logits=predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.initial_learning_rate)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Setup the saver to save only one checkpoint and not destroy disk space
        saver = tf.train.Saver(max_to_keep=1, filename='checkpoint')

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_gradient_descent_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30,
                            saver=saver)


# Gradient descent optimizer
def train_adadelta_cross_entropy():
    with tf.Graph().as_default():
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.inverse_time_decay(learning_rate=FLAGS.initial_learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=5000,
                                                    decay_rate=FLAGS.learning_rate_decay)

        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True,
                                          flip=False,
                                          crop=False)

        labels = tf.one_hot(labels, depth=200)

        # TF-Slim model
        # predictions = model.AlexNet(images)

        # Vanilla TensorFlow model
        predictions = alexnet.AlexNet(images, batch_size=FLAGS.batch_size).model

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, 
                                               logits=predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.initial_learning_rate)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Setup the saver to save only one checkpoint and not destroy disk space
        saver = tf.train.Saver(max_to_keep=1, filename='checkpoint')

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_adadelta_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30,
                            saver=saver)


# AdaM optimizer
def train_adam_cross_entropy():
    with tf.Graph().as_default():
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.inverse_time_decay(learning_rate=FLAGS.initial_learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=5000,
                                                    decay_rate=FLAGS.learning_rate_decay)

        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True,
                                          flip=False,
                                          crop=False)

        labels = tf.one_hot(labels, depth=200)

        # TF-Slim model
        # predictions = model.AlexNet(images)

        # Vanilla TensorFlow model
        predictions = alexnet.AlexNet(images, batch_size=FLAGS.batch_size).model

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, 
                                               logits=predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                           epsilon=FLAGS.adam_epsilon)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Setup the saver to save only one checkpoint and not destroy disk space
        saver = tf.train.Saver(max_to_keep=1, filename='checkpoint')

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_adam_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30,
                            saver=saver)


# AdaM optimizer
def train_rmsprop_momentum_cross_entropy():
    with tf.Graph().as_default():
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.inverse_time_decay(learning_rate=FLAGS.initial_learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=5000,
                                                    decay_rate=FLAGS.learning_rate_decay)

        images, labels = utils.load_batch(shards=TRAIN_SHARDS,
                                          batch_size=FLAGS.batch_size,
                                          train=True,
                                          flip=False,
                                          crop=False)

        labels = tf.one_hot(labels, depth=200)

        # TF-Slim model
        # predictions = model.AlexNet(images)

        # Vanilla TensorFlow model
        predictions = alexnet.AlexNet(images, batch_size=FLAGS.batch_size).model

        # Define loss function
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, 
                                               logits=predictions)
        tf.summary.scalar('loss', loss)

        # Define optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                              epsilon=FLAGS.adam_epsilon,
                                              momentum=FLAGS.momentum)

        # Create training op
        train_op = slim.learning.create_train_op(loss, optimizer)

        # Setup the saver to save only one checkpoint and not destroy disk space
        saver = tf.train.Saver(max_to_keep=1, filename='checkpoint')

        # Initialize training
        slim.learning.train(train_op,
                            FLAGS.trainlog_rmsprop_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=30,
                            save_interval_secs=30,
                            saver=saver)


if __name__ == '__main__':
    # TODO continue implementing this functionality
    # try:
    #     logs_dir = sys.argv[1]
    # except IndexError:
    #     logs_dir = './logs/'
    #
    # tf.app.flags.DEFINE_string('log_dir', logs_dir, 'Path to the log folder')

    train_momentum_cross_entropy()
    # train_gradient_descent_cross_entropy()
    # train_adadelta_cross_entropy()
    # train_adam_cross_entropy()
    # train_rmsprop_momentum_cross_entropy()
