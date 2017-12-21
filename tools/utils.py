# Reference: https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py

# Import packages
import os
import copy
import numpy as np
import tensorflow as tf
import skimage.io as io

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

# Define os flags
tf.app.flags.DEFINE_integer('num_threads', 4, 'Number of threads to use in preprocessing and loading')
tf.app.flags.DEFINE_integer('num_readers', 4, 'Number of readers to use in loading')
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 1, 'Hyperparameter for memory usage')
tf.app.flags.DEFINE_integer('min_after_dequeue', 10, 'Minimum queues after dequeue')

# Define a concurrent, multithreaded loader
def load_batch(shards, batch_size, train=True, resize=True, crop=True, flip=True):

  with tf.name_scope('load_batch'):
    if train:
      filename_queue = tf.train.string_input_producer([shards], shuffle=True, capacity=16)
    else:
      filename_queue = tf.train.string_input_producer([shards], shuffle=False, capacity=1)

  examples_per_shard = 100000
  min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor

  if train:
      examples_queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                             min_after_dequeue=min_queue_examples,
                                             dtypes=[tf.string])
  else:
      examples_queue = tf.FIFOQueue(capacity=examples_per_shard + 3 * batch_size,
                                    dtypes=[tf.string])

  enqueue_ops = []

  for _ in range(FLAGS.num_readers):
      reader = tf.TFRecordReader()
      _, value = reader.read(filename_queue)
      enqueue_ops.append(examples_queue.enqueue([value]))

  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue,
                                                                           enqueue_ops))
  example_serialized = examples_queue.dequeue()

  images_and_labels = []

  for thread_id in range(FLAGS.num_threads):
    # Parse a serialized Example proto to extract the image and metadata.
    single_features = tf.parse_single_example(example_serialized,
                                              features = {
                                                'height': tf.FixedLenFeature([], tf.int64),
                                                'width': tf.FixedLenFeature([], tf.int64),
                                                'image_raw': tf.FixedLenFeature([], tf.string),
                                                'label': tf.FixedLenFeature([], tf.int64)
                                              })

    image_buffer = tf.decode_raw(single_features['image_raw'], tf.uint8)
    image = tf.cast(image_buffer, tf.float32)
    image = tf.reshape(image, (64,64,3))

    if crop:
      FLAGS.image_size = 56
      #image = tf.random_crop(image, size=(56,56,3))
      image = tf.image.resize_image_with_crop_or_pad(image, 56, 56)
      
    if flip:
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_flip_up_down(image)

    annotation = tf.cast(single_features['label'], tf.int64)

    images_and_labels.append([image, annotation])

  images, label_index_batch = tf.train.batch_join(images_and_labels,
                                                  batch_size=batch_size,
                                                  capacity=2 * FLAGS.num_threads * batch_size)

  # Reshape images into these desired dimensions.
  height = FLAGS.image_size
  width = FLAGS.image_size
  depth = 3

  images = tf.cast(images, tf.float32)
  images = tf.reshape(images, shape=[batch_size, height, width, depth])

  # Display the training images in the visualizer.
  tf.summary.image('images', images, max_outputs=512)

  return images, tf.reshape(label_index_batch, [batch_size])


# Define a basic decoder for our local loader
def read_and_decode(filename_queue, crop=True, flip=True):

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(serialized_example,
                                     features = {
                                        'height': tf.FixedLenFeature([], tf.int64),
                                        'width': tf.FixedLenFeature([], tf.int64),
                                        'image_raw': tf.FixedLenFeature([], tf.string),
                                        'label': tf.FixedLenFeature([], tf.int64)})

  image_buffer = tf.decode_raw(features['image_raw'], tf.uint8)
  image = tf.cast(image_buffer, tf.float32)
  image = tf.reshape(image, (64,64,3))
  label = tf.cast(features['label'], tf.int64)

  if crop:
    image = tf.image.resize_image_with_crop_or_pad(image, 56, 56)

  if flip:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

  return image, label


# Define a basic loader without concurrency
# TODO: Import the dataset locally as a numpy array
def load_local_dataset(shards):
  filename_queue = tf.train.string_input_producer([shards])
  image, label = read_and_decode(filename_queue)
  capacity = FLAGS.min_after_dequeue + 3 * FLAGS.batch_size

  images, labels = tf.train.shuffle_batch([image, label],
                                          batch_size=FLAGS.batch_size,
                                          capacity=capacity,
                                          min_after_dequeue=FLAGS.min_after_dequeue)

  # Reshape images into these desired dimensions.
  height = FLAGS.image_size
  width = FLAGS.image_size
  depth = 3

  images = tf.cast(images, tf.float32)
  images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])

  # Display the training images in the visualizer.
  tf.summary.image('images', images, max_outputs=512)

  return images, tf.reshape(labels, [FLAGS.batch_size])