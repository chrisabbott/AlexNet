import tensorflow as tf
import os

from tensorflow.python.ops import array_ops

slim = tf.contrib.slim

########################################################################################################
# AlexNet architecture
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
########################################################################################################
def AlexNet(inputs, is_training=True):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      stride=1):
    net = slim.conv2d(inputs, 96, [5,5], stride=4)
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 256, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 256, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 64, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.fully_connected(net, 4096)
    net = slim.dropout(net, is_training=is_training)
    net = slim.fully_connected(net, 4096)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 200)
    return net

########################################################################################################
# Custom AlexNetLarge architecture, modified to contain extra feature maps
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
########################################################################################################
def AlexNetLarge(inputs, is_training=True):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      stride=1):
    net = slim.conv2d(inputs, 96, [11,11], stride=4)
    net = slim.max_pool2d(net, [3,3], stride=2)
    net = slim.conv2d(net, 256, [5,5], stride=1)
    net = slim.max_pool2d(net, [3,3], stride=2)
    net = slim.conv2d(net, 384, [3,3])
    net = slim.conv2d(net, 384, [3,3])
    net = slim.conv2d(net, 256, [3,3])
    net = slim.max_pool2d(net, [3,3])
    net = slim.fully_connected(net, 4096)
    net = slim.dropout(net, is_training=is_training)
    net = slim.fully_connected(net, 4096)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 200)
    return net

########################################################################################################
# Custom AlexNetXL architecture, modified for depth
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
########################################################################################################
def AlexNetXL(inputs, is_training=True):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      stride=1):
    net = slim.conv2d(inputs, 96, [11,11], stride=4)
    net = slim.conv2d(net, 256, [5,5], stride=1)
    net = slim.max_pool2d(net, [3,3], stride=2)
    net = slim.conv2d(net, 384, [3,3])
    net = slim.conv2d(net, 384, [3,3])
    net = slim.conv2d(net, 256, [3,3])
    net = slim.max_pool2d(net, [3,3])
    net = slim.conv2d(net, 256, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 64, [3,3])
    net = slim.max_pool2d(net, [3,3])
    net = slim.fully_connected(net, 4096)
    net = slim.dropout(net, is_training=is_training)
    net = slim.fully_connected(net, 4096)
    net = slim.dropout(net, is_training=is_training)
    net = slim.fully_connected(net, 200)
    net = array_ops.squeeze(net, [1, 2])
    return net

########################################################################################################
# Convolutional network utilizing fractional max-pooling to allow added depth
# Based off of Ben Graham's Fractional Max Pooling architecture
# https://arxiv.org/abs/1412.6071
########################################################################################################
def ChrisNet(inputs, is_training=True):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      stride=1):
    net = slim.conv2d(inputs, 96, [7,7], stride=4)
    net = tf.nn.fractional_max_pool(net, 
                                    pooling_ratio=[1.0,1.6,1.6,1.0],
                                    deterministic=True)[0]

    net = slim.conv2d(net, 256, [5,5])
    net = tf.nn.fractional_max_pool(net, 
                                    pooling_ratio=[1.0,1.6,1.6,1.0],
                                    deterministic=True)[0]

    net = slim.conv2d(net, 256, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = tf.nn.fractional_max_pool(net, 
                                    pooling_ratio=[1.0,1.6,1.6,1.0],
                                    deterministic=True)[0]

    net = slim.conv2d(net, 256, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 64, [3,3])
    net = tf.nn.fractional_max_pool(net, 
                                    pooling_ratio=[1.0,1.6,1.6,1.0],
                                    deterministic=True)[0]

    net = slim.fully_connected(net, 4096)
    net = slim.dropout(net, is_training=is_training)
    net = slim.fully_connected(net, 4096)
    net = slim.fully_connected(net, 200)
    net = array_ops.squeeze(net, [1, 2])
    return net

########################################################################################################
# A weak baseline model used for testing
########################################################################################################
def baseline(inputs):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.conv2d(inputs, 16, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 200)
    return net

########################################################################################################
# Custom SimpleNet architecture
# https://arxiv.org/pdf/1608.06037.pdf
########################################################################################################
def simplenetA(inputs, softmax=False, is_training=True):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.005),
                      stride=1):

    net = slim.conv2d(inputs, 64, [3,3])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [1,1])

    net = slim.conv2d(net, 128, [1,1])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])

    if softmax:
      net = slim.conv2d(net, 200, [1,1], activation_fn=slim.softmax)
      net = array_ops.squeeze(net, [1, 2])
      return net
    else:
      net = slim.conv2d(net, 200, [1,1], activation_fn=None)
      net = array_ops.squeeze(net, [1, 2])
      return net
    
########################################################################################################
# Custom SimpleNet architecture
# https://arxiv.org/pdf/1608.06037.pdf
########################################################################################################
def simplenetB(inputs, softmax=False, is_training=True):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.005),
                      stride=1):

    net = slim.conv2d(inputs, 16, [3,3])

    net = slim.conv2d(net, 32, [3,3])
    net = slim.conv2d(net, 32, [3,3])
    net = slim.conv2d(net, 64, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 32, [3,3])
    net = slim.conv2d(net, 64, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 32, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 16, [3,3])
    net = slim.conv2d(net, 32, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 64, [3,3])
    net = slim.conv2d(net, 128, [1,1])

    net = slim.conv2d(net, 64, [1,1])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 64, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.dropout(net)

    if softmax:
      net = slim.conv2d(net, 200, [1,1], activation_fn=slim.softmax)
      net = array_ops.squeeze(net, [1, 2])
      return net
    else:
      net = slim.conv2d(net, 200, [1,1], activation_fn=None)
      net = array_ops.squeeze(net, [1, 2])
      return net

########################################################################################################
# Custom SimpleNet architecture
# https://arxiv.org/pdf/1608.06037.pdf
########################################################################################################
def simplenetC(inputs, softmax=False, is_training=True):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.005),
                      stride=1):

    net = slim.conv2d(inputs, 16, [2,2])

    net = slim.conv2d(net, 32, [2,2])
    net = slim.conv2d(net, 32, [1,1])
    net = slim.conv2d(net, 64, [2,2])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 32, [2,2])
    net = slim.conv2d(net, 64, [2,2])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 32, [2,2])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 16, [2,2])
    net = slim.conv2d(net, 32, [2,2])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 64, [2,2])
    net = slim.conv2d(net, 128, [1,1])

    net = slim.conv2d(net, 64, [1,1])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 64, [2,2])
    net = slim.max_pool2d(net, [2,2])
    net = slim.dropout(net)

    if softmax:
      net = slim.conv2d(net, 200, [1,1], activation_fn=slim.softmax)
      net = array_ops.squeeze(net, [1, 2])
      return net
    else:
      net = slim.conv2d(net, 200, [1,1], activation_fn=None)
      net = array_ops.squeeze(net, [1, 2])
      return net
    
########################################################################################################
# Miniaturized YOLO Network, based off of Joseph Redmon's YOLO object detector
# https://arxiv.org/abs/1506.02640
########################################################################################################
def tiny_yolo(inputs, is_training=True, pretrain=False):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    
    net = slim.conv2d(inputs, 16, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 32, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.conv2d(net, 64, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.batch_norm(net, is_training=is_training)
    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])
    net = slim.dropout(net, is_training=is_training)
    net = slim.conv2d(net, 256, [3,3])
    net = slim.max_pool2d(net, [2,2], stride=1)
    net = slim.dropout(net, is_training=is_training)
    net = slim.conv2d(net, 512, [3,3], activation_fn=None)

    if pretrain:
      net = slim.avg_pool2d(net, [2,2], stride=1)
      net = slim.flatten(net)
      net = slim.fully_connected(net, 200, activation_fn=slim.softmax)
      return net

    net = slim.conv2d(net, 512, [3,3])
    net = slim.conv2d(net, 425, [3,3])
    net = slim.fully_connected(net, 4096)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 7 * 7 * 30, activation_fn=slim.softmax)
    return net

########################################################################################################
# Customized VGG network
# https://arxiv.org/abs/1409.1556
########################################################################################################
def VGG16Custom(inputs):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      stride=1):
    net = slim.conv2d(inputs, 32, [3,3])
    net = slim.conv2d(net, 32, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 64, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 128, [3,3])
    net = slim.conv2d(net, 128, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(net, 256, [3,3])
    net = slim.conv2d(net, 256, [3,3])
    net = slim.conv2d(net, 512, [3,3])
    net = slim.max_pool2d(net, [2,2])

    net = slim.fully_connected(net, 4096, activation_fn=tf.nn.relu)
    net = slim.fully_connected(net, 4096, activation_fn=tf.nn.relu)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 200, activation_fn=slim.softmax)
    return net


########################################################################################################
# Customized VGG network
# https://arxiv.org/abs/1409.1556
########################################################################################################
def VGGmod(inputs):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      #weights_regularizer=slim.l2_regularizer(0.005),
                      stride=1):

    net = slim.conv2d(inputs, 32, [2,2])
    net = slim.conv2d(net, 32, [2,1])
    net = slim.conv2d(net, 32, [1,2])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(inputs, 48, [2,2])
    net = slim.conv2d(net, 48, [2,2])
    net = slim.conv2d(net, 48, [2,2])
    net = slim.max_pool2d(net, [2,2])

    net = slim.conv2d(inputs, 80, [2,2])
    net = slim.conv2d(net, 80, [2,1])
    net = slim.conv2d(net, 80, [1,2])
    net = slim.max_pool2d(net, [2,2])

    net = slim.flatten(net)
    net = slim.fully_connected(net, 200, activation_fn=slim.softmax)
    return net