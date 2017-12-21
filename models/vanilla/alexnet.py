import tensorflow as tf

# Define tensor constants
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 200

# Layer definition convenience functions
class AlexNet:

    def __init__(self, X, batch_size):
        self.batch_size = batch_size
        self.model = self.build_alexnet(X)

    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

        return var


    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
    def _variable_with_weight_decay(self, name, shape, stddev, wd, xavier=False):
        dtype = tf.float32

        if xavier:
          var = self._variable_on_cpu(name,
                                      shape,
                                      tf.contrib.layers.xavier_initializer())
        else:
          var = self._variable_on_cpu(name,
                                      shape,
                                      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return var


    ''' Convenience function to generate a convolutional layer with leaky relu activation
            Args: input, filter, stride, name, layer
            Returns: output layer
    '''
    def convolution(self, input_, filter_, stride, name, padding='SAME', alpha=0.1, relu=True):
        with tf.variable_scope(name) as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=filter_,
                                                      stddev=5e-2,
                                                      wd=0.0,
                                                      xavier=relu)
            conv = tf.nn.conv2d(input_, kernel, stride, padding)
            biases = self._variable_on_cpu('biases', [filter_[-1]],
                                      tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)

            if relu:
                output = tf.nn.relu(pre_activation, name=scope.name)
            else:
                output = tf.nn.leaky_relu(pre_activation, alpha=alpha, name=scope.name)

        return output


    ''' Convenience function to generate a fully connected layer with leaky relu activation
            Args: input, filter, stride, name, layer
            Returns: output layer
    '''
    def fully_connected(self, input_, shape_, name, alpha=0.1, linear=False):
        with tf.variable_scope(name) as scope:
            weights = self._variable_with_weight_decay('weights', shape = shape_,
                                                                  stddev = 0.04,
                                                                  wd = 0.004)
            biases = self._variable_on_cpu('biases', shape_[-1], tf.constant_initializer(0.1))

            # TODO: Implement linear activation
            if linear:
                # Make this linear
                fc = tf.nn.leaky_relu(tf.matmul(input_, weights) + biases, alpha=alpha, name=scope.name)
            else:
                fc = tf.nn.leaky_relu(tf.matmul(input_, weights) + biases, alpha=alpha, name=scope.name)

            return fc


    def build_alexnet(self, images):

        conv1 = self.convolution(images,
                                 filter_=[5, 5, 3, 96],
                                 stride=[1, 4, 4, 1],
                                 name="conv1")

        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 1, 1, 1],
                               padding="SAME",
                               name="pool1")

        conv2 = self.convolution(pool1,
                                 filter_=[3, 3, 96, 256],
                                 stride=[1, 2, 2, 1],
                                 name="conv2")

        pool2 = tf.nn.max_pool(conv2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 1, 1, 1],
                               padding="SAME",
                               name="pool2")

        conv3 = self.convolution(pool2,
                                 filter_=[3, 3, 256, 256],
                                 stride=[1, 1, 1, 1],
                                 name="conv3")

        conv4 = self.convolution(conv3,
                                 filter_=[3, 3, 256, 128],
                                 stride=[1, 1, 1, 1],
                                 name="conv4")

        conv5 = self.convolution(conv4,
                                 filter_=[3, 3, 128, 64],
                                 stride=[1, 1, 1, 1],
                                 name="conv5")

        pool3 = tf.nn.max_pool(conv5,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 1, 1, 1],
                               padding="SAME",
                               name="pool2")

        pool3_f = tf.reshape(pool3, [self.batch_size, -1])
        i = pool3_f.get_shape()[1].value

        fc1 = self.fully_connected(pool3_f,
                                   shape_=[i, 4096],
                                   name="fc1")

        fc1_f = tf.reshape(fc1, [self.batch_size, -1])
        j = fc1_f.get_shape()[1].value

        fc2 = self.fully_connected(fc1_f,
                                   shape_=[j, 4096],
                                   name="fc2")

        fc2_f = tf.reshape(fc2, [self.batch_size, -1])
        k = fc2_f.get_shape()[1].value

        out = self.fully_connected(fc2_f,
                                   shape_=[k, 200],
                                   name="output")

        out = tf.clip_by_norm(out, clip_norm=100)

        return out