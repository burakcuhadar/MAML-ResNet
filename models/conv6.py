""" Conv6 class. """
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.misc import softmaxloss, xent, conv_block, conv_block_no_pool

FLAGS = flags.FLAGS

class Models:
    """The class that contains the code for the basic resnet models and SS weights"""
    def __init__(self):
        # Set the dimension number for the input feature maps
        self.dim_input = FLAGS.img_size * FLAGS.img_size * 3
        # Set the dimension number for the outputs
        self.dim_output = FLAGS.way_num
        # Load base learning rates from FLAGS
        self.update_lr = FLAGS.base_lr
        # Load the pre-train phase class number from FLAGS
        self.pretrain_class_num = FLAGS.pretrain_class_num
        # Set the initial meta learning rate
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        # Set the initial pre-train learning rate
        self.pretrain_lr = tf.placeholder_with_default(FLAGS.pre_lr, ())

        # Set the default objective functions for meta-train and pre-train
        self.loss_func = xent
        self.pretrain_loss_func = softmaxloss

        # Set the default channel number to 3
        self.channels = 3
        # Load the image size from FLAGS
        self.img_size = FLAGS.img_size


    def forward_pretrain(self, inp, weights, reuse=False, scope=''):
        """The function to forward the resnet during pre-train phase
        Args:
          inp: input feature maps.
          weights: input resnet weights.
          reuse: reuse the batch norm weights or not.
          scope: the label to indicate which layer we are processing.
        Return:
          The processed feature maps.
        """

        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
        hidden5 = conv_block_no_pool(hidden4, weights['conv5'], weights['b5'], reuse, scope + '4')
        hidden6 = conv_block_no_pool(hidden5, weights['conv6'], weights['b6'], reuse, scope + '5')
        hidden6 = tf.reshape(hidden6, [-1, np.prod([int(dim) for dim in hidden6.get_shape()[1:]])])

        return hidden6

    def forward(self, inp, weights, step, reuse=False, scope=''):
        """The function to forward the resnet during meta-train phase
        Args:
          inp: input feature maps.
          weights: input resnet weights.
          step: index for the inner loop, used to decide which bn layer to use
          reuse: reuse the batch norm weights or not.
          scope: the label to indicate which layer we are processing.
        Return:
          The processed feature maps.
        """
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0' + str(step))
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1' + str(step))
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2' + str(step))
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3' + str(step))
        hidden5 = conv_block_no_pool(hidden4, weights['conv5'], weights['b5'], reuse, scope + '4' + str(step))
        hidden6 = conv_block_no_pool(hidden5, weights['conv6'], weights['b6'], reuse, scope + '5' + str(step))
        hidden6 = tf.reshape(hidden6, [-1, np.prod([int(dim) for dim in hidden6.get_shape()[1:]])])

        return hidden6

    def forward_fc(self, inp, fc_weights):
        """The function to forward the fc layer
        Args:
          inp: input feature maps.
          fc_weights: input fc weights.
        Return:
          The processed feature maps.
        """
        net = tf.matmul(inp, fc_weights['w7']) + fc_weights['b7']
        return net

    def construct_fc_weights(self):
        """The function to construct fc weights.
        Return:
          The fc weights.
        """
        dtype = tf.float32
        fc_weights = {}
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        filter_num = 32

        if FLAGS.phase=='pre':
            fc_weights['w7'] = tf.get_variable('fc_w7', [filter_num, FLAGS.pretrain_class_num], initializer=fc_initializer)
            fc_weights['b7'] = tf.Variable(tf.zeros([FLAGS.pretrain_class_num]), name='fc_b7')
        else:
            # assumes max pooling
            fc_weights['w7'] = tf.get_variable('w7', [filter_num * 5 * 5, self.dim_output], initializer=fc_initializer)
            fc_weights['b7'] = tf.Variable(tf.zeros([self.dim_output]), name='b7')
        return fc_weights

    def construct_weights(self):
        """The function to construct resnet weights.
        Return:
          The resnet weights.
        """
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        k = 3
        filter_num = 32

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([filter_num]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, filter_num, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([filter_num]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, filter_num, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([filter_num]))
        weights['conv4'] = tf.get_variable('conv4', [k, k,filter_num, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([filter_num]))

        # Additional conv weights for conv6
        weights['conv5'] = tf.get_variable('conv5', [k, k, filter_num, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([filter_num]))
        weights['conv6'] = tf.get_variable('conv6', [k, k, filter_num, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b6'] = tf.Variable(tf.zeros([filter_num]))

        return weights


