""" Conv4 class. """
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.misc import softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block, conv_block

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
        hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

        return hidden4

    def forward(self, inp, weights, reuse=False, scope=''):
        """The function to forward the resnet during meta-train phase
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
        hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

        return hidden4


    def forward_fc(self, inp, fc_weights):
        """The function to forward the fc layer
        Args:
          inp: input feature maps.
          fc_weights: input fc weights.
        Return:
          The processed feature maps.
        """
        net = tf.matmul(inp, fc_weights['w5']) + fc_weights['b5']
        return net

    def construct_fc_weights(self):
        """The function to construct fc weights.
        Return:
          The fc weights.
        """
        dtype = tf.float32
        fc_weights = {}
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        filter_num = FLAGS.filter_num

        if FLAGS.phase=='pre':
            fc_weights['w5'] = tf.get_variable('fc_w5', [filter_num, FLAGS.pretrain_class_num], initializer=fc_initializer)
            fc_weights['b5'] = tf.Variable(tf.zeros([FLAGS.pretrain_class_num]), name='fc_b5')
        else:
            filter_dim = FLAGS.img_size // 16
            # assumes max pooling
            fc_weights['w5'] = tf.get_variable('w5', [filter_num * filter_dim * filter_dim, self.dim_output], initializer=fc_initializer)
            fc_weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return fc_weights

    def construct_weights(self):
        """The function to construct weights.
        Return:
          The resnet weights.
        """
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        k = 3
        filter_num = FLAGS.filter_num

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([filter_num]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, filter_num, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([filter_num]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, filter_num, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([filter_num]))
        weights['conv4'] = tf.get_variable('conv4', [k, k,filter_num, filter_num], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([filter_num]))

        return weights


