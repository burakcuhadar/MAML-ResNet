##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/cbfinn/maml
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

""" Models for meta-learning. """
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.misc import get_bn_vars, compute_prototypes, proto_maml_fc_weights, proto_maml_fc_bias

FLAGS = flags.FLAGS

def MakeMetaModel():
    """The function to make meta model.
    Arg:
      Meta-train model class.
    """
    if FLAGS.backbone_arch=='resnet12':
        try:#python2
            from resnet12 import Models
        except ImportError:#python3
            from models.resnet12 import Models
    elif FLAGS.backbone_arch=='conv4':
        try:#python2
            from conv4 import Models
        except ImportError:#python3
            from models.conv4 import Models
    elif FLAGS.backbone_arch == 'conv6':
        try:  # python2
            from conv6 import Models
        except ImportError:  # python3
            from models.conv6 import Models
    else:
        print('Please set the correct backbone')

    class MetaModel(Models):
        """The class for the meta models. This class is inheritance from Models, so some variables are in the Models class."""
        def construct_model(self):
            """The function to construct meta-train model."""
            # Set the placeholder for the input episode
            self.inputa = tf.placeholder(tf.float32) # episode train images
            self.inputb = tf.placeholder(tf.float32) # episode test images
            self.labela = tf.placeholder(tf.float32) # episode train labels
            self.labelb = tf.placeholder(tf.float32) # episode test labels

            with tf.variable_scope('meta-model', reuse=None) as training_scope:
                # construct the model weights
                self.weights = weights = self.construct_weights()
                # we do not need the fc weights of the pretrain model
                if FLAGS.backbone_arch=='conv4':
                    self.weights.pop('w5', None)
                    self.weights.pop('b5', None)
                if FLAGS.backbone_arch == 'conv6':
                    self.weights.pop('w7', None)
                    self.weights.pop('b7', None)
                self.fc_weights = fc_weights = self.construct_fc_weights()

                # Load base epoch number from FLAGS
                num_updates = FLAGS.train_base_epoch_num

                # Use different learning rate per conv block per step
                self.inner_lrs = inner_lrs = self.construct_inner_lrs(num_updates)

                def task_metalearn(inp, reuse=True):
                    """The function to process one episode in a meta-batch.
                    Args:
                      inp: the input episode.
                      reuse: whether reuse the variables for the normalization.
                    Returns:
                      A series of outputs like losses and accuracies.
                    """

                    fast_weights, fast_fc_weights = self.apply_inner_loop_update(inp,
                                                                                 weights,
                                                                                 fc_weights,
                                                                                 0,
                                                                                 reuse)

                    for j in range(1, num_updates):
                        fast_weights, fast_fc_weights = self.apply_inner_loop_update(inp,
                                                                                     fast_weights,
                                                                                     fast_fc_weights,
                                                                                     j,
                                                                                     reuse)

                    inputa, inputb, labela, labelb = inp

                    # Calculate final episode test predictions
                    emb_outputb = self.forward(inputb, fast_weights, step=num_updates, reuse=reuse)
                    outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                    # Calculate the final episode test loss, it is the loss for the episode on meta-train 
                    final_lossb = self.loss_func(outputb, labelb)
                    # Used for calculating accuracy and AUC score
                    probs = tf.nn.softmax(outputb)
                    # Calculate the final episode test accuracy
                    accb = tf.contrib.metrics.accuracy(tf.argmax(probs, 1), tf.argmax(labelb, 1))

                    # Reorganize all the outputs to a list
                    task_output = [final_lossb, accb, probs]

                    return task_output

                # Initialize the batch normalization weights
                if FLAGS.norm is not None:
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

                # Set the dtype of the outputs
                out_dtype = [tf.float32, tf.float32, tf.float32]

                # Run episodes for a meta batch using parallel setting
                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                # Separate the outputs to different variables
                lossb, accsb, softmax_probs = result

            print("Constructing output variables")
            # Set the variables to output from the tensorflow graph
            self.total_loss = total_loss = tf.reduce_sum(lossb) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracy = total_accuracy = tf.reduce_sum(accsb) / tf.to_float(FLAGS.meta_batch_size)
            # Used for computing AUC score
            self.softmax_probs = softmax_probs

            # Save batch normalization variables
            self.bn_vars = get_bn_vars('meta-model')

            # Set the meta-train optimizer
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            grads_and_vars = optimizer.compute_gradients(total_loss)
            grads_and_vars = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in grads_and_vars if grad is not None]
            self.metatrain_op = optimizer.apply_gradients(grads_and_vars)

            print("Setting the tensorboard")
            # Set the tensorboard
            self.training_summaries = []
            self.training_summaries.append(tf.summary.scalar('Meta Train Loss', (total_loss / tf.to_float(FLAGS.metatrain_epite_sample_num))))
            self.training_summaries.append(tf.summary.scalar('Meta Train Accuracy', total_accuracy))

            self.training_summ_op = tf.summary.merge(self.training_summaries)

            self.input_val_loss = tf.placeholder(tf.float32)
            self.input_val_acc = tf.placeholder(tf.float32)
            self.val_summaries = []
            self.val_summaries.append(tf.summary.scalar('Meta Val Loss', self.input_val_loss))
            self.val_summaries.append(tf.summary.scalar('Meta Val Accuracy', self.input_val_acc))
            self.val_summ_op = tf.summary.merge(self.val_summaries)

            print("Meta-model is constructed.")

        def construct_test_model(self):
            """The function to construct meta-test model."""
            # Set the placeholder for the input episode
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)

            with tf.variable_scope('meta-test-model', reuse=None) as training_scope:
                # construct the model weights
                self.weights = weights = self.construct_weights()
                # we do not need the fc weights of the pretrain model
                if FLAGS.backbone_arch=='conv4':
                    self.weights.pop('w5', None)
                    self.weights.pop('b5', None)
                if FLAGS.backbone_arch == 'conv6':
                    self.weights.pop('w7', None)
                    self.weights.pop('b7', None)
                self.fc_weights = fc_weights = self.construct_fc_weights()

                # Load test base epoch number from FLAGS
                num_updates = FLAGS.test_base_epoch_num

                # Use different learning rate per layer per step
                self.inner_lrs = inner_lrs = self.construct_inner_lrs(num_updates)

                def task_metalearn(inp, reuse=True):
                    """The function to process one episode in a meta-batch.
                    Args:
                      inp: the input episode.
                      reuse: whether reuse the variables for the normalization.
                    Returns:
                      A serious outputs like losses and accuracies.
                    """

                    fast_weights, fast_fc_weights = self.apply_inner_loop_update(inp,
                                                                                 weights,
                                                                                 fc_weights,
                                                                                 0,
                                                                                 reuse)

                    for j in range(1, num_updates):
                        fast_weights, fast_fc_weights = self.apply_inner_loop_update(inp,
                                                                                     fast_weights,
                                                                                     fast_fc_weights,
                                                                                     j,
                                                                                     reuse)

                    _, inputb, _, labelb = inp

                    emb_outputb = self.forward(inputb,
                                               fast_weights,
                                               step=num_updates,
                                               reuse=reuse)
                    outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                    # Used for calculating accuracy and AUC score
                    probs = tf.nn.softmax(outputb)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(probs, 1), tf.argmax(labelb, 1))

                    lossb = self.loss_func(outputb, labelb)

                    task_output = [lossb, accb, probs]

                    return task_output

                if FLAGS.norm is not None:
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

                out_dtype = [tf.float32, tf.float32, tf.float32]

                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                lossesb, accsb, softmax_probs = result

            # Save batch normalization variables
            self.bn_vars = get_bn_vars('meta-test-model')

            self.metaval_total_loss = total_loss = tf.reduce_sum(lossesb)
            self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb)
            self.metaval_softmax_probs = softmax_probs

        def construct_inner_lrs(self, num_updates):
            inner_lrs = []
            dtype = tf.float32

            for step in range(num_updates):
                for key in self.weights.keys():
                    inner_lrs.append(tf.Variable(initial_value=self.update_lr, dtype=dtype,
                                                 trainable=True, name="lr_" + key + str(step)))
                for key in self.fc_weights.keys():
                    inner_lrs.append(tf.Variable(initial_value=self.update_lr, dtype=dtype,
                                                 trainable=True, name="lr_" + key + str(step)))

            return inner_lrs

        def get_lr_idx(self, step, key):
            sorted_keys = sorted(list(self.weights.keys()) + list(self.fc_weights.keys()))
            return step * (len(self.weights) + len(self.fc_weights)) + sorted_keys.index(key)

        def apply_inner_loop_update(self, inp, weights, fc_weights, step, reuse):
            inputa, inputb, labela, labelb = inp

            # Forward and compute loss
            emb_outputa = self.forward(inputa, weights, step=step, reuse=reuse)

            if step == 0 and FLAGS.proto_maml:
                prototypes = compute_prototypes(emb_outputa, labela)
                for key in fc_weights:
                    if 'w' in key:
                        fc_weights[key] = proto_maml_fc_weights(prototypes)
                    elif 'b' in key:
                        fc_weights[key] = proto_maml_fc_bias(prototypes)

            outputa = self.forward_fc(emb_outputa, fc_weights)
            lossa = self.loss_func(outputa, labela)

            # Calculate the gradients for the fc layer and conv weights
            grads = tf.gradients(lossa, list(fc_weights.values()) + list(weights.values()))
            gradients = dict(zip(list(fc_weights.keys()) + list(weights.keys()), grads))
            # Use gradient descent to update the fc layer and conv layers
            fast_fc_weights = dict(zip(fc_weights.keys(), [fc_weights[key] - \
                                                           self.inner_lrs[self.get_lr_idx(step, key)] * gradients[key]
                                                           for key in fc_weights.keys()]))
            fast_weights = dict(zip(weights.keys(), [weights[key] - \
                                                     self.inner_lrs[self.get_lr_idx(step, key)] * gradients[key]
                                                     for key in weights.keys()]))

            return fast_weights, fast_fc_weights

    return MetaModel()

