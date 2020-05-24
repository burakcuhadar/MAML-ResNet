##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer
from data_generator.meta_data_generator import MetaDataGenerator


FLAGS = flags.FLAGS

### Basic options
flags.DEFINE_integer('img_size', 84, 'image size')
flags.DEFINE_integer('device_id', 0, 'GPU device ID to run the job.')
flags.DEFINE_float('gpu_rate', 0.9, 'the parameter for the full_gpu_memory_mode')
flags.DEFINE_string('phase', 'meta', 'pre or meta')
flags.DEFINE_string('exp_log_label', 'experiment_results', 'directory for summaries and checkpoints')
flags.DEFINE_string('logdir_base', './logs/', 'directory for logs')
flags.DEFINE_bool('full_gpu_memory_mode', False, 'in this mode, the code occupies GPU memory in advance')
flags.DEFINE_string('backbone_arch', 'conv4', 'network backbone')

### Pre-train phase options
flags.DEFINE_integer('pre_lr_dropstep', 5000, 'the step number to drop pre_lr')
flags.DEFINE_integer('pretrain_class_num', 64, 'number of classes used in the pre-train phase')
flags.DEFINE_integer('pretrain_batch_size', 64, 'batch_size for the pre-train phase')
flags.DEFINE_integer('pretrain_iterations', 30000, 'number of pretraining iterations.')
flags.DEFINE_integer('pre_sum_step', 10, 'the step number to summary during pretraining')
flags.DEFINE_integer('pre_save_step', 100, 'the step number to save the pretrain model')
flags.DEFINE_integer('pre_print_step', 1000, 'the step number to print the pretrain results')
flags.DEFINE_integer('pre_val_print_step', 500, 'the step number to print the preval results')
flags.DEFINE_float('pre_lr', 0.001, 'the pretrain learning rate')
flags.DEFINE_float('min_pre_lr', 0.0001, 'the pretrain learning rate min')
flags.DEFINE_float('pretrain_dropout_keep', 0.9, 'the dropout keep parameter in the pre-train phase')
flags.DEFINE_string('pretrain_folders', './data/mini-imagenet/train', 'directory for pre-train data')
flags.DEFINE_string('preval_dir', './data/mini-imagenet/val', 'directory for pre-val data')
flags.DEFINE_string('pretrain_label', 'mini_normal', 'additional label for the pre-train log folder')
flags.DEFINE_bool('pre_lr_stop', False, 'whether stop decrease the pre_lr when it is low')

### Meta phase options
flags.DEFINE_integer('way_num', 5, 'number of classes (e.g. 5-way classification)')
flags.DEFINE_integer('shot_num', 1, 'number of examples per class (K for K-shot learning)')
flags.DEFINE_integer('metatrain_epite_sample_num', 15, 'number of meta train episode-test samples')
flags.DEFINE_integer('metatest_epite_sample_num', 0, 'number of meta test episode-test samples, 0 means metatest_epite_sample_num=shot_num')
flags.DEFINE_integer('meta_sum_step', 10, 'the step number to summary during meta-training')
flags.DEFINE_integer('meta_save_step', 500, 'the step number to save the model')
flags.DEFINE_integer('meta_intrain_val_sample', 600, 'the number of samples used for val during meta-train')
flags.DEFINE_integer('meta_print_step', 100, 'the step number to print the meta-train results')
flags.DEFINE_integer('meta_val_print_step', 100, 'the step number to print the meta-val results during meta-training')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of meta-train iterations.')
flags.DEFINE_integer('meta_batch_size', 2, 'number of tasks sampled per meta-update')
flags.DEFINE_integer('train_base_epoch_num', 20, 'number of inner gradient updates during training.')
flags.DEFINE_integer('test_base_epoch_num', 100, 'number of inner gradient updates during test.')
flags.DEFINE_integer('lr_drop_step', 5000, 'the step number to drop meta_lr')
flags.DEFINE_integer('test_iter', 1000, 'iteration to load model')
flags.DEFINE_integer('resume_iter', 0, 'iteration to resume meta-training')
flags.DEFINE_float('resume_lr', 0.001, 'meta-lr to use when training is resumed')
flags.DEFINE_float('meta_lr', 0.001, 'the meta learning rate of the generator')
flags.DEFINE_float('lr_drop_rate', 0.5, 'the step number to drop meta_lr')
flags.DEFINE_float('min_meta_lr', 0.0001, 'the min meta learning rate of the generator')
flags.DEFINE_float('base_lr', 1e-3, 'step size alpha for inner gradient update.')
flags.DEFINE_string('metatrain_dir', './data/mini-imagenet/train', 'directory for meta-train set')
flags.DEFINE_string('metaval_dir', './data/mini-imagenet/val', 'directory for meta-val set')
flags.DEFINE_string('metatest_dir', './data/mini-imagenet/test', 'directory for meta-test set')
flags.DEFINE_string('activation', 'leaky_relu', 'leaky_relu, relu, or None')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('metatrain', True, 'is this the meta-train phase')
flags.DEFINE_bool('base_augmentation', True, 'whether do data augmentation during base learning')
flags.DEFINE_bool('redo_init', True, 're-build the initialization weights')
flags.DEFINE_bool('from_scratch', False, 'start meta-train from scratch, do not use pre-train weights')
flags.DEFINE_bool('proto_maml', False, 'whether to use proto-maml initialization for fc weights')

# Generate experiment key words string
exp_string = 'arch(' + FLAGS.backbone_arch + ')'
exp_string +=  '.cls(' + str(FLAGS.way_num) + ')'
exp_string += '.shot(' + str(FLAGS.shot_num) + ')'
exp_string += '.meta_batch(' + str(FLAGS.meta_batch_size) + ')'
exp_string += '.base_epoch(' + str(FLAGS.train_base_epoch_num) + ')'
exp_string += '.meta_lr(' + str(FLAGS.meta_lr) + ')'
exp_string += '.base_lr(' + str(FLAGS.base_lr) + ')'
exp_string += '.pre_iterations(' + str(FLAGS.pretrain_iterations) + ')'
exp_string += '.pre_dropout(' + str(FLAGS.pretrain_dropout_keep) + ')'
exp_string += '.acti(' + str(FLAGS.activation) + ')'
exp_string += '.lr_drop_step(' + str(FLAGS.lr_drop_step) + ')'
exp_string += '.lr_drop_rate(' + str(FLAGS.lr_drop_rate) + ')'
exp_string += '.pre_label(' + str(FLAGS.pretrain_label) + ')'

if FLAGS.norm == 'batch_norm':
    exp_string += '.norm(batch)'
elif FLAGS.norm == 'layer_norm':
    exp_string += '.norm(layer)'
elif FLAGS.norm == 'None':
    exp_string += '.norm(none)'
else:
    raise Exception('Norm setting is not recognized')

print('Parameters: ' + exp_string)

# Generate pre-train key words string
pre_save_str = 'arch(' + FLAGS.backbone_arch + ')'
pre_save_str += '.pre_lr(' + str(FLAGS.pre_lr) + ')'
pre_save_str += '.pre_lrdrop(' + str(FLAGS.pre_lr_dropstep) + ')'
pre_save_str += '.pre_class(' + str(FLAGS.pretrain_class_num) + ')'
pre_save_str += '.pre_batch(' + str(FLAGS.pretrain_batch_size) + ')'
pre_save_str += '.pre_dropout(' + str(FLAGS.pretrain_dropout_keep) + ')'
if FLAGS.pre_lr_stop:
    pre_save_str += '.pre_lr_stop(True)'
else:
    pre_save_str += '.pre_lr_stop(False)'
pre_save_str += '.pre_label(' + FLAGS.pretrain_label + ')'
pre_string = pre_save_str

# Generate log folders
logdir = FLAGS.logdir_base + FLAGS.exp_log_label
pretrain_dir = FLAGS.logdir_base + 'pretrain_weights'

if not os.path.exists(FLAGS.logdir_base):
    os.mkdir(FLAGS.logdir_base)
if not os.path.exists(logdir):
    os.mkdir(logdir)
if not os.path.exists(pretrain_dir):
    os.mkdir(pretrain_dir)

# If FLAGS.redo_init is true, delete the previous intialization weights.
if FLAGS.redo_init:
    if os.path.exists('./logs/init_weights'):
        os.system('rm -r ./logs/init_weights')
        print('Init weights have been deleted')
    else:
        print('No init weights')

def main():
    tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # Set GPU device id
    print('Using GPU ' + str(FLAGS.device_id))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device_id)
    #os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    # Select pre-train phase or meta-learning phase
    if FLAGS.phase=='pre':
        trainer = PreTrainer(pre_string, pretrain_dir)
    elif FLAGS.phase=='meta':
        trainer = MetaTrainer(exp_string, logdir, pre_string, pretrain_dir)
    else:
        raise Exception('Please set correct phase')           


if __name__ == "__main__":
    main()
