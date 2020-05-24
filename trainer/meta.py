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

""" Trainer for meta-learning. """
import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf

from tqdm import trange
from data_generator.meta_data_generator import MetaDataGenerator
from models.meta_model import MakeMetaModel
from tensorflow.python.platform import flags
from utils.misc import process_batch
from sklearn.metrics import roc_auc_score


FLAGS = flags.FLAGS


class MetaTrainer:
    """The class that contains the code for the meta-train and meta-test."""
    def __init__(self, exp_string, logdir, pre_string, pretrain_dir):
        # Remove the saved datalist for a new experiment
        os.system('rm -r ./logs/processed_data/*')
        self.exp_string = exp_string
        self.logdir = logdir
        self.pre_string = pre_string
        self.pretrain_dir = pretrain_dir
        data_generator = MetaDataGenerator()
        if FLAGS.metatrain:
            # Build model for meta-train phase
            print('Building meta-train model')
            self.model = MakeMetaModel()
            self.model.construct_model()

            print('Meta-train model is built')

            # Start tensorflow session
            self.start_session()
            # Generate data for meta-train phase
            random.seed(5)
            data_generator.generate_data(data_type='train')
            random.seed(7)
            data_generator.generate_data(data_type='test')
            random.seed(9)
            data_generator.generate_data(data_type='val')
        else:
            # Build model for meta-test phase
            print('Building meta-test mdoel')
            self.model = MakeMetaModel()
            self.model.construct_test_model()
            self.model.summ_op = tf.summary.merge_all()
            print('Meta-test model is built')
            # Start tensorflow session
            self.start_session()
            # Generate data for meta-test phase
            random.seed(7)
            data_generator.generate_data(data_type='test')

        # Global initialization and starting queue
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        if FLAGS.metatrain:
            if FLAGS.from_scratch:
                print("No pretrain weights are loaded.")
            elif FLAGS.resume_iter > 0:
                # Load the saved weights of meta-train
                weights = np.load(self.logdir + '/' + exp_string + '/weights_' + str(FLAGS.resume_iter) + '.npy',
                                  allow_pickle=True, encoding="latin1").tolist()
                fc_weights = np.load(self.logdir + '/' + exp_string + '/fc_weights_' + str(FLAGS.resume_iter) + '.npy',
                                     allow_pickle=True, encoding="latin1").tolist()
                inner_lrs = np.load(self.logdir + '/' + exp_string + '/inner_lrs_' + str(FLAGS.resume_iter) + '.npy',
                                    allow_pickle=True, encoding="latin1").tolist()
                bn_vars = np.load(self.logdir + '/' + exp_string + '/bn_vars_' + str(FLAGS.resume_iter) + '.npy',
                                    allow_pickle=True, encoding="latin1").tolist()

                # Assign the weights to the tensorflow variables
                for key in weights.keys():
                    self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                for key in fc_weights.keys():
                    self.sess.run(tf.assign(self.model.fc_weights[key], fc_weights[key]))
                for idx in range(len(inner_lrs)):
                    self.sess.run(tf.assign(self.model.inner_lrs[idx], inner_lrs[idx]))
                for key in bn_vars:
                    self.sess.run(tf.assign(self.model.bn_vars[key], bn_vars[key]))

                print("Resuming meta-training from iteration", FLAGS.resume_iter)

            else:
                # Process initialization weights for meta-train
                init_dir = FLAGS.logdir_base + 'init_weights/'
                if not os.path.exists(init_dir):
                    os.mkdir(init_dir)
                pre_save_str = self.pre_string
                this_init_dir = init_dir + pre_save_str + '.pre_iter(' + str(FLAGS.pretrain_iterations) + ')/'
                if not os.path.exists(this_init_dir):
                    # If there is no saved initialization weights for meta-train, load pre-train model and save initialization weights
                    os.mkdir(this_init_dir)
                    print('Loading pretrain weights')
                    weights_save_dir_base = self.pretrain_dir
                    weights_save_dir = os.path.join(weights_save_dir_base, pre_save_str)
                    weights = np.load(os.path.join(weights_save_dir, "weights_{}.npy".format(FLAGS.pretrain_iterations)),
                        allow_pickle=True, encoding="latin1").tolist()
                    bn_vars = np.load(
                        os.path.join(weights_save_dir, "bn_vars_{}.npy".format(FLAGS.pretrain_iterations)),
                        allow_pickle=True, encoding="latin1").tolist()

                    # Assign pretrained weights to tensorflow variables
                    for key in weights.keys():
                        self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                    for key in bn_vars.keys():
                        for step in range(FLAGS.train_base_epoch_num):
                            self.sess.run(tf.assign(self.model.bn_vars[key + str(step)], bn_vars[key]))
                    print('Pretrain weights loaded, saving init weights')
                    # Load and save init weights for the model
                    new_weights = self.sess.run(self.model.weights)
                    new_bn_vars = self.sess.run(self.model.bn_vars)
                    #fc_weights = self.sess.run(self.model.fc_weights)
                    np.save(this_init_dir + 'weights_init.npy', new_weights)
                    np.save(this_init_dir + 'bn_vars_init.npy', new_bn_vars)
                    #np.save(this_init_dir + 'fc_weights_init.npy', fc_weights)
                else:
                    # If the initialization weights are already generated, load the previous saved ones
                    print('Loading previous saved init weights')
                    weights = np.load(this_init_dir + 'weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                    bn_vars = np.load(this_init_dir + 'bn_vars_init.npy', allow_pickle=True, encoding="latin1").tolist()
                    #fc_weights = np.load(this_init_dir + 'fc_weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                    inner_lrs = np.load(this_init_dir + 'inner_lrs_init.npy', allow_pickle=True, encoding="latin1").tolist()
                    for key in weights.keys():
                        self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                    #for key in fc_weights.keys():
                    #    self.sess.run(tf.assign(self.model.fc_weights[key], fc_weights[key]))
                    for key in bn_vars.keys():
                        self.sess.run(tf.assign(self.model.bn_vars[key], bn_vars[key]))
                    for idx in range(len(inner_lrs)):
                        self.sess.run(tf.assign(self.model.inner_lrs[idx], inner_lrs[idx]))
                    print('Init weights loaded')
        else:
            # Load the saved weights of meta-train
            weights = np.load(self.logdir + '/' + exp_string +  '/weights_' + str(FLAGS.test_iter) + '.npy',
                allow_pickle=True, encoding="latin1").tolist()
            if not FLAGS.proto_maml:
                fc_weights = np.load(self.logdir + '/' + exp_string +  '/fc_weights_' + str(FLAGS.test_iter) + '.npy',
                    allow_pickle=True, encoding="latin1").tolist()
            inner_lrs = np.load(self.logdir + '/' + exp_string + '/inner_lrs_' + str(FLAGS.test_iter) + '.npy',
                                allow_pickle=True, encoding="latin1").tolist()
            bn_vars = np.load(self.logdir + '/' + exp_string + '/bn_vars_' + str(FLAGS.test_iter) + '.npy',
                                allow_pickle=True, encoding="latin1").tolist()
            # Assign the inner learning rates to the tensorflow variables
            for idx in range(len(inner_lrs)):
                self.sess.run(tf.assign(self.model.inner_lrs[idx], inner_lrs[idx]))
            for key in bn_vars:
                self.sess.run(tf.assign(self.model.bn_vars[key], bn_vars[key]))


            # Assign the weights to the tensorflow variables
            for key in weights.keys():
                self.sess.run(tf.assign(self.model.weights[key], weights[key]))
            if not FLAGS.proto_maml:
                for key in fc_weights.keys():
                    self.sess.run(tf.assign(self.model.fc_weights[key], fc_weights[key]))


            print('Weights loaded')
            print('Test iter: ' + str(FLAGS.test_iter))

        if FLAGS.metatrain:
            self.train(data_generator)
        else:
            self.test(data_generator)

    def start_session(self):
        """The function to start tensorflow session."""

        if FLAGS.full_gpu_memory_mode:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
            self.sess = tf.InteractiveSession(config=gpu_config)
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.InteractiveSession(config=config)

    def train(self, data_generator):
        """The function for the meta-train phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Load the experiment setting string from FLAGS 
        exp_string = self.exp_string
        # Generate tensorboard file writer
        train_writer = tf.summary.FileWriter(self.logdir + '/' + exp_string, self.sess.graph)
        print('Start meta-train phase')
        # Generate empty list to record losses and accuracies
        loss_list, acc_list = [], []
        # Load the meta learning rate from FLAGS
        train_lr = FLAGS.meta_lr
        if FLAGS.resume_iter > 0:
            train_lr = FLAGS.resume_lr

        # Load data for meta-train and meta validation
        data_generator.load_data(data_type='train')
        data_generator.load_data(data_type='val')


        for train_idx in trange(FLAGS.metatrain_iterations):
            # Load the episodes for this meta batch
            inputa = []
            labela = []
            inputb = []
            labelb = []
            for meta_batch_idx in range(FLAGS.meta_batch_size):
                this_episode = data_generator.load_episode(index=train_idx*FLAGS.meta_batch_size+meta_batch_idx, data_type='train')
                inputa.append(this_episode[0])
                labela.append(this_episode[1])
                inputb.append(this_episode[2])
                labelb.append(this_episode[3])
            inputa = np.array(inputa)
            labela = np.array(labela)
            inputb = np.array(inputb)
            labelb = np.array(labelb)

            # Generate feed dict for the tensorflow graph
            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb,
                self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: train_lr}

            # Set the variables to load from the tensorflow graph
            input_tensors = [self.model.metatrain_op] # The meta train optimizer
            input_tensors.extend([self.model.total_loss]) # The meta train loss
            input_tensors.extend([self.model.total_accuracy]) # The meta train accuracy
            input_tensors.extend([self.model.training_summ_op]) # The tensorboard summary operation

            # run this meta-train iteration
            result = self.sess.run(input_tensors, feed_dict)

            # record losses, accuracies and tensorboard
            loss_list.append(result[1])
            acc_list.append(result[2])
            train_writer.add_summary(result[3], train_idx)

            # print meta-train information on the screen after several iterations
            if (train_idx!=0) and train_idx % FLAGS.meta_print_step == 0:
                print_str = 'Iteration:' + str(train_idx)
                print_str += ' Loss:' + str(np.mean(loss_list)) + ' Acc:' + str(np.mean(acc_list))
                print(print_str)
                loss_list, acc_list = [], []

            # Save the model during meta-train
            if train_idx != 0 and train_idx % FLAGS.meta_save_step == 0:
                weights = self.sess.run(self.model.weights)
                if not FLAGS.proto_maml:
                    fc_weights = self.sess.run(self.model.fc_weights)
                inner_lrs = self.sess.run(self.model.inner_lrs)
                bn_vars = self.sess.run(self.model.bn_vars)
                np.save(self.logdir + '/' + exp_string + '/weights_'    + str(train_idx + FLAGS.resume_iter) + '.npy', weights)
                if not FLAGS.proto_maml:
                    np.save(self.logdir + '/' + exp_string + '/fc_weights_' + str(train_idx + FLAGS.resume_iter) + '.npy', fc_weights)
                np.save(self.logdir + '/' + exp_string + '/inner_lrs_'  + str(train_idx + FLAGS.resume_iter) + '.npy', inner_lrs)
                np.save(self.logdir + '/' + exp_string + '/bn_vars_'    + str(train_idx + FLAGS.resume_iter) + '.npy', bn_vars)

            # Run the meta-validation during meta-train
            if train_idx != 0 and train_idx % FLAGS.meta_val_print_step == 0:
                test_loss = []
                test_accs = []
                test_aucs = []
                for test_itr in range(FLAGS.meta_intrain_val_sample):
                    this_episode = data_generator.load_episode(index=test_itr, data_type='val')
                    test_inputa = this_episode[0][np.newaxis, :]
                    test_labela = this_episode[1][np.newaxis, :]
                    test_inputb = this_episode[2][np.newaxis, :]
                    test_labelb = this_episode[3][np.newaxis, :]

                    test_feed_dict = {self.model.inputa: test_inputa, self.model.inputb: test_inputb,
                        self.model.labela: test_labela, self.model.labelb: test_labelb,
                        self.model.meta_lr: 0.0}
                    test_input_tensors = [self.model.total_loss, self.model.total_accuracy, self.model.softmax_probs]
                    test_result = self.sess.run(test_input_tensors, test_feed_dict)
                    test_loss.append(test_result[0])
                    test_accs.append(test_result[1])

                    test_aucs.append(roc_auc_score(test_labelb[0], test_result[2][0]))


                valsum_feed_dict = {self.model.input_val_loss: \
                    np.mean(test_loss)*np.float(FLAGS.meta_batch_size)/np.float(FLAGS.shot_num),
                    self.model.input_val_acc: np.mean(test_accs)*np.float(FLAGS.meta_batch_size)}
                valsum = self.sess.run(self.model.val_summ_op, valsum_feed_dict)
                train_writer.add_summary(valsum, train_idx)
                print_str = '[***] Val Loss:' + str(np.mean(test_loss)*FLAGS.meta_batch_size) + \
                    ' Val Acc:' + str(np.mean(test_accs)*FLAGS.meta_batch_size) + \
                    ' Val Auc:' + str(np.mean(test_aucs))
                print(print_str)

            # Reduce the meta learning rate to half after several iterations
            if (train_idx!=0) and train_idx % FLAGS.lr_drop_step == 0 and train_lr > FLAGS.min_meta_lr:
                train_lr = train_lr * FLAGS.lr_drop_rate
                if train_lr < FLAGS.min_meta_lr:
                    train_lr = FLAGS.min_meta_lr
                print('Train LR: {}'.format(train_lr))

        # Save the final model
        weights = self.sess.run(self.model.weights)
        if not FLAGS.proto_maml:
            fc_weights = self.sess.run(self.model.fc_weights)
        inner_lrs = self.sess.run(self.model.inner_lrs)
        bn_vars = self.sess.run(self.model.bn_vars)
        np.save(self.logdir + '/' + exp_string + '/weights_'    + str(train_idx+1 + FLAGS.resume_iter) + '.npy', weights)
        if not FLAGS.proto_maml:
            np.save(self.logdir + '/' + exp_string + '/fc_weights_' + str(train_idx+1 + FLAGS.resume_iter) + '.npy', fc_weights)
        np.save(self.logdir + '/' + exp_string + '/inner_lrs_'  + str(train_idx+1 + FLAGS.resume_iter) + '.npy', inner_lrs)
        np.save(self.logdir + '/' + exp_string + '/bn_vars_'    + str(train_idx+1 + FLAGS.resume_iter) + '.npy', bn_vars)

    def test(self, data_generator):
        """The function for the meta-test phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Set meta-test episode number
        NUM_TEST_POINTS = 600
        # Load the experiment setting string from FLAGS
        exp_string = self.exp_string
        print('Start meta-test phase')
        np.random.seed(1)
        # Generate empty list to record accuracies
        metaval_accuracies = []
        metaval_aucs = []
        # Load data for meta-test
        data_generator.load_data(data_type='test')
        for test_idx in trange(NUM_TEST_POINTS):
            # Load one episode for meta-test
            this_episode = data_generator.load_episode(index=test_idx, data_type='test')
            inputa = this_episode[0][np.newaxis, :]
            labela = this_episode[1][np.newaxis, :]
            inputb = this_episode[2][np.newaxis, :]
            labelb = this_episode[3][np.newaxis, :]
            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb,
                self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: 0.0}
            result = self.sess.run([self.model.metaval_total_accuracy, self.model.metaval_softmax_probs],
                                   feed_dict)
            metaval_accuracies.append(result[0])
            metaval_aucs.append(roc_auc_score(labelb[0], result[1][0]))

        # Calculate the mean accuracies and the confidence intervals
        #metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies)
        stds = np.std(metaval_accuracies)
        ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

        # Print the meta-test results
        print('Test accuracies and confidence intervals')
        print((means, ci95))

        # Calculate the mean auc score and the confidence interval
        auc_mean = np.mean(metaval_aucs)
        std = np.std(metaval_aucs)
        ci95 = 1.96 * std / np.sqrt(NUM_TEST_POINTS)
        print('Test AUC')
        print(auc_mean, " +- ", ci95)


        '''
        # Save the meta-test results in the csv files
        if not FLAGS.load_saved_weights:
            out_filename = self.logdir +'/'+ exp_string + '/' + 'result_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.csv'
            out_pkl = self.logdir +'/'+ exp_string + '/' + 'result_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.pkl'
            with open(out_pkl, 'wb') as f:
                pickle.dump({'mses': metaval_accuracies}, f)
            with open(out_filename, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['update'+str(i) for i in range(len(means))])
                writer.writerow(means)
                writer.writerow(stds)
                writer.writerow(ci95)
        '''

