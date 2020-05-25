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

""" Data generator for meta-learning. """
import numpy as np
import os
import random
import tensorflow as tf

from tqdm import trange
from tensorflow.python.platform import flags
from utils.misc import get_images, process_batch, process_batch_augmentation

FLAGS = flags.FLAGS

class MetaDataGenerator(object):
    """The class to generate data lists and episodes for meta-train and meta-test."""
    def __init__(self):
        pass

    def generate_data(self, data_type='train'):
        """The function to generate the data lists.
        Arg:
          data_type: the phase for meta-learning.
        """
        if data_type=='train':
            metatrain_folder = FLAGS.metatrain_dir
            folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            num_total_batches = FLAGS.metatrain_iterations * FLAGS.meta_batch_size + 10
            num_samples_per_class = FLAGS.shot_num + FLAGS.metatrain_epite_sample_num
            batch_size = FLAGS.meta_batch_size

        elif data_type=='test':          
            metatest_folder = FLAGS.metatest_dir
            folders = [os.path.join(metatest_folder, label) \
                for label in os.listdir(metatest_folder) \
                if os.path.isdir(os.path.join(metatest_folder, label)) \
                ]
            num_total_batches = 600
            if FLAGS.metatest_epite_sample_num==0:
                num_samples_per_class = FLAGS.shot_num*2
            else:
                num_samples_per_class = FLAGS.shot_num + FLAGS.metatest_epite_sample_num
            batch_size = 1
        elif data_type=='val':
            metaval_folder = FLAGS.metaval_dir
            folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            num_total_batches = 600
            if FLAGS.metatest_epite_sample_num==0:
                num_samples_per_class = FLAGS.shot_num*2
            else:
                num_samples_per_class = FLAGS.shot_num + FLAGS.metatest_epite_sample_num
            batch_size = 1
        else:
            raise Exception('Please check data list type')

        print('Generating ' + data_type + ' data')
        data_list = []
        for epi_idx in trange(num_total_batches):
            sampled_character_folders = random.sample(folders, FLAGS.way_num)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders,
                range(FLAGS.way_num), nb_samples=num_samples_per_class)
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            data_list.extend(filenames)


        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(data_list), shuffle=False)
        
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file, channels=3)
        image.set_shape((FLAGS.img_size, FLAGS.img_size, 3))
        image = tf.reshape(image, [(FLAGS.img_size ** 2) * 3])
        image = tf.cast(image, tf.float32) / 255.0
    
        num_preprocess_threads = 4
        min_queue_examples = 256
        examples_per_batch = FLAGS.way_num * num_samples_per_class
        batch_image_size = batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
                
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(num_samples_per_class):
                class_idxs = tf.range(0, FLAGS.way_num)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))
            
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, FLAGS.way_num)
        return all_image_batches, all_label_batches
