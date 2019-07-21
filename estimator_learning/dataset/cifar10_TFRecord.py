#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: cifar10_TFRecord.py
#   author: Bolun Wu
#   email: bowenwu@sjtu.edu.cn
#   created date: 2019/7/20
#   description: create TFRecord for CIFAR10 dataset and input_fn
#
#================================================================

import os

import numpy as np
import tensorflow as tf
from PIL import Image


def _cifar10_unpickle(file):
    '''Args:
        file - a cifar10 dataset file download from the official website
    '''
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _create_TFRecord(input_files, output_file, mode):
    '''Args
        input_files - a list of names of dataset files
        output_file - the name of output tfrecord file
        mode - train or eval
    '''

    with tf.python_io.TFRecordWriter(output_file + '_' + mode + '.tfrecord') as writer:
        for input_file in input_files:

            dict = _cifar10_unpickle(input_file) # dict: ['batch_label', 'labels', 'data', 'filenames']

            # get images
            images = dict.get(b'data')   # (10000, 3072)
            num_images = images.shape[0]
            
            # get labels
            labels = dict.get(b'labels') # (10000, 1)

            for i in range(num_images):

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image':_bytes_feature(images[i].tobytes()),
                    'label':_int64_feature(labels[i])
                }))

                writer.write(example.SerializeToString())


def create_TFRecord(path):
    '''Args:
        path - the root path of data file
    '''
    cifar10_data = {
        'train': [os.path.join(path, 'data_batch_{}'.format(i)) for i in range(1, 6)],
        'eval': [os.path.join(path, 'test_batch')]}
    
    for key in cifar10_data:
        _create_TFRecord(input_files=cifar10_data[key], 
                         output_file=os.path.join(path, 'cifar10'),
                         mode=key)
        print('{} data stored in {}'.format(
            key, 
            os.path.join(path, 'cifar10_{}.tfrecord'.format(key))))


def cifar10_tfrecord_input_fn(filenames, batch_size=1000, shuffle=True):

    def _parser(serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([3*32*32])

        image = tf.cast(
            tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        return image, tf.one_hot(label, depth=10)
    
    def _input_fn():
        dataset = tf.data.TFRecordDataset(filenames)
        print(dataset)
        dataset = dataset.map(_parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.repeat(None)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        print(tf.shape(features))
        print(tf.shape(labels))
        return features, labels
    
    return _input_fn


if __name__ == '__main__':
    CIFAR10_PATH = 'data\cifar10'
    create_TFRecord(CIFAR10_PATH)
    print('CIFAR10_TFRecord has been created.')

