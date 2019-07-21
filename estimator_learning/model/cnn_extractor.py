#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================
#   Wish you good luck.
#
#   file name: cnn_extractor.py
#   author: Bowen
#   email: bowenwu@sjtu.edu.cn
#   created date: 2019/07/020
#
#=================================================================

import tensorflow as tf


class CNN(object):
    '''base cnn model. Extract feature of a image'''

    def __init__(self, scope='cnn_feature_extractor'):
        self.scope = scope
    
    def build():
        raise NotImplementedError('CNN not Implemented.')


class AlexNet(CNN):
    '''AlexNet cnn feature extractor'''

    def __init__(self, *args, **kwargs):
        super(AlexNet, self).__init__(*args, **kwargs)
    
    def build(self, image_tensor):

        with tf.variable_scope(self.scope):

            self.conv1 = tf.layers.Conv2D(64, kernel_size=11, strides=4, 
                padding='same', name='conv1')(image_tensor)
            self.relu1 = tf.keras.layers.Activation('relu', name='relu1')(self.conv1)
            self.maxp1 = tf.layers.MaxPooling2D(pool_size=3, 
                strides=2, name='maxp1')(self.relu1)

            self.conv2 = tf.layers.Conv2D(192, kernel_size=5, strides=1,
                padding='valid', name='conv2')(self.maxp1)
            self.relu2 = tf.keras.layers.Activation('relu', name='relu2')(self.conv2)
            self.maxp2 = tf.layers.MaxPooling2D(pool_size=3,
                strides=2, name='maxp2')(self.relu2)
            
            self.conv3 = tf.layers.Conv2D(384, kernel_size=3, strides=1,
                padding='valid', name='conv3')(self.maxp2)
            self.relu3 = tf.keras.layers.Activation('relu', name='relu3')(self.conv3)
            
            self.conv4 = tf.layers.Conv2D(256, kernel_size=3, strides=1,
                padding='valid', name='conv4')(self.relu3)
            self.relu4 = tf.keras.layers.Activation('relu', name='relu4')(self.conv4)

            self.conv5 = tf.layers.Conv2D(256, kernel_size=3, strides=1,
                padding='valid', name='conv5')(self.relu4)
            self.relu5 = tf.keras.layers.Activation('relu', name='relu5')(self.conv5)
            self.maxp3 = tf.layers.MaxPooling2D(pool_size=3,
                strides=2, name='maxp3')(self.relu5)
            
            return self.maxp3
