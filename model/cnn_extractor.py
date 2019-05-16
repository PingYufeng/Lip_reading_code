#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: cnn_extractor.py
#   author: Bolun Wu
#   email: bowenwu@sjtu.edu.cn
#   created date: 2019/5/15
#   description:
#
#================================================================

import tensorflow as tf


class CNN(object):
    '''
    Base cnn model. Extract feature of the video tensor.

    Input:
        video tensor: Tensor of shape (batch_size, T, H, W, C).
        In GRID, H = 50 and W = 100.
    
    Output:
        Tensor of shape (T, feature_len)
    '''

    def __init__(self, feature_len, training, scope='cnn_extractor'):
        self.feature_len = feature_len
        self.training = training
        self.scope = scope
    

    def build():
        raise NotImplementedError('CNN not Implemented.')
    

class LipNet(CNN):
    '''
    LipNet cnn extractor.

    Reference
        Paper: <LipNet: End-to-End Sentence-level Lipreading>
        Url: <https://arxiv.org/abs/1611.01599>
    '''

    def __init__(self, *args, **kwargs):
        super(LipNet, self).__init__(*args, **kwargs)
    

    def build(self, video_tensor):
        '''
        Input:
            video_tensor: Tensor of shape (batch_size, T, H, W, C).
            In GRID, H = 50 and W = 100.
        
        Output:
            Tensor of shape (T, feature_len)
        '''
        with tf.variable_scope(self.scope):

            self.zero1 = tf.keras.layers.ZeroPadding3D(
                padding=(1, 2, 2), name='zero1')(video_tensor)
            self.conv1 = tf.layers.Conv3D(
                32, (3, 5, 5), strides=(1, 2, 2),
                kernel_initializer='he_normal', name='conv1')(self.zero1)
            self.batch1 = tf.layers.batch_normalization(
                self.conv1, training=self.training, name='batch1')
            self.actv1 = tf.keras.layers.Activation(
                'relu', name='actv1')(self.batch1)
            self.drop1 = tf.keras.layers.SpatialDropout3D(0.5)(self.actv1)
            self.maxpool1 = tf.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='maxp1')(self.drop1)
            
            self.zero2 = tf.keras.layers.ZeroPadding3D(
                padding=(1, 2, 2), name='zero2')(self.maxpool1)
            self.conv2 = tf.layers.Conv3D(
                64, (3, 5, 5), strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv2')(self.zero2)
            self.batch2 = tf.layers.batch_normalization(
                self.conv2, training=self.training, name='batch2')
            self.actv2 = tf.keras.layers.Activation(
                'relu', name='actv2')(self.batch2)
            self.drop2 = tf.keras.layers.SpatialDropout3D(0.5)(self.actv2)
            self.maxpool2 = tf.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='maxp2')(self.drop2)
            
            self.zero3 = tf.keras.layers.ZeroPadding3D(
                padding=(1, 1, 1), name='zero3')(self.maxpool2)
            self.conv3 = tf.layerss.Conv3D(
                96, (3, 3, 3), strides=(1, 1, 1),
                kernel_initializer='he_normal', name='conv3')(self.zero3)
            self.batch3 = tf.layers.batch_normalization(
                self.conv3, training=self.training, name='batch3')
            self.actv3 = tf.keras.layers.Activation(
                'relu', name='actv3')(self.batch3)
            self.drop3 = tf.keras.layers.SpatialDropout3D(0.5)(self.actv3)
            self.maxp3 = tf.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='maxp3')(self.drop3)

            self.conv4 = tf.layers.Conv3D(
                self.feature_len, (1, 1, 1), strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv4')(self.maxp3)
            self.output = tf.keras.layers.TimeDistributed(
                tf.keras.layers.GlobalMaxPooling2D(name='global_maxp1'),
                name='timeDistributed1')(self.conv4) # shape: (T, feature_len)
            
            return self.output

