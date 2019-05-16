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
                