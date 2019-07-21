#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================
#   Wish you good luck.
#
#   file name: alex_estimator.py
#   author: Bowen
#   email: bowenwu@sjtu.edu.cn
#   created date: 2019/07/020
#
#=================================================================

import tensorflow as tf

from .base_estimator import BaseEstimator
from .cnn_extractor import AlexNet


class AlexEstimator(BaseEstimator):
    '''AlexNet model'''

    def __init__(self, model_params, run_config, **kwargs):
        super(AlexEstimator, self).__init__(model_params, run_config, **kwargs)

    def model_fn(self, features, labels, mode, params):
        dropout_rate = params.get('dropout_rate')
        num_classes = params.get('num_classes')
        use_decayed_lr = params.get('use_decayed_lr') # True or False

        feature_extractor = AlexNet()

        # Cnn
        feature_cnn = feature_extractor.build(image_tensor=features)

        # Dense
        with tf.variable_scope('fc_layers'):

            fc_shape = feature_cnn.shape.as_list()
            flatten_len = fc_shape[1] * fc_shape[2] * fc_shape[3]
            reshape = tf.reshape(feature_cnn, [-1, flatten_len])

            drop1 = tf.layers.Dropout(dropout_rate)(reshape)
            dense1 = tf.layers.Dense(4096, 
                activation=tf.nn.relu, name='dense1')(drop1)
        
            drop2 = tf.layers.Dropout(dropout_rate)(dense1)
            dense2 = tf.layers.Dense(4096,
                activation=tf.nn.relu, name='dense2')(drop2)
        
            logits = tf.layers.Dense(num_classes, name='logits')(dense2)
        
        # predictions
        predictions = {'class_ids': tf.argmax(logits, axis=1)}

        # 1. Prediction mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.argmax(labels, axis=1), logits=logits)
        
        metric_dict = self.cal_metrics(labels, logits)

        # 2. Evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, 
                eval_metric_ops={'Accuracy': metric_dict.get('metric_op')})
        
        assert mode == tf.estimator.ModeKeys.TRAIN, 'TRAIN is the only ModeKey left.'

        if use_decayed_lr:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                learning_rate=0.001, 
                global_step=global_step,
                decay_steps=2000,
                decay_rate=0.85,
                staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        
        tf.summary.scalar('Accuracy', metric_dict.get('metric_value'))
        logging_hook = tf.train.LoggingTensorHook({'Accuracy': metric_dict.get('metric_value')}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
            train_op=train_op, training_hooks=[logging_hook])


