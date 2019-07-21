#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================
#   Wish you good luck.
#
#   file name: base_estimator.py
#   author: Bowen
#   email: bowenwu@sjtu.edu.cn
#   created date: 2019/07/020
#
#=================================================================

import tensorflow as tf

class BaseEstimator(object):
    '''base estimator for classification

    Args:
        model_params: Dict. parameters to build model_fn
        run_config: RunConfig. config for Estimator
    '''

    def __init__(self, model_params, run_config, **kwargs):
        super(BaseEstimator, self).__init__()
        self.model_params = model_params
        self.run_config = run_config
        self.estimator = tf.estimator.Estimator(
            model_fn = self.model_fn, 
            params=self.model_params, 
            config=self.run_config, 
            **kwargs)
    
    def train_and_evaluate(self, 
                           train_input_fn, 
                           eval_input_fn,
                           max_steps=1000000,
                           eval_steps=100,
                           throttle_secs=200):
        '''train and eval

        Args:
            train_input_fn: Input function for Train
            eval_input_fn: Input function for Evaluation
        
        Kwargs:
            max_steps: Max training steps
            eval_steps: Steps to evaluate
            throttle_secs: Evaluate interval. 
            # evaluation will perform only when new checkpoints exists
        '''

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn, 
            throttle_secs=throttle_secs, 
            steps=eval_steps)

        tf.estimator.train_and_evaluate(
            estimator=self.estimator, 
            train_spec=train_spec,
            eval_spec=eval_spec)
    
    def evaluate(self, eval_input_fn, steps=None, checkpoint_path=None):
        '''evaluate and print

        Args:
            eval_input_fn: Input function for Eval
        Kwargs:
            steps: Evaluate steps
            checkpoint_path: Checkpoint to evaluate
        Returns: Evaluate results
        '''
        return self.estimator.evaluate(
            input_fn=eval_input_fn, steps=steps, checkpoint_path=checkpoint_path)
        # checkpoint_path: Path of a specific checkpoint to evaluate. If `None`, the
        # latest checkpoint in `model_dir` is used.
    
    def predict(self, predict_input_fn, checkpoint_path=None):
        '''predict new examples
        Args:
            predict_input_fn: Input function for prediction
        '''
        id_to_classes = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat',4 :'dear',
                         5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
        predictions = self.estimator.predict(
            input_fn=predict_input_fn, checkpoint_path=checkpoint_path)
        for prediction in predictions:
            print(id_to_classes.get(prediction))
    
    def model_fn(self, features, labels, mode, params):
        '''model_fn of the estimator
        Returns: EstimatorSpec
        '''
        raise NotImplementedError('model function is not implemented')
    
    def cal_metrics(self, labels, logits):
        ''' calculate accuracy
        Returns:
            metric_dict {'metric_op', 'metric_value'}
            'metric_op': used in eval_metric_ops
            'metirc_value': used in tf.summary.scalar and tf.train.LoggingTensorHook
        '''
        correct_prediction = tf.equal(
            tf.argmax(labels, axis=1), 
            tf.argmax(logits, axis=1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

        metric_dict = {
            'metric_op': tf.metrics.mean(correct_prediction),
            'metric_value': tf.reduce_mean(correct_prediction)}
        return metric_dict
    
    @staticmethod
    def get_runConfig(model_dir,
                      save_checkpoints_steps,
                      multi_gpu = False,
                      keep_checkpoint_max=100):
        ''' get RunConfig for Estimator
        Args:
            model_dir: The directory to save and load checkpoints
            save_checkpoints_steps: Step intervals to save checkpoints
            keep_checkpoint_max: The max checkpoints to keep.
        Returns: tf.estimator.RunConfig
        '''
        sess_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        if multi_gpu:
            distribution = tf.contrib.distribute.MirroredStrategy()
        else:
            distribution = None
        return tf.estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=100,
            train_distribute=distribution,
            session_config=sess_config)

