#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: cifar10_TFRecord.py
#   author: Bolun Wu
#   email: bowenwu@sjtu.edu.cn
#   created date: 2019/7/20
#   description: train and evaluate alexnet on cifar10
#
#================================================================

import argparse
import os
import sys

import tensorflow as tf

from dataset.cifar10_TFRecord import cifar10_tfrecord_input_fn
from model.alex_estimator import AlexEstimator

# sys.path.append('D:\Bowen\SJTU\实验室\tensorflow_learning')

def arg_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='either train, eval, predict')
    parser.add_argument(
        '--data_dir', default='./data/cifar10',
        help='Directory where TFRecord files are stored')
    parser.add_argument(
        '--model_dir', default='./ckpts',
        help='Directory where checkpoints are stored')
    parser.add_argument(
        '--save_steps', type=int, default=100,
        help='steps interval to save checkpoint')
    parser.add_argument(
        '--eval_steps', type=int, default=100,
        help='steps to evaluate')
    parser.add_argument('--gpu', help='gpu id to use', default='')
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    tf.logging.set_verbosity(tf.logging.INFO)

    multi_gpu = (len(args.gpu.split(',')) > 1)

    run_config = AlexEstimator.get_runConfig(
        model_dir=args.model_dir,
        save_checkpoints_steps=args.save_steps,
        multi_gpu=multi_gpu,
        keep_checkpoint_max=100)

    train_batch_size = 256
    eval_batch_size = 64

    train_input_fn = cifar10_tfrecord_input_fn(
        filenames=os.path.join(args.data_dir, 'cifar10_train.tfrecord'),
        batch_size=train_batch_size)
    
    eval_input_fn = cifar10_tfrecord_input_fn(
        filenames=os.path.join(args.data_dir, 'cifar10_eval.tfrecord'),
        batch_size=eval_batch_size)

    # Create a custom estimator using model_fn to define the model
    model_params = {
        'dropout_rate':0.5,
        'num_classes':10}
    
    model = AlexEstimator(model_params, run_config)

    if args.mode == 'train':
        model.train_and_evaluate(
            train_input_fn, eval_input_fn, eval_steps=args.eval_steps)
    elif args.mode == 'eval':
        res = model.evaluate(
            eval_input_fn, steps=args.eval_steps, checkpoint_path=args.model_dir)
        print(res)
    elif args.mode == 'predict':
        model.predict(eval_input_fn, checkpoint_path=args.model_dir)
    else:
        raise ValueError(
            'arg mode should be one of "train", "eval", "predict"')
        

if __name__ == '__main__':
    main()

    