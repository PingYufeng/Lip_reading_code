# Tensorflow Estimator

### Introduction
The most basic way of building a neural network training framework is to use tf.Estimator object. You need to define a model function that defines a loss function, a train op, one or a set of predictions, and optionally a set of metric ops for evaluation.

```python
import tensorflow as tf
def model_fn(features, labels, mode, params):
    
    # 1. Prediction mode
    predictions = ...
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictinos)
    
    # 2. Evaluation mode
    loss = ...
    metric_ops = ...
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss,
            eval_metric_ops=metric_ops)
    
    # 3. Training mode
    assert mode == tf.estimator.ModeKeys.TRAIN
    train_op = ...
    logging_hook = tf.train.LoggingTensorHook(
        {'Accuracy':...}, every_n_iter=100)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss,
        train_op=train_op, 
        training_hoooks=[logging_hook])

params = ...
run_config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_checkpoints_steps=save_checkpoints_steps)
estimator = tf.estimator.Estimator(
    model_fn=model_fn, config=run_config, params=params)
```

To train and evaluate the model, call **tf.estimator.train_and_evaluate()**:
```python
# 1. Train and Evaluate
train_spec = tf.estimator.TrainSpec(
    input_fn=train_input_fn, max_steps=max_steps)
eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input_fn, throttle_secs=throttle_secs,
    steps=eval_steps)
tf.estimator.train_and_evaluate(
    estimator=estimator,
    train_spec=train_spec,
    eval_spec=eval_spec)
```

To evaluate the model, simply call **Estimator.evaluate()**
```python
# 2. Evaluate
estimator.evaluate(
    input_fn=eval_input_fn,
    steps=steps, 
    checkpoint_path=checkpoint_path)
```

To make prediction for new samples, call **Estimator.predict()**
```python
# 3. Predict
estimator.predict(
    input_fn=predict_input_fn,
    checkpoint_path=checkpoint_path)
```

The input function returns two tensors (or dictionaries of tensors) providing the features and labels to be passed to the model:
```python
def input_fn():
    features = ...
    labels = ...
    return features, labels
```

### Code Structure
```
|--estimator_learning
|----model              CNN models. New models should inherit from `base_estimator.BaseEstimator`.
|----dataset            Scripts for creating Cifar TFRecord.
```

A BaseEstimator is defined in *model/base_estimator.py*, which contains useful functions when building your framework such as **train_and_eval()**. Then the AlexEstimator inherit from BaseEstimator is defined in *model/alex_estimator.py*, which contrains **model_fn()** described above.

### Usage

```
$python AlexNet_cifar10.py [-h] [--mode] [--data_dir] [--model_dir]
                           [--save_steps] [--eval_steps] [--gpu]
arguments: 
    -h, --help          show help message
    --data_dir          directory where TFRecord files are stored
    --model_dir         directory where checkpoints are stored
    --save_steps        steps interval to save checkpoints
    --eval_steps        steps to evaluate
    --gpu               gpu id to use
```


