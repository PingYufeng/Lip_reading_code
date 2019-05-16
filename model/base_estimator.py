import tensorflow as tf


class BaseEstimator(object):
    '''
    Base estimator.

    Input:
        model_parms: Dict. parameters to build model_fn
        run_config: tf.estimator.RunConfig. config for estimator
    '''

    def __init__(self, model_parms, run_config, **kwargs):
        super(BaseEstimator, self).__init__()
        self.model_parms = model_parms
        self.run_config = run_config
        self.estimator = tf.estimator.Estimator(
            self.model_fn, params = self.model_parms, config = self.run_config)
    

    def train_and_evaluate(self, train_input_fn, eval_input_fn,
                           max_steps=10000000,
                           eval_steps=100,
                           throttle_secs=200):
        '''
        Args:
            train_input_fn: Input fn for Train.
            eval_input_fn: Input fn for Evaluation.
        
        Kwargs:
            max_steps: Max training steps.
            eval_steps: Steps to evaluate.
            throttle_secs: Evaluate interval.
        '''
        train_spec = tf.estimator.TrainSpec(
            input_fn = train_input_fn,
            max_steps=max_steps,
            hooks=None)
        eval_spec = tf.estimator.EvalSpec(
            input_fn = eval_input_fn,
            throttle_secs=throttle_secs,
            steps=eval_steps)
        # TODO:
        #   What is tf.estimator.TrainSpec's param hooks?
        #   What is tf.estimator.EvalSpec's param throttle_secs and steps?

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    
    def evaluate(self, eval_input_fn, steps=None, checkpoint_path=None):
        '''
        Args:
            eval_input_fn: Input fn for Evaluation.
        
        Kwargs:
            steps: Evaluate steps
            checkpoint_path: Checkpoint to evaluate
        
        Returns:
            Evaluate result: tf.estimator.Estimator.evaluate
        '''
        return self.estimator.evaluate(
            eval_input_fn, steps=steps, checkpoint_path=checkpoint_path)
    

    def predict(self, predict_input_fn, checkpoint_path=None):
        '''
        Args:
            predict_input_fn: Input fn for Predict.
        '''
        predictions = self.estimator.predict(
            predict_input_fn, checkpoint_path=checkpoint_path)
        for prediction in predictions:
            print(prediction)
    

    def model_fn(self, features, labels, mode, params):
        '''
        The model_fn of estimator.

        Args:
            features: 5-D Tensor. Videos of shape (batch_size, T, H, W, C)
            labels: 1-D Tensor. Labels of shape (batch_size, )
            mode: tf.estimator.ModeKeys.<PREDICT/TRAIN/EVAL>
            params: Dict. Parameters of the estimator (model_parms).
        
        Return:
            tf.estimator.EstimatorSpec
        '''

        raise NotImplementedError('Model Function not Implemented')

    
    



