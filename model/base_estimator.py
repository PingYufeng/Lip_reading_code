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



