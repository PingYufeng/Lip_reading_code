import tensorflow as tf

from ..utils import *
from .cnn_extractor import LipNet
from .base_estimator import BaseEstimator


class CtcEstimator(BaseEstimator):
    '''
    LipNet model: crnn + ctc

    Reference
        Paper: <LipNet: End-to-End Sentence-level Lipreading>
        Url: <https://arxiv.org/abs/1611.01599>
    '''

    def __init__(self, model_parms, run_config, **kwargs):
        super(CtcEstimator, Self).__init__(model_parms, run_config, **kwargs)
    

    def model_fn(self, features, labels, mode, params):
        '''
        Args:
            features: 5-D Tensor. Shape of (batch_size, T, H, W, C)
            labels: 1-D Tensor. Shape of (batch_size,)
            mode: tf.estimator.ModeKeys.<PREDICT/TRAIN/EVAL>
            params: Dict. Parameters of the Estimator.
        
        Returns:
            tf.estimator.EstimatorSpec
        '''

        feature_len = params.get('feature_len', 256)
        gru_layer_num = params.get('gru_layer_num', 2)
        gru_units = params.get('gru_unit', 256)
        beam_width = params.get('beam_width', 4)
        learning_rate = params.get('learning_rate', 0.001)
        use_tcn = params.get('use_tcn', False)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        video = features['video']
        lipnet_extractor = LipNet(
            feature_len=feature_len,
            training=is_training,
            scope='cnn_extractor')
        output_size = params.get('output_size', 28)
        net = lipnet_extractor.build(video_tensor=video) # (B, T, feature_len)

        # rnn
        if use_tcn:
            # TODO:
            # What is TCN ?
            # How to implement ?
            pass
        else:
            with tf.variable_scope('brnn'):
                for i in range(gru_layer_num):
                    net = tf.keras.layers.Bidirectional(
                        tf.keras.layers.GRU(
                            gru_units,
                            return_sequences=True,
                            kernel_initializer='Orthogonal',
                            name='gru_{}'.format(i)),
                        merge_mode='concat',
                        name='gru_concat_{}'.format(i))(net)
        
        # linear
        with tf.variable_scope('fc1'):
            logits = tf.layers.Dense(
                units=output_size, kernal_initializer='he_normal',
                name='dense1')(net) # (batch_size, T, output_size)
            logits = tf.transpose(logits, (1, 0, 2)) # (T, batch_size, output_size)
        
        # decode logits
        batch_size = tf.expand_dims(tf.shape(video)[0], 0) # [batch_size]
        input_length = tf.expand_dims(tf.shape(video)[1], 0) # [input_length]
        sequence_length = tf.tile(input_length, batch_size) # [input_length, input_length, ...]
        decoded, log_probs = tf.nn.ctc_beam_search_decoder(
            logits,
            sequence_length,
            beam_width=beam_width,
            merge_repeated=False)
        
        predictions = decoded[0]

        predicted_char_list = indices2string(predictions)
        predicted_string = char_list2string(
            predicted_char_list) # ['ab', 'abc]
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            predict_output = {
                'predictions': tf.sparse_tensor_to_dense(predictions, default_value=-1),
                'predicted_string': predicted_string
            }
            export_output = {
                'prediction': tf.estimator.export.PredictOutput(predict_output)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predict_output,
                export_outputs=export_output)

