import string

import tensorflow as tf

from ..utils import *
from .base_estimator import BaseEstimator
from .cnn_extractor import LipNet


class seq2seqAttention(BaseEstimator):

    DIC = ['P', 'G', 'E'] + list(string.ascii_lowercase) + [' ']

    def __init__(self, model_parms, run_config, **kwargs):
        super(seq2seqAttention, self).__init__(model_parms, run_config, **kwargs)
    
    def model_fn(self, features, labels, mode, params):
        vocab_size = len(self.DIC)
        learning_rate = params.get('learning_rate', 0.001)
        embedding_size = params.get('embedding_size', 32)
        self.num_layers = params.get('num_layers', 2)
        self.rnn_size = params.get('rnn_size', 256)

        video = features['video']
        batch_size = tf.shape(video)[0]

        in_training = mode == tf.estimator.ModeKeys.TRAIN

        feature_extractor = LipNet(
            feature_len=256,
            training=in_training,
            scope='cnn_feature_extractor')
        
        net = feature_extractor.build(video)

        # encoder
        with tf.variable_scope('encoder', reuser=tf.AUTO_REUSE):
            encoder_cell = self._create_rnn_cell(mode)
            encoder_batch_size = tf.expand_dims(tf.shape(net)[0], 0)
            encoder_input_length = tf.expand_dims(tf.shape(net)[1], 0)
            encoder_sequence_length = tf.tile(encoder_input_length, encoder_batch_size)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, net, sequence_length=encoder_sequence_length)

        # decoder
        with tf.variable_scope('decoder', reuser=tf.AUTO_REUSE):
            embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
            # attention
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.rnn_size, memory=encoder_outputs, 
                memory_sequence_length=encoder_sequence_length)
            decoder_cell = self._create_rnn_cell(mode)
            decoder_cell = tf.contrib.seq2seq.AttenionWrapper(
                cell=decoder_cell, attention_mechanism=attention_mechanism,
                attention_layer_size=self.rnn_size, name='Attention_Wrapper')
            # initial state
            decoder_initial_state = decoder_cell.zero_state(
                batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            output_layer = tf.layers.Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # predict
        if mode == tf.estimator.ModeKeys.PREDICT:
            start_tokens = tf.ones([batch_size, ], tf.int32) # 'G'
            end_token = 2

            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding, start_tokens=start_tokens, end_tokens=end_token)
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                herlper=decoding_helper,
                initial_state=decoder_initial_state,
                output_layer=output_layer)
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, maximum_iterations=30)

            predicted_string = self.id_to_string(decoder_outputs.sample_id)
            predicted_output = {
                'predictions': decoder_outputs.sample_id,
                'predicted_string': predicted_string
            }
            export_outputs = {
                'predictions': tf.estimator.export.PredictOutput(predict_output)
            }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predict_output,
                export_outputs=export_outputs)

        # process label  
        labels = tf.squeeze(labels[ 'label'] )
        dense_numeric_label=self.preprocess_labels(labels)
          
        decoder_batch_size=tf.expand_dims(tf.shape(dense_numeric_label)[0],0)
        decoder_input_length = tf.expand_dims(tf.shape(dense_numeric_label)[1], 0)
        decoder_targets_length = tf.tile(decoder_input_length, decoder_batch_size)
        self.max_target_sequence_length = tf.reduce_max(decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(decoder_targets_length,self.max_target_sequence_length, dtype=tf.float32, name='masks') 
          
        # add '<go>' to target,and delete '<end>'
        ending = tf.strided_slice(dense_numeric_label, [0, 0], [batch_size, -1], [1, 1]) 
        decoder_input = tf.concat([tf.fill([batch_size, 1], 1), ending], 1)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input) 
        # decoder_inputs_embedded:[batch_size, decoder_targets_length, embedding_size] 
          
        #training，TrainingHelper+BasicDecoder
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded, sequence_length=decoder_targets_length, time_major=False, name='training_helper') 
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
                                                           helper=training_helper, 
                                                           initial_state=decoder_initial_state, 
                                                           output_layer=output_layer) 
          
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder, impute_finished=True, maximum_iterations=self.max_target_sequence_length) 
        #decoder_outputs:(rnn_outputs, sample_id) 
        # rnn_output: [batch_size, decoder_targets_length, vocab_size]
        # sample_id: [batch_size], tf.int32
          
        #cal loss
        self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
        loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train, targets=dense_numeric_label, weights=self.mask)
        cost = tf.reduce_mean(loss)
          
          
        #train
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.train.get_global_step()
            tvars = tf.trainable_variables()
            gradients = optimizer.compute_gradients(
            cost, tvars, colocate_gradients_with_ops=True)
            minimize_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(minimize_op, update_ops)

            labels = label_util.string2char_list(labels)
            predicted_string = self.id_to_string(decoder_outputs.sample_id)
            predicted_char_list = label_util.string2char_list(predicted_string)
            cer, wer = self.cal_metrics(labels, predicted_char_list)
            
            tf.summary.scalar('loss',cost)
            tf.summary.scalar('cer', tf.reduce_mean(cer))
            tf.summary.scalar('wer', tf.reduce_mean(wer))
            logging_hook = tf.train.LoggingTensorHook(
            {
                'loss': cost,
                'cer': tf.reduce_mean(cer),
                'wer': tf.reduce_mean(wer),
                'predicted': predicted_string[:5],
                'labels': label_util.char_list2string(labels)[:5]
            },
            every_n_iter=100)
            
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cost,
                train_op=train_op,
                training_hooks=[logging_hook]
                )
            return estimator_spec
       
        #eval
        if mode==tf.estimator.ModeKeys.EVAL:
            start_tokens = tf.ones([batch_size, ], tf.int32) #'G'
            end_token = 2 #'E'
            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding, start_tokens=start_tokens, end_token=end_token)
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
                                                                helper=decoding_helper, 
                                                                initial_state=decoder_initial_state, 
                                                                output_layer=output_layer) 
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, maximum_iterations=self.max_target_sequence_length)
            
            predicted_string = self.id_to_string(decoder_outputs.sample_id)
            predicted_char_list = label_util.string2char_list(predicted_string)
            labels = label_util.string2char_list(labels)
            cer, wer = self.cal_metrics(labels, predicted_char_list)
            
            tf.summary.scalar('cer', tf.reduce_mean(cer))
            tf.summary.scalar('wer', tf.reduce_mean(wer))
            eval_metric_ops = {
                'cer': tf.metrics.mean(cer),
                'wer': tf.metrics.mean(wer)
            }

            logging_hook = tf.train.LoggingTensorHook(
            {
                'loss': cost,
                'cer': tf.reduce_mean(cer),
                'wer': tf.reduce_mean(wer),
                'predicted': predicted_string[:5],
                'labels': label_util.char_list2string(labels)[:5]
            },
            every_n_iter=10)
            
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cost,
                predictions={"predictions": predicted_string},
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=[logging_hook]
                
            )
    
    def _create_rnn_cell(self, mode):
        def single_rnn_cell():
            single_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
            if mode == tf.estimator.ModeKeys.TRAIN:
                cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=0.5)
            else:
                cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=1.0)
            return cell
        cell = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell
    
    def preprocess_labels(self, labels):
        ''' preprocess labels
        Args:
            labels: 1-D Tensor. (batch_size,)
        Returns:
            2-D int32 Tensor with shape [batch_size, T]. EOS is padded to each label.
        '''
        pad_tensor = tf.constant(['E'], tf.string)
        # append 'E' to label
        eos = tf.tile(pad_tensor, tf.shape(labels))
        labels = tf.string_join([labels, eos])
        labels = string2char_list(labels)
        numeric_label = string2indices(labels, self.DIC)
        numeric_label = tf.cast(numeric_label, tf.int32)
        dense_numeric_label = tf.sparse_tensor_to_dense(numeric_label, default_value=0)
        return dense_numeric_label
    
    def id_to_string(self, predictions):
        """convert predictions to string.

        Args:
            predictions: 3-D int64 Tensor with shape: [batch_size, T, vocab_size]

        Returns: 1-D string SparseTensor with dense shape: [batch_size,]

        """
        predictions = tf.contrib.layers.dense_to_sparse(
            predictions, eos_token=0)  # remove PAD_ID
        predictions = tf.sparse_tensor_to_dense(predictions, 1)
        predictions = tf.contrib.layers.dense_to_sparse(
            predictions, eos_token=1)  # remove GO_ID
        predictions = tf.sparse_tensor_to_dense(predictions, 2)
        predictions = tf.contrib.layers.dense_to_sparse(
            predictions, eos_token=2)  # remove EOS_ID
        predicted_char_list = label_util.indices2string(predictions, self.DIC)
        predicted_string = label_util.char_list2string(
            predicted_char_list) 
        
        return predicted_string
