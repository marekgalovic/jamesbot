import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq

from jamesbot.utils.padding import add_pad_eos

class DecoderCritic(object):

    CELL_SIZE = 300

    def __init__(self, agent, targets, targets_length, scope='critic', reuse=False):
        self._agent = agent
        self._targets = targets
        self._targets_length = targets_length

        with tf.variable_scope(scope, reuse=reuse):
            self._embeddings()
            self._targets_encoder()
            self._values_decoder()

            self._vars = [var for var in tf.trainable_variables() if var.name.startswith(scope)]
            self.saver = tf.train.Saver(var_list = self._vars, max_to_keep=None)

        print('DecoderCritic(cell_size={0}, scope={1})'.format(self.CELL_SIZE, str(scope)))

    def _embeddings(self):
        with tf.name_scope('embeddings'):
            self._targets_embedded = tf.nn.embedding_lookup(
                self._agent._word_embeddings,
                add_pad_eos(self._targets, self._targets_length, pre_pad=False)
            )

    def _targets_encoder(self):
        with tf.name_scope('targets_encoder'):
            (_outputs, _state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = rnn.GRUCell(self.CELL_SIZE, activation=tf.nn.tanh),
                cell_bw = rnn.GRUCell(self.CELL_SIZE, activation=tf.nn.tanh),
                inputs = self._targets_embedded,
                sequence_length = self._targets_length,
                dtype = tf.float32
             )

            self._targets_encoder_outputs = tf.concat(_outputs, -1)
            self._targets_encoder_state = tf.concat(_state, -1)

    def _values_decoder(self):
        with tf.name_scope('values_decoder'):
            decoder_cell, decoder_initial_state = self._decoder_cell()

            _outputs, _ = tf.nn.dynamic_rnn(
                decoder_cell,
                inputs = self._agent._decoder_logits,
                sequence_length = (self._targets_length + 2),
                initial_state = decoder_initial_state,
                dtype = tf.float32
            )

            self.values = tf.layers.dense(_outputs, self._agent._word_embeddings_shape[0])

    def _decoder_cell(self):
        batch_size, _ = tf.unstack(tf.shape(self._targets))

        attention = seq2seq.BahdanauAttention(
            num_units = 2*self.CELL_SIZE,
            memory = self._targets_encoder_outputs,
            memory_sequence_length = self._targets_length
        )
        
        attentive_cell = seq2seq.AttentionWrapper(
            cell = rnn.GRUCell(2*self.CELL_SIZE, activation=tf.nn.tanh),
            attention_mechanism = attention,
            attention_layer_size = 2*self.CELL_SIZE,
            initial_cell_state = self._targets_encoder_state
        )

        return (
            attentive_cell,
            attentive_cell.zero_state(batch_size, tf.float32),
        )
