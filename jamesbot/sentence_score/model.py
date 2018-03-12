import tensorflow as tf
from tensorflow.contrib import rnn

class SentenceScoreModel(object):
    
    def __init__(self, inputs, inputs_length, embeddings_shape, dropout=None):
        self._inputs = inputs
        self._inputs_length = inputs_length
        self._embeddings_shape = embeddings_shape
        
        self._hidden_size = 512
        self._dropout = dropout
        
        with tf.name_scope('sentence_score_model'):
            self._embeddings()
            self._encoder()
            self._classifier()
            
            self.saver = tf.train.Saver(max_to_keep=None)
        
    def _embeddings(self):
        with tf.name_scope('embeddings'):
            self._word_embeddings = tf.Variable(
                tf.zeros(self._embeddings_shape, dtype=tf.float32),
                trainable = True
            )
            
            self._inputs_embedded = tf.nn.embedding_lookup(self._word_embeddings, self._inputs)
            
    def _encoder(self):
        with tf.name_scope('encoder'):
            _, _state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = rnn.GRUCell(self._hidden_size, activation=tf.nn.tanh),
                cell_bw = rnn.GRUCell(self._hidden_size, activation=tf.nn.tanh),
                inputs = self._inputs_embedded,
                sequence_length = self._inputs_length,
                dtype = tf.float32
            )
            
            self._encoder_state = tf.concat(_state, -1)
            
    def _classifier(self):
        with tf.name_scope('classifier'):
            L1 = tf.layers.dense(
                tf.layers.dropout(self._encoder_state, rate=self._dropout),
                self._hidden_size,
                activation = tf.nn.tanh
            )
            
            L2 = tf.layers.dense(
                tf.layers.dropout(L1, rate=self._dropout),
                int(self._hidden_size / 2),
                activation = tf.nn.tanh
            )
            
            self.p = tf.layers.dense(
                tf.layers.dropout(L2, rate=self._dropout),
                1,
                activation = tf.nn.sigmoid
            )
            
            self.p = tf.squeeze(self.p, -1)
