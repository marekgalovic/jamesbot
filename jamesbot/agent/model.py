import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense

from jamesbot.utils.padding import add_pad_eos

class Agent(object):
    
    def __init__(self, previous_state, inputs, inputs_length, previous_output, previous_output_length, query_result_state, query_result_slots, query_result_values, query_result_slots_count, query_result_values_length, word_embeddings_shape, n_slots, n_actions, decoder_targets=None, decoder_targets_length=None, decoder_sampling_p=0.0, trainable_embeddings=True, hidden_size=300, dropout=0.0):
        super(Agent, self).__init__()
        
        # Inputs
        self._previous_state = previous_state
        self._inputs = inputs
        self._inputs_length = inputs_length
        self._previous_output = previous_output
        self._previous_output_length = previous_output_length
        
        # Query result
        self._query_result_state = query_result_state
        self._query_result_slots = query_result_slots
        self._query_result_values = query_result_values
        self._query_result_slots_count = query_result_slots_count
        self._query_result_values_length = query_result_values_length
            
        # Decoder targets (only for training)
        self._decoder_targets = decoder_targets
        self._decoder_targets_length = decoder_targets_length
        self._decoder_sampling_p = decoder_sampling_p
        
        # Conf
        self._hidden_size = int(hidden_size)
        self._word_embeddings_shape = list(word_embeddings_shape)
        self._n_slots = int(n_slots)
        self._n_actions = int(n_actions)
        self._n_query_states = 3
        self._trainable_embeddings = bool(trainable_embeddings)
        self._dropout = dropout
        tf.summary.scalar('dropout', self._dropout)

        print('Agent(hidden_size={0}, n_slots={1}, n_actions={2})'.format(self._hidden_size, self._n_slots, self._n_actions))
        
        # Build
        self._embeddings_module()
        self._input_encoder()
        self._slot_parser()
        self._query_result_encoder()
        self._policy()
        self._response_generator()
        
        self._saver_ops()
        
    @property
    def policy_state_size(self):
        return 2*self._hidden_size
        
    def _saver_ops(self):
        self.saver = tf.train.Saver(max_to_keep=None)
        
    def _embeddings_module(self):
        with tf.name_scope('embeddings'):
            self._word_embeddings = tf.Variable(tf.zeros(self._word_embeddings_shape), trainable=self._trainable_embeddings, name='word_embeddings')
            self._slot_embeddings = tf.Variable(tf.zeros([self._n_slots, self._word_embeddings_shape[1]]), trainable=True, name='slot_embeddings')
            
            self._inputs_embedded = tf.nn.embedding_lookup(self._word_embeddings, self._inputs)
            self._previous_output_embedded = tf.nn.embedding_lookup(self._word_embeddings, self._previous_output)
            
            self._query_result_slots_embedded = tf.nn.embedding_lookup(self._slot_embeddings, self._query_result_slots)
            self._query_result_values_embedded = tf.nn.embedding_lookup(self._word_embeddings, self._query_result_values)
            
    def _rnn_cell(self, size=None, activation=None, dropout=None, residual=False):
        cell = rnn.GRUCell((size or self._hidden_size), activation=activation)

        if residual:
            cell = rnn.ResidualWrapper(cell)

        if dropout is not None:
            cell = rnn.DropoutWrapper(cell, input_keep_prob=(1.0 - dropout))

        return cell
        
    def _text_encoder(self, inputs, inputs_length, scope='text_encoder', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            _outputs, _state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self._rnn_cell(activation=tf.nn.tanh),
                cell_bw = self._rnn_cell(activation=tf.nn.tanh),
                inputs = inputs,
                sequence_length = inputs_length,
                dtype = tf.float32
            )

        return (
            tf.concat(_outputs, -1),
            tf.concat(_state, -1)
        )
 
    def _input_encoder(self):
        with tf.name_scope('inputs_encoder'):            
            (self._inputs_encoder_outputs,
             self._inputs_encoder_state) = self._text_encoder(self._inputs_embedded, self._inputs_length)
            (self._previous_output_encoder_outputs,
             self._previous_output_encoder_state) = self._text_encoder(self._previous_output_embedded, self._previous_output_length, reuse=True)
            
    def _slot_parser(self):
        with tf.variable_scope('slot_parser'):
            # Project text encoder state
            e_inputs = tf.layers.dense(
                self._inputs_encoder_outputs,
                self._hidden_size,
                activation = tf.nn.tanh,
                name = 'text_encoder_outputs_projection'
            )
            e_previous_output = tf.layers.dense(
                self._previous_output_encoder_outputs,
                self._hidden_size,
                activation = tf.nn.tanh,
                name = 'text_encoder_outputs_projection',
                reuse = True
            )
            
            # Compare inputs with agent's previous output
            e = tf.matmul(e_inputs, e_previous_output, transpose_b=True, name='e')
            beta = tf.matmul(tf.nn.softmax(e), self._previous_output_encoder_outputs)
            
            inputs_compared = tf.layers.dense(
                tf.concat([self._inputs_encoder_outputs, beta], 2),
                self._hidden_size,
                activation = tf.nn.tanh
            )
            
            # Final slot logits/probabilities
            self._slot_logits = tf.layers.dense(
                tf.layers.dropout(inputs_compared, rate=self._dropout),
                self._n_slots
            )
            self.slot_probabilities = tf.nn.softmax(self._slot_logits)
            self.slot_ids = tf.argmax(self._slot_logits, -1)

            # Slot (any)
            state_proj = tf.layers.dense(
                tf.concat([self._inputs_encoder_state, self._previous_output_encoder_state], -1),
                self._hidden_size,
                activation = tf.nn.tanh
            )
            
            self._slot_any_logits = tf.layers.dense(
                tf.layers.dropout(state_proj, rate=self._dropout),
                self._n_slots
            )
            self.slot_any_probabilites = tf.sigmoid(self._slot_any_logits)
            self.slot_any = tf.greater(self.slot_any_probabilites, .5)
    
    def _query_result_encoder(self):
        with tf.name_scope('query_result_encoder'):
            batch_size, n_slots, n_tokens = tf.unstack(tf.shape(self._query_result_values))
    
            _, _value_encoder_state = self._text_encoder(
                inputs = tf.reshape(self._query_result_values_embedded, [-1, n_tokens, self._word_embeddings_shape[1]]),
                inputs_length = tf.reshape(self._query_result_values_length, [-1]),
                reuse = True
            )
        
            query_result_slot_value = tf.concat([
                self._query_result_slots_embedded,
                tf.reshape(_value_encoder_state, [batch_size, n_slots, 2*self._hidden_size])
            ], -1)
            
            _, self._query_result_encoder_state = tf.nn.dynamic_rnn(
                self._rnn_cell(activation=tf.nn.tanh, dropout=self._dropout),
                inputs = query_result_slot_value,
                sequence_length = self._query_result_slots_count,
                dtype = tf.float32
            )
        
    def _policy(self):
        with tf.name_scope('policy'):
            policy_context = tf.layers.dense(
                tf.concat([
                    self._inputs_encoder_state,
                    self._query_result_encoder_state,
                    tf.one_hot(self._query_result_state, self._n_query_states, dtype=tf.float32),
                ], -1),
                2*self.policy_state_size
            )

            policy_cell = rnn.MultiRNNCell([
                self._rnn_cell(self.policy_state_size, activation=tf.nn.tanh, dropout=self._dropout),
                self._rnn_cell(self.policy_state_size, activation=tf.nn.tanh, dropout=self._dropout),
                self._rnn_cell(self.policy_state_size, activation=tf.nn.tanh, dropout=self._dropout)
            ])

            self._policy_output, self.policy_state = policy_cell(
                policy_context,
                tuple([self._previous_state[i,:,:] for i in range(3)])
            )

            # Value
            value_l1 = tf.layers.dense(
                tf.layers.dropout(self._policy_output, rate=self._dropout),
                self._hidden_size,
                activation = tf.nn.tanh
            )
            self.value = tf.layers.dense(
                tf.layers.dropout(value_l1, rate=self._dropout),
                1
            )
            self.value = tf.squeeze(self.value, -1)

            # Action
            action_l1 = tf.layers.dense(
                tf.layers.dropout(self._policy_output, rate=self._dropout),
                self._hidden_size,
                activation = tf.nn.tanh
            )
            self._action_logits = tf.layers.dense(
                tf.layers.dropout(action_l1, rate=self._dropout),
                self._n_actions
            )
            self.action_probabilities = tf.nn.softmax(self._action_logits)
            self.action_ids = tf.argmax(self._action_logits, -1, output_type=tf.int32)
        
    def _response_generator(self):
        with tf.name_scope('response_generator'):
            batch_size, _ = tf.unstack(tf.shape(self._inputs))
            self._decoder_output_layer = Dense(self._word_embeddings_shape[0])
            
            if self._decoder_targets is not None:
                print('Training decoder helper.')
                
                decoder_targets_embedded = tf.nn.embedding_lookup(
                    self._word_embeddings,
                    add_pad_eos(self._decoder_targets, self._decoder_targets_length)
                )
                helper = seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs = decoder_targets_embedded,
                    sequence_length = (self._decoder_targets_length + 2),
                    embedding = self._word_embeddings,
                    sampling_probability = self._decoder_sampling_p
                )
                tf.summary.scalar('decoder_sampling_p', self._decoder_sampling_p)
                
            else:
                print('Inference decoder helper.')
                
                helper = seq2seq.GreedyEmbeddingHelper(
                    embedding = self._word_embeddings,
                    start_tokens = tf.tile([0], [batch_size]),
                    end_token = 1
                )
                
            decoder_cell, decoder_initial_state = self._decoder_cell()
            decoder = seq2seq.BasicDecoder(
                decoder_cell,
                helper = helper,
                initial_state = decoder_initial_state,
                output_layer = self._decoder_output_layer
            )
            
            decoder_outputs, _, _ = seq2seq.dynamic_decode(
                decoder = decoder,
                impute_finished = True
            )
            
            self._decoder_logits = decoder_outputs.rnn_output
            self.decoder_token_ids = tf.argmax(self._decoder_logits, -1, output_type=tf.int32)
            
    def _decoder_cell(self):
        batch_size, _ = tf.unstack(tf.shape(self._policy_output))

        attention = seq2seq.BahdanauAttention(
            num_units = 2*self._hidden_size,
            memory = self._inputs_encoder_outputs,
            memory_sequence_length = self._inputs_length
        )
        
        attentive_cell = seq2seq.AttentionWrapper(
            cell = self._rnn_cell(self.policy_state_size, activation=tf.nn.tanh),
            attention_mechanism = attention,
            attention_layer_size = 2*self._hidden_size,
            initial_cell_state = self._policy_output
        )

        cell = rnn.MultiRNNCell([
            attentive_cell,
            self._rnn_cell(self.policy_state_size, activation=tf.nn.tanh),
        ])

        initial_state = tuple([
            attentive_cell.zero_state(batch_size, tf.float32),
            self._policy_output
        ])

        return cell, initial_state
