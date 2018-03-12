import tensorflow as tf
import numpy as np

from jamesbot.utils.padding import add_pad_eos

from model import Agent

class Trainer(object):

    DILATATION_RATES = [4, 2, 1]
    
    def __init__(self, n_slots, n_actions, word_embeddings_shape, save_path, hidden_size=300, graph=None, batch_size=64):
        self._sess = tf.Session(graph=graph)
        self._save_path = save_path
        
        self._n_slots = n_slots
        self._n_actions = n_actions
        self._word_embeddings_shape = word_embeddings_shape
        self._hidden_size = hidden_size
        
        self._batch_size = batch_size
        
        self.reset()

    @property
    def states_memory_shape(self):
        return (4, 3, self._batch_size, 2*self._hidden_size)
    
    def _metrics_writers(self):
        print('Metrics path {0}/metrics/'.format(self._save_path))
        
        self._train_writer = tf.summary.FileWriter('{0}/metrics/train'.format(self._save_path), self._sess.graph)
        self._test_writer = tf.summary.FileWriter('{0}/metrics/test'.format(self._save_path), self._sess.graph)
        self._metrics_op = tf.summary.merge_all()
        
    def initialize_word_embeddings(self, embeddings):
        embeddings_ph = tf.placeholder(tf.float32, self._word_embeddings_shape)
        init_op = self.agent._word_embeddings.assign(embeddings_ph)
        
        return self._sess.run(init_op, feed_dict={embeddings_ph: embeddings})

    def save_checkpoint(self, step):
        print('Write checkpoint:', self.agent.saver.save(self._sess, '{0}/checkpoints/model.ckpt'.format(self._save_path), global_step=step))
    
    def reset(self):
        self._states_index = np.zeros(self._batch_size, dtype=np.int32)
        self._states_memory = np.zeros(self.states_memory_shape, dtype=np.float32)

        assert self._states_memory.shape == self.states_memory_shape
        print('Reset states:', self._states_memory.shape)

    def _reset_states(self, predicate):
        self._states_index[predicate] = 0
        self._states_memory[:,:,predicate] = np.zeros(self.states_memory_shape[3], dtype=np.float32)            

        assert self._states_memory.shape == self.states_memory_shape

    def _get_states(self):
        cell_states = []
        for cell_id, dilatation in enumerate(self.DILATATION_RATES):
            batch_states = []
            for batch_sample_id, location_get_id in enumerate(self._states_index % dilatation):
                batch_states.append(self._states_memory[location_get_id,cell_id,batch_sample_id,:])
            cell_states.append(batch_states)
        return np.asarray(cell_states)
        
    def _update_states(self, states):
        states = np.array(states, copy=True)

        for cell_id, dilatation in enumerate(self.DILATATION_RATES):
            location_update_id = (dilatation - 1) - self._states_index % dilatation
            self._states_memory[location_update_id,cell_id,:,:] = states[cell_id,:,:]

        self._states_index += 1
        assert self._states_memory.shape == self.states_memory_shape

class SupervisedTrainer(Trainer):
    
    def __init__(self, **kwargs):
        super(SupervisedTrainer, self).__init__(**kwargs)
        
        self.state = tf.placeholder(tf.float32, self.states_memory_shape[1:])
        self.inputs = tf.placeholder(tf.int32, [None, None])
        self.inputs_length = tf.placeholder(tf.int32, [None])
        self.previous_output = tf.placeholder(tf.int32, [None, None])
        self.previous_output_length = tf.placeholder(tf.int32, [None])

        self.query_result_state = tf.placeholder(tf.int32, [None])
        self.query_result_slots = tf.placeholder(tf.int32, [None, None])
        self.query_result_values = tf.placeholder(tf.int32, [None, None, None])
        self.query_result_slots_count = tf.placeholder(tf.int32, [None])
        self.query_result_values_length = tf.placeholder(tf.int32, [None, None])
        
        self.slot_targets = tf.placeholder(tf.int32, [None, None])
        self.slot_any_targets = tf.placeholder(tf.int32, [None, None])
        self.action_targets = tf.placeholder(tf.int32, [None])
        self.value_targets = tf.placeholder(tf.float32, [None])
        self.targets = tf.placeholder(tf.int32, [None, None])
        self.targets_length = tf.placeholder(tf.int32, [None])
        
        self._decoder_sampling_p = tf.placeholder(tf.float32, [])
        self._loss_mixture_weights = tf.placeholder(tf.float32, [None])
        self._dropout = tf.placeholder(tf.float32, [])
        
        self.agent = Agent(
            self.state, self.inputs, self.inputs_length, self.previous_output, self.previous_output_length, self.query_result_state,
            self.query_result_slots, self.query_result_values, self.query_result_slots_count, self.query_result_values_length,
            word_embeddings_shape = self._word_embeddings_shape,
            n_slots = self._n_slots, n_actions = self._n_actions,
            decoder_targets = self.targets, decoder_targets_length = self.targets_length, decoder_sampling_p = self._decoder_sampling_p,
            hidden_size = self._hidden_size, dropout=self._dropout
        )
        
        self._loss()
        self._optimizer()
        self._metrics_writers()
        print('Dilatation rates:', self.DILATATION_RATES)
        
    def _loss(self):
        is_speak_sample = tf.cast(tf.equal(self.action_targets, 1), tf.float32)
        padded_targets = add_pad_eos(self.targets, self.targets_length, pre_pad=False)
        
        # Decoder
        stepwise_ce = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.agent._decoder_logits,
            labels = tf.one_hot(padded_targets, self._word_embeddings_shape[0])
        )
        stepwise_ce *= tf.sequence_mask(self.targets_length+2, dtype=tf.float32)
        
        self.decoder_loss = tf.reduce_sum((tf.reduce_sum(stepwise_ce, -1) / tf.cast(self.targets_length+2, tf.float32)) * is_speak_sample) / tf.reduce_sum(is_speak_sample)
        self.decoder_accuracy = tf.reduce_mean(tf.cast(tf.equal(padded_targets, self.agent.decoder_token_ids), tf.float32))
        
        # Slot parser
        slotwise_ce = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.agent._slot_logits,
            labels = tf.one_hot(self.slot_targets, self._n_slots)
        )
        slotwise_ce *= tf.sequence_mask(self.inputs_length, dtype=tf.float32)
        
        self.slots_loss = tf.reduce_sum((tf.reduce_sum(slotwise_ce, -1) / tf.cast(self.inputs_length, tf.float32)) * is_speak_sample) / tf.reduce_sum(is_speak_sample)
        self.slots_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.slot_targets, tf.argmax(self.agent.slot_probabilities, -1, output_type=tf.int32)), tf.float32))

        # Slot states
        slot_any_onehot_targets = tf.reduce_sum(tf.one_hot(self.slot_any_targets, self._n_slots), 1)
        slot_any_ce = tf.nn.sigmoid_cross_entropy_with_logits(
            logits = self.agent._slot_any_logits,
            labels = slot_any_onehot_targets
        )
        
        self.slot_any_loss = tf.reduce_mean(slot_any_ce)
        self.slot_any_accuracy = tf.reduce_mean(tf.cast(tf.equal(slot_any_onehot_targets, tf.cast(self.agent.slot_any, tf.float32)), tf.float32))
        
        # Policy
        action_ce = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.agent._action_logits,
            labels = tf.one_hot(self.action_targets, self._n_actions)
        )

        self.action_loss = tf.reduce_mean(action_ce)
        self.action_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.action_targets, self.agent.action_ids), tf.float32))

        # Value
        self.value_loss = tf.losses.mean_squared_error(
            labels = self.value_targets,
            predictions = self.agent.value
        )
        
        # Total Loss
        self.loss = tf.reduce_sum(tf.multiply(
            tf.stack([self.decoder_loss, self.slots_loss, self.slot_any_loss, self.action_loss, self.value_loss]),
            self._loss_mixture_weights
        ))
        
        # Metrics
        tf.summary.scalar('decoder_loss', self.decoder_loss)
        tf.summary.scalar('decoder_accuracy', self.decoder_accuracy)
        tf.summary.scalar('slots_loss', self.slots_loss)
        tf.summary.scalar('slots_accuracy', self.slots_accuracy)
        tf.summary.scalar('slot_any_loss', self.slot_any_loss)
        tf.summary.scalar('slot_any_accuracy', self.slot_any_accuracy)
        tf.summary.scalar('action_loss', self.action_loss)
        tf.summary.scalar('action_accuracy', self.action_accuracy)
        tf.summary.scalar('value_loss', self.value_loss)
        tf.summary.scalar('loss', self.loss)
        
    def _optimizer(self):
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def _compute_decoder_sampling_p(self, e, z=0.125, k=0.1):
        return ((2*z) / (1 + np.exp(-k*e)) - z)
        
    def _feed_dict(self, e, batch, opts={}):
        fd = {
            self.state: self._get_states(),
            self.inputs: batch['inputs'],
            self.inputs_length: batch['inputs_length'],
            self.previous_output: batch['previous_output'],
            self.previous_output_length: batch['previous_output_length'],
            self.targets: batch['targets'],
            self.targets_length: batch['targets_length'],
            self.slot_targets: batch['slot_targets'],
            self.slot_any_targets: batch['slot_any_targets'],
            self.action_targets: batch['action_targets'],
            self.value_targets: batch['value_targets'],
            self.query_result_state: batch['query_result_state'],
            self.query_result_slots: batch['query_result']['slots'],
            self.query_result_values: batch['query_result']['values'],
            self.query_result_slots_count: batch['query_result']['slots_count'],
            self.query_result_values_length: batch['query_result']['values_length'],
            self._loss_mixture_weights: [1., 1., 1., 1., 1.],
            self._decoder_sampling_p: self._compute_decoder_sampling_p(e)
        }

        for opt, val in opts.items():
            fd[opt] = val

        return fd
        
    def train_batch(self, e, i, batch):
        self._reset_states(batch['reset_state'])
        
        _, new_states, metrics_val = self._sess.run(
            [self.train_op, self.agent.policy_state, self._metrics_op],
            feed_dict=self._feed_dict(e, batch, {self._dropout: 0.3})
        )

        if i % 20 == 0:
            self._train_writer.add_summary(metrics_val)
        self._update_states(new_states)

    def test_batch(self, e, i, batch):
        self._reset_states(batch['reset_state'])
        
        new_states, metrics_val = self._sess.run(
            [self.agent.policy_state, self._metrics_op],
            feed_dict=self._feed_dict(e, batch, {self._dropout: 0.0})
        )

        if i % 20 == 0:
            self._test_writer.add_summary(metrics_val)
        self._update_states(new_states)
