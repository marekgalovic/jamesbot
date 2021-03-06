import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq

from jamesbot.utils.bleu import compute_bleu
from jamesbot.utils.padding import add_pad_eos, pad_sequences
from jamesbot.seq2seq import GreedyEmbeddingTrainingHelper

from model import Agent
from decoder_critic import DecoderCritic


class Trainer(object):

    DILATATION_RATES = [4, 2, 1]
    
    def __init__(self, n_slots, n_actions, word_embeddings_shape, save_path, hidden_size=300, sess=None, graph=None, batch_size=64):
        self._sess = (sess or tf.Session(graph=graph))
        self._save_path = save_path
        
        self._n_slots = n_slots
        self._n_actions = n_actions
        self._word_embeddings_shape = word_embeddings_shape
        self._hidden_size = hidden_size
        self._batch_size = batch_size

        print('Trainer - Dilatation rates:', self.DILATATION_RATES)
        print('Metrics path:', self.metrics_path)
        print('Checkpoints path:', self.checkpoints_path)
        
        self.reset()

    def set_batch_size(self, batch_size):
        self._batch_size = int(batch_size)

    @property
    def states_memory_shape(self):
        return (4, 3, self._batch_size, 2*self._hidden_size)

    @property
    def metrics_path(self):
        return '{0}/metrics'.format(self._save_path)

    @property
    def checkpoints_path(self):
        return '{0}/checkpoints/model.ckpt'.format(self._save_path)

    def _metrics_writers(self):
        self._train_writer = tf.summary.FileWriter('{0}/train'.format(self.metrics_path), self._sess.graph)
        self._test_writer = tf.summary.FileWriter('{0}/test'.format(self.metrics_path), self._sess.graph)
        self._metrics_op = tf.summary.merge_all()
        
    def initialize_word_embeddings(self, embeddings):
        embeddings_ph = tf.placeholder(tf.float32, self._word_embeddings_shape)
        init_op = self.agent._word_embeddings.assign(embeddings_ph)
        
        return self._sess.run(init_op, feed_dict={embeddings_ph: embeddings})

    def save_checkpoint(self, step):
        self.agent.saver.save(self._sess, self.checkpoints_path, global_step=step)
    
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

class CrossEntropyTrainer(Trainer):
    
    def __init__(self, scope='agent', **kwargs):
        super(CrossEntropyTrainer, self).__init__(**kwargs)
        
        self.targets = tf.placeholder(tf.int32, [None, None], name='ph/targets')
        self.targets_length = tf.placeholder(tf.int32, [None], name='ph/targets_length')
        self.slot_targets = tf.placeholder(tf.int32, [None, None], name='ph/slot_targets')
        self.slot_any_targets = tf.placeholder(tf.int32, [None, None], name='ph/slot_any_targets')
        self.action_targets = tf.placeholder(tf.int32, [None], name='ph/action_targets')
        self.value_targets = tf.placeholder(tf.float32, [None], name='ph/value_targets')
        
        self._decoder_sampling_p = tf.placeholder(tf.float32, [], name='ph/decoder_sampling_p')
        self._loss_mixture_weights = tf.placeholder(tf.float32, [None], name='ph/loss_mixture_weights')

        self._dropout = tf.placeholder(tf.float32, [], name='ph/dropout')
        tf.summary.scalar('dropout', self._dropout)
        
        self.agent = Agent(
            n_slots = self._n_slots, n_actions = self._n_actions,
            word_embeddings_shape = self._word_embeddings_shape,
            hidden_size = self._hidden_size, dropout=self._dropout,
            decoder_helper_initializer = self._decoder_helper(),
            scope = scope
        )
        
        self._loss()
        self._optimizer()
        self._metrics_writers()

    def _decoder_helper(self):
        def _closure(word_embeddings):
            tf.summary.scalar('decoder_sampling_p', self._decoder_sampling_p)

            decoder_targets_embedded = tf.nn.embedding_lookup(
                word_embeddings,
                add_pad_eos(self.targets, self.targets_length)
            )

            return seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs = decoder_targets_embedded,
                sequence_length = (self.targets_length + 2),
                embedding = word_embeddings,
                sampling_probability = self._decoder_sampling_p
            )
            
        return _closure
        
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
        slotwise_ce *= tf.sequence_mask(self.agent.inputs_length, dtype=tf.float32)
        
        self.slots_loss = tf.reduce_sum((tf.reduce_sum(slotwise_ce, -1) / tf.cast(self.agent.inputs_length, tf.float32)) * is_speak_sample) / tf.reduce_sum(is_speak_sample)
        self.slots_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.slot_targets, tf.argmax(self.agent.slot_probabilities, -1, output_type=tf.int32)), tf.float32))

        # Slot states
        slot_any_targets = tf.reduce_sum(tf.one_hot(self.slot_any_targets, self._n_slots, dtype=tf.int32), 1)
        slot_any_ce = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.agent._slot_any_logits,
            labels = tf.one_hot(slot_any_targets, 2)
        )
        
        self.slot_any_loss = tf.reduce_mean(slot_any_ce)
        self.slot_any_accuracy = tf.reduce_mean(tf.cast(tf.equal(slot_any_targets, self.agent.slot_any), tf.float32))
        
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
            # Inputs
            self.agent.previous_context_state: self._get_states(),
            self.agent.inputs: batch['inputs'],
            self.agent.inputs_length: batch['inputs_length'],
            self.agent.previous_output: batch['previous_output'],
            self.agent.previous_output_length: batch['previous_output_length'],
            self.agent.query_result_state: batch['query_result_state'],
            self.agent.query_result_slots: batch['query_result']['slots'],
            self.agent.query_result_values: batch['query_result']['values'],
            self.agent.query_result_slots_count: batch['query_result']['slots_count'],
            self.agent.query_result_values_length: batch['query_result']['values_length'],
            # Targets
            self.targets: batch['targets'],
            self.targets_length: batch['targets_length'],
            self.slot_targets: batch['slot_targets'],
            self.slot_any_targets: batch['slot_any_targets'],
            self.action_targets: batch['action_targets'],
            self.value_targets: batch['value_targets'],
            # Conf
            self._loss_mixture_weights: [1., 1., 1., .5, .5],
            self._decoder_sampling_p: self._compute_decoder_sampling_p(e)
        }

        for opt, val in opts.items():
            fd[opt] = val

        return fd
        
    def train_batch(self, e, i, batch):
        self._reset_states(batch['reset_state'])
        
        _, new_states, metrics_val = self._sess.run(
            [self.train_op, self.agent.context_state, self._metrics_op],
            feed_dict=self._feed_dict(e, batch, {self._dropout: 0.3})
        )

        if i % 20 == 0:
            self._train_writer.add_summary(metrics_val)
        self._update_states(new_states)

    def test_batch(self, e, i, batch):
        self._reset_states(batch['reset_state'])
        
        new_states, metrics_val = self._sess.run(
            [self.agent.context_state, self._metrics_op],
            feed_dict=self._feed_dict(e, batch, {self._dropout: 0.0})
        )

        if i % 20 == 0:
            self._test_writer.add_summary(metrics_val)
        self._update_states(new_states)


class ACTrainer(Trainer):

    LAMBDA_C = 1e-4
    LAMBDA_LL = 1e-3

    GAMMA_ACTOR = 1e-5
    GAMMA_CRITIC = 1e-5

    def __init__(self, agent_path=None, critic_path=None, **kwargs):
        super(ACTrainer, self).__init__(**kwargs)

        self.targets = tf.placeholder(tf.int32, [None, None], name='ph/targets')
        self.targets_length = tf.placeholder(tf.int32, [None], name='ph/targets_length')

        self._generated_token_ids = tf.placeholder(tf.int32, [None, None], name='ph/generated_token_ids')
        self._r = tf.placeholder(tf.float32, [None, None], name='ph/r')

        avg_bleu = tf.reduce_mean(tf.reduce_sum(self._r, -1) / tf.cast(self.targets_length, tf.float32))
        tf.summary.scalar('avg_bleu', avg_bleu)

        self._dropout = tf.placeholder(tf.float32, [], name='ph/dropout')
        tf.summary.scalar('dropout', self._dropout)

        self.target_agent = self._agent_builder(scope='target_agent')
        self.agent = self._agent_builder()

        self.target_critic = self._critc_builder(scope='target_critic')
        self.critic = self._critc_builder()

        self._objectives()
        self._interpolate_ops()
        self._train_ops()
        self._initialize(agent_path, critic_path)
        self._metrics_writers()

    def save_checkpoint(self, step):
        self.target_agent.saver.save(self._sess, '{0}/checkpoints/target_agent.ckpt'.format(self._save_path), global_step=step)
        self.target_critic.saver.save(self._sess, '{0}/checkpoints/target_critic.ckpt'.format(self._save_path), global_step=step)

    def reset_gammas(self):
        self.GAMMA_ACTOR = 1e-5
        self.GAMMA_CRITIC = 1e-5

    def _initialize(self, agent_path, critic_path):
        self._sess.run(tf.global_variables_initializer())

        print('Restore agent:', agent_path)
        self.target_agent.saver.restore(self._sess, agent_path)

        if critic_path is not None:
            print('Restore critic:', critic_path)
            self.target_critic.saver.restore(self._sess, critic_path)

        copy_ops = []
        for var, target_var in zip(self.agent._vars, self.target_agent._vars):
            copy_ops.append(var.assign(target_var))
            
        for var, target_var in zip(self.critic._vars, self.target_critic._vars):
            copy_ops.append(var.assign(target_var))

        copy_ops = tf.group(*copy_ops)
        self._sess.run(copy_ops)

    def _agent_builder(self, scope='agent', reuse=False):
        dropout = (0.0 if scope.startswith('target') else self._dropout)

        return Agent(
            n_slots = self._n_slots, n_actions = self._n_actions,
            word_embeddings_shape = self._word_embeddings_shape,
            hidden_size = self._hidden_size, dropout=dropout,
            decoder_helper_initializer = self._decoder_helper(),
            scope = scope, reuse = reuse
        )

    def _decoder_helper(self):
        def _initializer(word_embeddings):
            return GreedyEmbeddingTrainingHelper(
                start_tokens = tf.tile([0], [tf.shape(self.targets)[0]]),
                sequence_length = (self.targets_length + 2),
                embedding = word_embeddings
            )

        return _initializer

    def _critc_builder(self, scope='critic'):
        return DecoderCritic(
            self.agent, self.targets, self.targets_length,
            scope = scope
        )

    def _feed_dict(self, batch, opts = {}):
        # TODO: Use single set of placeholders
        context_states = self._get_states()
        fd = {
            # Agent inputs
            self.agent.previous_context_state: context_states,
            self.agent.inputs: batch['inputs'],
            self.agent.inputs_length: batch['inputs_length'],
            self.agent.previous_output: batch['previous_output'],
            self.agent.previous_output_length: batch['previous_output_length'],
            self.agent.query_result_state: batch['query_result_state'],
            self.agent.query_result_slots: batch['query_result']['slots'],
            self.agent.query_result_values: batch['query_result']['values'],
            self.agent.query_result_slots_count: batch['query_result']['slots_count'],
            self.agent.query_result_values_length: batch['query_result']['values_length'],
            # Target agent inputs
            self.target_agent.previous_context_state: context_states,
            self.target_agent.inputs: batch['inputs'],
            self.target_agent.inputs_length: batch['inputs_length'],
            self.target_agent.previous_output: batch['previous_output'],
            self.target_agent.previous_output_length: batch['previous_output_length'],
            self.target_agent.query_result_state: batch['query_result_state'],
            self.target_agent.query_result_slots: batch['query_result']['slots'],
            self.target_agent.query_result_values: batch['query_result']['values'],
            self.target_agent.query_result_slots_count: batch['query_result']['slots_count'],
            self.target_agent.query_result_values_length: batch['query_result']['values_length'],
            # Critic inputs
            self.targets: batch['targets'],
            self.targets_length: batch['targets_length'],
            self._dropout: 0.3
        }

        for opt, val in opts.items():
            fd[opt] = val

        return fd

    @property
    def _generated_token_weights(self):
        # TODO: Use gather_nd
        token_mask = tf.one_hot(self._generated_token_ids, self._word_embeddings_shape[0])
        return tf.reduce_sum(token_mask * self.critic.values, -1)

    @property
    def _generated_token_probabilites(self):
        # TODO: Use gather_nd
        token_mask = tf.one_hot(self._generated_token_ids, self._word_embeddings_shape[0])
        return tf.reduce_sum(token_mask * self.agent.decoder_probabilities, -1)

    def _objectives(self):
        q = self._r + tf.stop_gradient(tf.reduce_sum(self.target_agent.decoder_probabilities * self.target_critic.values, -1))
        c = tf.reduce_sum(tf.square(self.critic.values - tf.reduce_mean(self.critic.values, -1, keep_dims=True)), -1)
        self._critic_objective = tf.reduce_mean(tf.reduce_sum(tf.square(self._generated_token_weights - q) + (self.LAMBDA_C * c), -1))

        ll_reg = tf.reduce_sum(self._generated_token_probabilites, -1)
        self._agent_objective = tf.reduce_mean(tf.reduce_sum(self.agent.decoder_probabilities * tf.stop_gradient(self.critic.values), [-1, -2]) + self.LAMBDA_LL * ll_reg)

        tf.summary.scalar('critic_objective', self._critic_objective)
        tf.summary.scalar('agent_objective', self._agent_objective)

    def _interpolate_ops(self):
        agent_ops, critic_ops = [], []

        for var, target_var in zip(self.agent._vars, self.target_agent._vars):
            assert 'target_%s' % var.name == target_var.name
            op = target_var.assign(self.GAMMA_ACTOR * var + (1. - self.GAMMA_ACTOR) * target_var)
            agent_ops.append(op)

        for var, target_var in zip(self.critic._vars, self.target_critic._vars):
            assert 'target_%s' % var.name == target_var.name
            op = target_var.assign(self.GAMMA_CRITIC * var + (1. - self.GAMMA_CRITIC) * target_var)
            critic_ops.append(op)

        self._interpolate_agent_weights = tf.group(*agent_ops)
        self._interpolate_critic_weights = tf.group(*critic_ops)

    def _train_ops(self):
        self._train_critic = (
            tf.train.AdamOptimizer()
            .minimize(self._critic_objective, var_list=self.critic._vars)
        )

        self._train_agent = (
            tf.train.AdamOptimizer()
            .minimize(self._agent_objective, var_list=self.agent._vars)
        )

    def _bleu_reward(self, batch, predictions):
        # Shaped BLEU Scores
        rewards = []
        for i, target in enumerate(batch['targets']):
            rewards.append([])

            for j in range(1, batch['targets_length'][i] + 1):
                bleu, _, _, _, _, _ = compute_bleu(
                    [[target[:j]]], [predictions[i][:j]],
                    smooth=True
                )

                previous_reward = (rewards[i][-1] if len(rewards[i]) > 0 else 0.0)
                rewards[i].append(bleu - previous_reward)

        return pad_sequences(rewards, max_len=max(batch['targets_length'])+2, dtype=np.float32)

    def train_critic_batch(self, e, i, batch):
        token_ids, new_states = self._sess.run(
            [self.target_agent.decoder_token_ids, self.target_agent.context_state],
            feed_dict = self._feed_dict(batch)
        )

        r = self._bleu_reward(batch, token_ids)
        _, _, metrics_val = self._sess.run(
            [self._train_critic, self._interpolate_critic_weights, self._metrics_op],
            feed_dict = self._feed_dict(batch, {
                self._r: r,
                self._generated_token_ids: token_ids
            })
        )

        if i % 20 == 0:
            self._train_writer.add_summary(metrics_val)
        self._update_states(new_states)

    def test_critic_batch(self, e, i, batch):
        token_ids, new_states = self._sess.run(
            [self.target_agent.decoder_token_ids, self.target_agent.context_state],
            feed_dict = self._feed_dict(batch)
        )

        r = self._bleu_reward(batch, token_ids)
        metrics_val = self._sess.run(
            self._metrics_op,
            feed_dict = self._feed_dict(batch, {
                self._r: r,
                self._generated_token_ids: token_ids,
                self._dropout: 0.0
            })
        )

        if i % 20 == 0:
            self._test_writer.add_summary(metrics_val)
        self._update_states(new_states)

    def train_batch(self, e, i, batch):
        token_ids, new_states = self._sess.run(
            [self.target_agent.decoder_token_ids, self.target_agent.context_state],
            feed_dict = self._feed_dict(batch)
        )

        r = self._bleu_reward(batch, token_ids)
        self._sess.run(
            self._train_critic,
            feed_dict = self._feed_dict(batch, {
                self._r: r,
                self._generated_token_ids: token_ids,
                self._dropout: 0.0
            })
        )

        _, metrics_val = self._sess.run(
            [self._train_agent, self._metrics_op],
            feed_dict = self._feed_dict(batch, {
                self._r: r,
                self._generated_token_ids: token_ids,
                self._dropout: 0.0
            })
        )

        self._sess.run([self._interpolate_agent_weights, self._interpolate_critic_weights])

        if i % 100 == 0:
            self._train_writer.add_summary(metrics_val)

        self._update_states(new_states)

    def test_batch(self, e, i, batch):
        token_ids, new_states = self._sess.run(
            [self.target_agent.decoder_token_ids, self.target_agent.context_state],
            feed_dict = self._feed_dict(batch)
        )

        r = self._bleu_reward(batch, token_ids)
        metrics_val = self._sess.run(
            self._metrics_op,
            feed_dict = self._feed_dict(batch, {
                self._r: r,
                self._generated_token_ids: token_ids,
                self._dropout: 0.0
            })
        )

        if i % 100 == 0:
            self._test_writer.add_summary(metrics_val)

        self._update_states(new_states)
