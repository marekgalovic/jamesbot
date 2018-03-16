import json
import tensorflow as tf
import numpy as np
from optparse import OptionParser
from tensorflow.python.lib.io.file_io import FileIO

from utils import SamplesIterator
from trainer import CrossEntropyTrainer, ACTrainer


parser = OptionParser()
parser.add_option('--data-dir', dest='data_dir')
parser.add_option('--job-dir', dest='models_dir')
parser.add_option('--run-name', dest='run_name')
parser.add_option('--agent-checkpoint', dest='agent_checkpoint', default=None)

options, _ = parser.parse_args()
print('Data dir:', options.data_dir)
print('Models dir:', options.models_dir)
print('Run name:', options.run_name)
print('Agent checkpoint:', options.agent_checkpoint)


def load_data(name):
    fullpath = '{0}/{1}'.format(options.data_dir, name)
    print('Load:', fullpath)
    return json.load(FileIO(fullpath, 'r'))


embeddings = np.asarray(load_data('embeddings.json'))
slots_dict = load_data('slots_dictionary.json')
actions_dict = load_data('actions_dictionary.json')

samples_train = load_data('embedded_frames_train.json')
samples_test = load_data('embedded_frames_test.json')

def train(actor_epochs = 15, critic_epochs = 5, ac_epochs = 3, batch_size=64, test=False):
    print('train(actor_epochs=%d, critic_epochs=%d, ac_epochs=%d, batch_size=%d, test=%r)' % (actor_epochs, critic_epochs, ac_epochs, batch_size, test))
    base_path = '{0}/ac_agent_{1}'.format(options.models_dir, options.run_name)

    train_samples_iterator = SamplesIterator(samples_train, batch_size=batch_size)
    test_samples_iterator = SamplesIterator(samples_test, batch_size=batch_size)

    if options.agent_checkpoint is None:
        # Pre-train agent
        ce_trainer = CrossEntropyTrainer(
            n_slots = len(slots_dict),
            n_actions = len(actions_dict),
            word_embeddings_shape = embeddings.shape,
            save_path = '{0}/ce'.format(base_path),
            scope = 'target_agent'
        )

        ce_trainer._sess.run(tf.global_variables_initializer())
        ce_trainer.initialize_word_embeddings(embeddings)

        for e in range(actor_epochs):
            print('Agent pre-training epoch:', e)

            ce_trainer.reset()
            for i, batch in enumerate(train_samples_iterator.batches()):
                ce_trainer.train_batch(e, i, batch)

                if test:
                    break

            ce_trainer.reset()
            for i, batch in enumerate(test_samples_iterator.batches()):
                ce_trainer.test_batch(e, i, batch)

                if test:
                    break

        # Save pre-trained agent
        ce_trainer.save_checkpoint(0)
        tf.reset_default_graph()

    ac_trainer = ACTrainer(
        n_slots = len(slots_dict),
        n_actions = len(actions_dict),
        word_embeddings_shape = embeddings.shape,
        save_path = '{0}/ac'.format(base_path), 
        agent_path = (options.agent_checkpoint or '%s-0' % (ce_trainer.checkpoints_path))
    )

    # Pre-train critic
    ac_trainer.GAMMA_CRITIC = 1.0
    for e in range(critic_epochs):
        print('Critic pre-training epoch:', e)

        ac_trainer.reset()
        for i, batch in enumerate(train_samples_iterator.batches()):
            ac_trainer.train_critic_batch(e, i, batch)

            if test:
                break

        ac_trainer.reset()
        for i, batch in enumerate(test_samples_iterator.batches()):
            ac_trainer.test_critic_batch(e, i, batch)

            if test:
                break

    ac_trainer.reset_gammas()
    ac_trainer.set_batch_size(20)
    train_samples_iterator = SamplesIterator(samples_train, batch_size=20)
    test_samples_iterator = SamplesIterator(samples_test, batch_size=20)

    # AC
    for e in range(ac_epochs):
        print('AC epoch:', e)

        ac_trainer.reset()
        for i, batch in enumerate(train_samples_iterator.batches()):
            ac_trainer.train_batch(e, i, batch)

            if test:
                break

        ac_trainer.reset()
        for i, batch in enumerate(test_samples_iterator.batches()):
            ac_trainer.test_batch(e, i, batch)

            if test:
                break

        ac_trainer.save_checkpoint(e)


if __name__ == '__main__':
    train_spec = {
        'batch_size': 64
    }

    test_spec = {
        'batch_size': 64,
        'actor_epochs': 1,
        'critic_epochs': 1,
        'ac_epochs': 1,
        'test': True
    }

    train(**train_spec)
