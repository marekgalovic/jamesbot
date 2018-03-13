import json
import tensorflow as tf
import numpy as np
from optparse import OptionParser
from tensorflow.python.lib.io.file_io import FileIO

from utils import SamplesIterator
from trainer import SupervisedTrainer


parser = OptionParser()
parser.add_option('--data-dir', dest='data_dir')
parser.add_option('--job-dir', dest='models_dir')
parser.add_option('--run-name', dest='run_name')

options, _ = parser.parse_args()
print('Data dir:', options.data_dir)
print('Models dir:', options.models_dir)
print('Run name:', options.run_name)


def load_data(name):
    fullpath = '{0}/{1}'.format(options.data_dir, name)
    print('Load:', fullpath)
    return json.load(FileIO(fullpath, 'r'))


embeddings = np.asarray(load_data('embeddings.json'))
slots_dict = load_data('slots_dictionary.json')
actions_dict = load_data('actions_dictionary.json')

samples_train = load_data('embedded_frames_train.json')
samples_test = load_data('embedded_frames_test.json')

def train(n_epochs, batch_size=64):
    train_samples_iterator = SamplesIterator(samples_train, batch_size=batch_size)
    test_samples_iterator = SamplesIterator(samples_test, batch_size=batch_size)

    trainer = SupervisedTrainer(
        n_slots=len(slots_dict),
        n_actions=len(actions_dict),
        word_embeddings_shape=embeddings.shape,
        save_path='{0}/ac_agent_{1}'.format(options.models_dir, options.run_name),
        batch_size=batch_size
    )

    trainer._sess.run(tf.global_variables_initializer())
    trainer.initialize_word_embeddings(embeddings)

    for e in range(n_epochs):
        print('Epoch:', e)

        trainer.reset()
        for i, batch in enumerate(train_samples_iterator.batches()):
            trainer.train_batch(e, i, batch)

        trainer.save_checkpoint(e)

        trainer.reset()
        for i, batch in enumerate(test_samples_iterator.batches()):
            trainer.test_batch(e, i, batch)


if __name__ == '__main__':
    train(
        n_epochs = 20,
        batch_size = 64
    )
