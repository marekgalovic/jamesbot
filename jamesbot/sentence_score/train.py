import json
import tensorflow as tf
import numpy as np
from optparse import OptionParser
from tensorflow.python.lib.io.file_io import FileIO

from util import samples_iterator
from trainer import SentenceScoreModelTrainer

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
samples_train = load_data('samples_train.json')
samples_test = load_data('samples_test.json')

def train(n_epochs, batch_size=64):
    with tf.Graph().as_default():
        trainer = SentenceScoreModelTrainer(embeddings, '{0}/sentence_score_{1}'.format(options.models_dir, options.run_name))
        
        p_shuffle, p_swap = 0.5, 0.7
        for e in range(n_epochs):
            if e > 0:
                p_shuffle /= 1.5
                p_shuffle = float(np.clip(p_shuffle, 0.1, 0.5))
            if e > 1:
                p_swap /= 1.2
                p_swap = float(np.clip(p_swap, 0.3, 0.7))
            
            print('Epoch:', e, 'P(shuffle):', p_shuffle, 'P(swap):', p_swap)
            
            for i, batch in enumerate(samples_iterator(samples_train, p_shuffle=p_shuffle, p_swap=p_swap, batch_size=batch_size)):
                trainer.train_batch(i, batch)
                
            trainer.save_checkpoint(e)

            for i, batch in enumerate(samples_iterator(samples_test, p_shuffle=p_shuffle, p_swap=p_swap, batch_size=batch_size)):
                trainer.test_batch(i, batch)

if __name__ == '__main__':
    train(
        n_epochs = 30,
        batch_size = 64
    )
