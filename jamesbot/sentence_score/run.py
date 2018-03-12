import json
import re
import tensorflow as tf
import numpy as np
from optparse import OptionParser
from tensorflow.python.lib.io.file_io import FileIO

from jamesbot.utils.tokenization import tokenize
from model import SentenceScoreModel

parser = OptionParser()
parser.add_option('--data-dir', dest='data_dir')
parser.add_option('--checkpoint', dest='checkpoint')

options, _ = parser.parse_args()
print('Data dir:', options.data_dir)

def load_data(name):
    fullpath = '{0}/{1}'.format(options.data_dir, name)
    print('Load:', fullpath)
    return json.load(FileIO(fullpath, 'r'))

word_dict = load_data('word_dictionary.json')

class SentenceScore:
    
    def __init__(self, word_dict, path):
        self._sess = tf.Session()
        self._word_dict = word_dict
        self._word_index = {val: key for key, val in word_dict.items()}
        
        self._inputs = tf.placeholder(tf.int32, [None, None])
        self._inputs_length = tf.placeholder(tf.int32, [None])
        
        self._model = SentenceScoreModel(self._inputs, self._inputs_length, [len(word_dict), 300], 0.0)
        self._model.saver.restore(self._sess, path)
        
    def _embed(self, text):
        return [self._word_dict.get(token, 2) for token in tokenize(text)]
    
    def score(self, text):
        token_ids = self._embed(text)
        
        return self._sess.run(self._model.p, feed_dict = {
            self._inputs: [token_ids],
            self._inputs_length: [len(token_ids)]
        })

with tf.Graph().as_default():
    model = SentenceScore(word_dict, options.checkpoint)

    while True:
        score = model.score(input('> '))
        print('Score: %.4f' % score)
