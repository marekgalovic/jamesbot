import tensorflow as tf

from model import SentenceScoreModel

class SentenceScoreModelTrainer(object):
    
    def __init__(self, embeddings, save_path):
        self._sess = tf.Session()
        
        self._embeddings_shape = [len(embeddings), len(embeddings[0])]
        self._save_path = str(save_path)
        
        self._inputs = tf.placeholder(tf.int32, [None, None])
        self._inputs_length = tf.placeholder(tf.int32, [None])
        self._labels = tf.placeholder(tf.int32, [None])
        self._dropout = tf.placeholder(tf.float32, [])
        
        self._model = SentenceScoreModel(self._inputs, self._inputs_length, self._embeddings_shape, dropout=self._dropout)
        
        self._loss()
        self._optimizer()
        self._metrics()
        
        # Init TF variables and embeddings
        self._sess.run(tf.global_variables_initializer())
        self.initialize_word_embeddings(embeddings)
        
    def initialize_word_embeddings(self, embeddings):
        embeddings_ph = tf.placeholder(tf.float32, self._embeddings_shape)
        init_op = self._model._word_embeddings.assign(embeddings_ph)
        
        return self._sess.run(init_op, feed_dict={embeddings_ph: embeddings})
    
    def save_checkpoint(self, step):
        print('Write checkpoint:', self._model.saver.save(self._sess, '{0}/checkpoints/model.ckpt'.format(self._save_path), global_step=step))
        
    def _loss(self):
        self.loss = tf.losses.log_loss(
            predictions = self._model.p,
            labels = self._labels
        )
        
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self._labels, tf.cast(tf.greater(self._model.p, 0.5), tf.int32)), tf.float32))

        _labels = tf.cast(self._labels, tf.float32)
        true_p = tf.reduce_sum(_labels * self._model.p) / tf.reduce_sum(_labels)
        randomized_p = tf.reduce_sum((1. - _labels) * self._model.p) / tf.reduce_sum(1. - _labels)
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('true_p', true_p)
        tf.summary.scalar('randomized_p', randomized_p)
        
    def _optimizer(self):
        self._train_op = tf.train.AdamOptimizer().minimize(self.loss)
        
    def _metrics(self):
        print('Metrics path {0}/metrics/'.format(self._save_path))
        
        self._train_writer = tf.summary.FileWriter('{0}/metrics/train'.format(self._save_path), self._sess.graph)
        self._test_writer = tf.summary.FileWriter('{0}/metrics/test'.format(self._save_path), self._sess.graph)
        self._metrics_op = tf.summary.merge_all()
        
    def _feed_dict(self, batch, opts={}):
        fd = {
            self._inputs: batch['inputs'],
            self._inputs_length: batch['inputs_length'],
            self._labels: batch['labels'],
            self._dropout: 0.4
        }
        
        for key, val in opts.items():
            fd[key] = val
        
        return fd
        
    def train_batch(self, i, batch):
        _, metrics_val = self._sess.run(
            [self._train_op, self._metrics_op],
            feed_dict = self._feed_dict(batch)
        )
        
        if i % 20 == 0:
            self._train_writer.add_summary(metrics_val)
    
    def test_batch(self, i, batch):
        metrics_val = self._sess.run(
            self._metrics_op,
            feed_dict = self._feed_dict(batch, {self._dropout: 0.0})
        )
        
        if i % 20 == 0:
            self._test_writer.add_summary(metrics_val)
