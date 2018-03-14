import tensorflow as tf
from tensorflow.contrib.seq2seq import CustomHelper

class GreedyEmbeddingTrainingHelper(CustomHelper):
    '''
        Returns embedding with the highest probability at previous step.
    '''

    def __init__(self, start_tokens, sequence_length, embedding):
        def _init():
            return (
                (0 >= sequence_length),
                tf.nn.embedding_lookup(embedding, start_tokens)
            )

        def _sample(time, outputs, state):
            return tf.cast(tf.argmax(outputs, -1), tf.int32)

        def _next_inputs(time, outputs, state, sample_ids):
            return (
                (time >= sequence_length),
                tf.nn.embedding_lookup(embedding, sample_ids),
                state,
            )

        super(GreedyEmbeddingTrainingHelper, self).__init__(
            initialize_fn = _init,
            sample_fn = _sample,
            next_inputs_fn = _next_inputs,
        )

class WeightedEmbeddingSumTrainingHelper(CustomHelper):
    '''
        Uses sum of embeddings weighted by their respective probabilites
        instead of argmax as next input.
    '''

    def __init__(self, inputs, sequence_length, embedding):
        def _init():
            return (
                (0 >= sequence_length),
                inputs[:,0,:]
            )

        def _sample(time, outputs, state):
            return tf.cast(tf.argmax(outputs, -1), tf.int32)

        def _next_inputs(time, outputs, state, sample_ids):
            return (
                (time >= sequence_length),
                tf.matmul(tf.nn.softmax(outputs), embedding),
                state,
            )

        super(WeightedEmbeddingSumHelper, self).__init__(
            initialize_fn = _init,
            sample_fn = _sample,
            next_inputs_fn = _next_inputs
        )

