import json
import re
import tensorflow as tf
import numpy as np
from optparse import OptionParser
from tensorflow.python.lib.io.file_io import FileIO

from jamesbot.utils.embeddings import EmbeddingHandler
from jamesbot.utils.padding import pad_array_of_complex
from jamesbot.utils.tokenization import tokenize

from model import Agent

from jamesbot.db import query


parser = OptionParser()
parser.add_option('--data-dir', dest='data_dir')
parser.add_option('--checkpoint', dest='checkpoint')

options, _ = parser.parse_args()
print('Data dir:', options.data_dir)

def load_data(name):
    fullpath = '{0}/{1}'.format(options.data_dir, name)
    print('Load:', fullpath)
    return json.load(FileIO(fullpath, 'r'))


word_embeddings = EmbeddingHandler(load_data('word_dictionary.json'))
slot_embeddings = EmbeddingHandler(load_data('slots_dictionary.json'))
action_embeddings = EmbeddingHandler(load_data('actions_dictionary.json'))


class ChatSession(object):

    DILATATION_RATES = [4, 2, 1]

    def __init__(self, word_embeddings: EmbeddingHandler, slot_embeddings: EmbeddingHandler, action_embeddings: EmbeddingHandler, checkpoint, hidden_size=300):
        self._sess = tf.Session()
        self._checkpoint = checkpoint
        self._hidden_size = hidden_size
        self._i = 0
        self._state_size = 2*self._hidden_size

        self._word_embeddings = word_embeddings
        self._slot_embeddings = slot_embeddings
        self._action_embbeddings = action_embeddings

        self._build_agent()
        self.reset_state()

    def reset_state(self):
        self._state_memory = np.zeros((4, 3, 1, self._state_size))
        self._previous_output = [0]
        self._query_params = {}
        self._result = {}

    def _update_state(self, new_states):
        new_states = np.asarray(new_states)

        for cell_id, dilatation in enumerate(self.DILATATION_RATES):
            location_update_id = (dilatation - 1) - self._i % dilatation
            self._state_memory[location_update_id,cell_id,0,:] = new_states[cell_id,0,:]

    def _get_state(self):
        result = []
        for cell_id, dilatation in enumerate(self.DILATATION_RATES):
            location_id = self._i % dilatation
            result.append(self._state_memory[location_id,cell_id,:,:])

        return np.asarray(result)
        
    def _parse_slots(self, message, slots, slot_any):
        tokens = tokenize(message, replace_digits=False)
        slots = self._slot_embeddings.tokens(slots[0])

        parsed = {}
        for i, slot in enumerate(slots):
            if slot != '<NO_SLOT>':
                if slot in parsed:
                    parsed[slot].append(tokens[i])
                else:
                    parsed[slot] = [tokens[i]]

        for slot_any_id in set(np.argwhere(slot_any[0]).reshape((-1)).tolist()):
            parsed[self._slot_embeddings.get_token(slot_any_id)] = None

        return dict(self._query_params, **{key: ' '.join(val) if isinstance(val, list) else val for key, val in parsed.items()})

    def _query_db(self, action):
        # Parse slots and update state
        if action == 'query':
            print('Query:', self._query_params)
            results = query(self._query_params)
            if len(results) == 0:
                print('NO RESULT')
                return 1, {}
            print(results[0])
            return 2, results[0]

        return 0, {}

    def _build_agent(self):
        self.agent = Agent(
            word_embeddings_shape = [len(self._word_embeddings), 300],
            n_slots = len(self._slot_embeddings), n_actions = len(self._action_embbeddings),
            hidden_size = self._hidden_size, scope='target_agent',
            decoder_max_iter = 50
        )

        self.agent.saver.restore(self._sess, self._checkpoint)

    def _embed_complex(self, struct):
        embedded = {}
        for key in struct.keys():
            embedded[self._slot_embeddings.get_embedding(key)] = self._word_embeddings.embeddings(tokenize(struct[key]))

        return embedded

    def _fill_slots(self, tokens):
        result = []
        pattern = re.compile('\.SLOT\.(\w+)')
        for token in tokens:
            match = pattern.match(token)
            if match is not None:
                slot = match.group(1)
                result.append(str(self._result.get(slot, self._query_params.get(slot, token))))
                continue
            result.append(token)

        return result

    def _feed_dict(self, inputs):
        if len(self._query_result) == 0 or self._query_state < 1:
            query_result_embedded = {'slots': [[0]], 'values': [[[0]]], 'slots_count': [1], 'values_length': [[1]]}
        else:
            query_result_embedded = pad_array_of_complex([self._embed_complex(self._query_result)])

        return {
            self.agent.previous_context_state: self._get_state(),
            self.agent.inputs: [inputs],
            self.agent.inputs_length: [len(inputs)],
            self.agent.previous_output: [self._previous_output],
            self.agent.previous_output_length: [len(self._previous_output)],
            self.agent.query_result_state: [self._query_state],
            self.agent.query_result_slots: query_result_embedded['slots'],
            self.agent.query_result_values: query_result_embedded['values'],
            self.agent.query_result_slots_count: query_result_embedded['slots_count'],
            self.agent.query_result_values_length: query_result_embedded['values_length']
        }

    def response(self, message):
        message_tokens = tokenize(message)
        inputs = self._word_embeddings.embeddings(message_tokens)

        # Parse slots
        slots, slot_any, actions, new_state = self._sess.run([self.agent.slot_ids, self.agent.slot_any, self.agent.action_ids, self.agent.context_state], feed_dict={
            self.agent.previous_context_state: self._get_state(),
            self.agent.inputs: [inputs],
            self.agent.inputs_length: [len(inputs)],
            self.agent.previous_output: [self._previous_output],
            self.agent.previous_output_length: [len(self._previous_output)],
            self.agent.query_result_state: [0],
            self.agent.query_result_slots: [[0]],
            self.agent.query_result_values: [[[0]]],
            self.agent.query_result_slots_count: [1],
            self.agent.query_result_values_length: [[1]]
        })
        # Update query params
        self._query_params = self._parse_slots(message, slots, slot_any)
        # Query DB if action=query
        action = self._action_embbeddings.get_token(actions[0])
        self._query_state, self._query_result = self._query_db(action)
        if len(self._query_result) > 0:
            self._result = self._query_result

        if action == 'query':
            self._update_state(new_state)
            self._i += 1

        # Generate response
        outputs, new_state, actions, value = self._sess.run(
            [self.agent.decoder_token_ids, self.agent.context_state, self.agent.action_ids, self.agent.value],
            feed_dict=self._feed_dict(inputs)
        )

        self._previous_output = outputs[0]
        self._update_state(new_state)

        self._i += 1
        return (
            ' '.join(self._fill_slots(self._word_embeddings.tokens(outputs[0]))),
            self._action_embbeddings.get_token(actions[0]),
            value
        )

session = ChatSession(word_embeddings, slot_embeddings, action_embeddings, options.checkpoint)

print('Agent: Hi. How can I help you?')
while True:
    inputs = input('> ')
    response, action, value = session.response(inputs)
    print('Agent (%s, %.2f):' % (action, value*100), response)
    # print(query(session._query_params))
    if action == 'end':
        exit()
    print()
