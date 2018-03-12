import numpy as np

from jamesbot.utils.iterator import StatefulSamplesIterator
from jamesbot.utils.padding import pad_sequences, pad_array_of_complex


class SamplesIterator(StatefulSamplesIterator):

    def batches(self):
        super(SamplesIterator, self).batches()

        while True:
            reset_state = []
            inputs, inputs_length, slot_targets = [], [], []
            previous_output, previous_output_length = [], []
            targets, targets_length = [], []
            action_targets, query_result_states, query_results = [], [], []
            slot_any_targets, slot_any_target_lengths = [], []
            values = []
            
            for batch_sample in self._next_batch():
                reset_state.append(batch_sample['reset_state'])
                inputs.append(batch_sample['token_ids'])
                inputs_length.append(len(batch_sample['token_ids']))
                slot_targets.append(batch_sample['token_slot_ids'])
                previous_output.append(batch_sample['previous_response_delexicalized_token_ids'])
                previous_output_length.append(len(batch_sample['previous_response_delexicalized_token_ids']))
                targets.append(batch_sample['next_response_delexicalized_token_ids'])
                targets_length.append(len(batch_sample['next_response_delexicalized_token_ids']))
                action_targets.append(batch_sample['next_action']),
                query_result_states.append(batch_sample['query_state'])
                query_results.append(batch_sample['query_result'])
                values.append(batch_sample['was_booked'])
                slot_any_targets.append(batch_sample['slot_any'])
                slot_any_target_lengths.append(len(batch_sample['slot_any']))
            
            if len(inputs) != self._batch_size:
                return
            
            max_inputs_length = np.clip(max(inputs_length), 1, self._max_sequence_len)
            max_previous_output_length = np.clip(max(previous_output_length), 1, self._max_sequence_len)
            max_targets_length = np.clip(max(targets_length), 1, self._max_sequence_len)
            
            yield {
                'reset_state': reset_state,
                'inputs': pad_sequences(inputs, max_inputs_length),
                'inputs_length': np.clip(inputs_length, 1, self._max_sequence_len),
                'slot_targets': pad_sequences(slot_targets, max_inputs_length),
                'previous_output': pad_sequences(previous_output, max_previous_output_length),
                'previous_output_length': np.clip(previous_output_length, 1, self._max_sequence_len),
                'targets': pad_sequences(targets, max_targets_length),
                'targets_length': np.clip(targets_length, 1, self._max_sequence_len),
                'action_targets': action_targets,
                'query_result_state': query_result_states,
                'query_result': pad_array_of_complex(query_results),
                'value_targets': values,
                'slot_any_targets': pad_sequences(slot_any_targets, max(slot_any_target_lengths), -1)
            }
