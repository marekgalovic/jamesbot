class StatefulSamplesIterator(object):
    
    def __init__(self, samples, batch_size=64, max_sequence_len=50):
        self._samples = samples
        
        self._batch_size = batch_size
        self._max_sequence_len = max_sequence_len
        
    def _next_batch(self):
        for i in range(self._batch_size):
            dialog_id = self._dialog_indices[i]
            turn_id = self._turn_indices[i]
            
            if len(self._samples[dialog_id]) > turn_id:
                yield dict(self._samples[dialog_id][turn_id], reset_state=False)
                
                self._turn_indices[i] += 1
            else:
                if len(self._samples) == self._next_dialog_idx:
                    continue
                    
                yield dict(self._samples[self._next_dialog_idx][0], reset_state=True)
                
                self._dialog_indices[i] = self._next_dialog_idx
                self._turn_indices[i] = 0
                self._next_dialog_idx += 1
                
    def _reset(self):
        self._dialog_indices = list(range(self._batch_size))
        self._next_dialog_idx = self._batch_size
        self._turn_indices = [0]*self._batch_size
                
    def batches(self):
        self._reset()

        return []
