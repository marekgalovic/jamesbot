import numpy as np
import random
import copy

from jamesbot.utils.padding import pad_sequences

def samples_iterator(samples, batch_size = 64, p_shuffle=.3, p_swap=.3):
    def _shuffle(arr):
        random.shuffle(arr)
        return arr
    
    def _swap(arr):
        for i in range(0, len(arr), 2):
            if (np.random.uniform() <= p_swap) and (i < len(arr)-1):
                curr = arr[i]
                arr[i] = arr[i+1]
                arr[i+1] = curr
        return arr

    def _randomize_rows(rows):        
        for row in rows:
            if np.random.uniform() <= p_shuffle:
                yield _shuffle(row)
            else:
                yield _swap(row)
        
        
    batch_size = int(batch_size/2)
    for batch_id in range(int(len(samples) / batch_size)):
        rows = samples[batch_id*batch_size:batch_id*batch_size+batch_size]
        
        random_rows = list(_randomize_rows(copy.deepcopy(rows)))
        
        assert rows != random_rows
                        
        rows += random_rows
        lengths = list(map(len, rows))
        labels = [1]*(len(rows) - len(random_rows)) + [0]*len(random_rows)
        
        assert len(rows) == len(labels)
        
        yield {
            'inputs': pad_sequences(rows, int(np.clip(max(lengths), 0, 50))),
            'inputs_length': np.clip(lengths, 0, 50),
            'labels': labels
        }
