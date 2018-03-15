import tensorflow as tf
import numpy as np
import random


def add_pad_eos(indices, sequence_length, eos_token = 1, pre_pad = True):
    batch_size, max_length = tf.unstack(tf.shape(indices))
    
    pad = tf.zeros([batch_size, 1], dtype=tf.int32)
    if pre_pad:
        eos = tf.one_hot(sequence_length+1, max_length+2, dtype=tf.int32) * eos_token
        return tf.concat([pad, indices, pad], 1) + eos
    else:
        eos = tf.one_hot(sequence_length, max_length+2, dtype=tf.int32) * eos_token
        return tf.concat([indices, pad, pad], 1) + eos


def pad_sequences(sequences, max_len=None, pad_value=0, dtype=np.int32):
    '''
    :param sequences: An array of arrays of different lengths that will be padded to [len(sequences) x max_len] matrix
    '''
    if max_len is None:
        max_len = max(map(len, sequences))
        
    result = []
    for sequence in sequences:
        if len(sequence) < max_len:
            result.append(sequence + [pad_value]*(max_len - len(sequence)))
        if len(sequence) >= max_len:
            result.append(sequence[:max_len])
    return np.array(result, dtype=dtype)


def pad_complex(struct, max_keys_len, max_values_len, pad_value=0, shuffle=True):
    '''
    :param struct: Dictionary of arrays. 
    :returns: Vector of slot indices, matrix of value indices, number of slots and a vector of value lengths.
    '''
    if len(struct) == 0:
        return (
            np.zeros(shape=(max_keys_len), dtype=np.int32),
            np.zeros(shape=(max_keys_len, max_values_len), dtype=np.int32)
        )
    
    struct_items = list(struct.items())
    if shuffle == True:
        random.shuffle(struct_items)
        
    keys, values = [], []
    for (key, value) in struct_items:
        keys.append(key)
        values.append(value)
    if len(keys) < max_keys_len:
        keys += [pad_value]*(max_keys_len-len(keys))
        values += [[pad_value]]*(max_keys_len-len(values))
    
    return (
        np.array(keys).astype(int),
        pad_sequences(values, max_values_len)
    )

def pad_array_of_complex(structs):
    '''
    :param structs: An array of structs
    :returns: Padded keys, values, struct sizes and value lengths
    '''
    key_counts = [len(struct) for struct in structs]
    max_keys_count = np.clip(max(key_counts), 1, 15)
    
    value_lengths = [[len(value) for (_, value) in struct.items()] for struct in structs]
    max_values_length = np.clip(max([(max(lens) if len(lens) > 0 else 0) for lens in value_lengths]), 1, 20)
    
    if len(structs) == 0:
        return (
            np.zeros(shape=(1,max_keys_count), dtype=np.int32),
            np.zeros(shape=(1,max_keys_count,max_values_length), dtype=np.int32),
            np.zeros(shape=(1,), dtype=np.int32),
            np.zeros(shape=(1,max_keys_count), dtype=np.int32)
        )
    
    keys_padded, values_padded = [], []
    for struct in structs:
        keys, values = pad_complex(struct, max_keys_count, max_values_length)
        keys_padded.append(keys)
        values_padded.append(values)
    
    keys_padded = np.array(keys_padded, dtype=np.int32)
    values_padded = np.array(values_padded, dtype=np.int32)
    
    for i in range(len(structs)):
        if len(value_lengths[i]) < max_keys_count:
            value_lengths[i] += [0]*(max_keys_count-len(value_lengths[i]))
    
    return {
        'slots': keys_padded,
        'values': values_padded,
        'slots_count': np.array(key_counts, dtype=np.int32),
        'values_length': np.array(value_lengths, dtype=np.int32)
    }
