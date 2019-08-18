import tensorflow as tf
import string

DICT = list(string.ascii_lowercase) + [' ', '_']

def join_string(string_tensor):
    '''
    Join the strings.
    e.g. [['a', 'b', ''], ['a', 'b', 'c']] -> ['ab', 'abc']

    Args:
        string_tensor: Tensor.
    
    Returns:
        The joined string: Tensor. 0-D or 1-D.
    '''
    i = tf.constant(0)
    string_num = tf.shape(string_tensor)[-1] # on the last dimension
    if tf.rank(string_tensor) == 1:
        joined_string = ''
    else:
        joined_string = tf.tile([''],
                                tf.expand_dims(tf.shape(string_tensor)[0], 0))
    
    loop_cond = lambda i, joined_string: tf.less(i, string_num)
    loop_body = lambda i, joined_string: [tf.add(i, 1), joined_string+string_tensor[...,i]]
    i, joined_string = tf.while_loop(loop_cond, loop_body, 
                                    [i, joined_string], back_prop=False)

    return joined_string


def charlist2string(char_list):
    '''
    Convert a char list to string.
    e.g. ['a', 'b'] -> ['ab']

    Args:
        char_list: Tensor or SparseTensor.
    
    Returns:
        Tensor.
    '''

    if isinstance(char_list, tf.SparseTensor):
        dense_list = tf.sparse_tensor_to_dense(char_list)
    else:
        dense_list = char_list
    
    return join_string(dense_list)


def char2word(predicted_char_list):
    '''
    Convert char list to word list. Words are divides by space.
    e.g. [['a', 'b', ' ', 'd']] -> [['ab', 'd']]

    Args:
        predicted_char_list: SparseTensor of shape (batch_size, char_len)

    Returns:
        SparseTensor of shape (batch_size, word_len)
    '''

    string = charlist2string(predicted_char_list) # [['ab c']]
    word_list = tf.string_split(string) # [['ab', 'c']]
    return word_list


def indices2string(predictions, dic=DICT):
    '''
    map int64. to char.
    e.g.
        [0, 1, 2] -> ['a', 'b', 'c']
    
    Args:
        predictions: Tensor or SparseTensor. int64 indices to convert.
    
    Returns:
        Map the corresponding values in the DICT.
    '''
    if predictions.dtype != tf.int64:
        predictions = tf.cast(predictions, tf.int64)
    
    index2string_table = tf.contrib.lookup.index_to_string_table_from_tensor(
        dic, default_value='_')
    return index2string_table.lookup(predictions)



def string2char_list(string):
    """convert a string to char based list. For example:
    [ 'ab'] -> [ 'a', 'b']

    Args:
        string: `1-D` string `Tensor`. The string to convert

    Returns: `2-D` string `SparseTensor`. The string are split to list of chars.
    """
    return tf.string_split(string, delimiter='')


def string2indices(s, dic=DICT):
    """ map the char values in `s` to numeric int64 indices
    For example:
        [ 'a', 'b', 'c'] -> [ 0, 1, 2]

    Args:
        s: Tensor or SparseTensor.

    Returns: The same type of input. the Dtype is `tf.int64`

    """
    string2index_table = tf.contrib.lookup.index_table_from_tensor(
        dic, num_oov_buckets=1, default_value=-1)
    return string2index_table.lookup(s)