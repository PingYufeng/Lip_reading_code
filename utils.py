import tensorflow as tf


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