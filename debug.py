import tensorflow as tf


def print_value(tensor, **kwargs):
    return tf.Print(tensor,
                    data=[tensor],
                    message="Value of {}: ".format(tensor.name),
                    **kwargs)


def print_num_nans(tensor, **kwargs):
    return tf.Print(tensor,
                    data=[tf.count_nonzero(tf.is_nan(tensor))],
                    message="NaNs in {}: ".format(tensor.name),
                    **kwargs)


def print_num_zeros(tensor, **kwargs):
    return tf.Print(tensor,
                    data=[tf.count_nonzero(tf.equal(tensor, 0))],
                    message="Zeros in {}: ".format(tensor.name),
                    **kwargs)


def print_shape(tensor, **kwargs):
    return tf.Print(tensor,
                    data=[tf.shape(tensor)],
                    message="Shape of {}: ".format(tensor.name),
                    **kwargs)


def print_range(tensor, **kwargs):
    return tf.Print(tensor,
                    data=[tf.reduce_min(tensor), tf.reduce_max(tensor)],
                    message="Range of {}: ".format(tensor.name),
                    **kwargs)

