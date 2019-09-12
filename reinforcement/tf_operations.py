try:
    import tensorflow as tf
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")


def reduce_sum(*args, **kwargs):
    return tf.math.reduce_sum(*args, **kwargs)


def reduce_mean(*args, **kwargs):
    return tf.math.reduce_mean(*args, **kwargs)


def log(*args, **kwargs):
    return tf.math.log(*args, **kwargs)


def one_hot(*args, **kwargs):
    return tf.one_hot(*args, **kwargs)
