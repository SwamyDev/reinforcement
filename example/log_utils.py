try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")


def log_2d_tensor_as_img(name, mat):
    return tf1.summary.image(name, tf.reshape(mat, shape=(1, mat.shape[0].value, mat.shape[1].value, 1)))


class NoLog:
    def add_summary(self, *args, **kwargs):
        pass

    def add_graph(self, *args, **kwargs):
        pass
