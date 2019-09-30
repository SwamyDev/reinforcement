try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")

from example.log_utils import log_2d_tensor_as_img


class ValueBaseline:
    def __init__(self, session, obs_dims, summary_writer, lr=0.1):
        self._session = session
        self._in_observations = tf1.placeholder(shape=(None, obs_dims), dtype=tf.float32, name="observations")
        self._in_returns = tf1.placeholder(shape=(None,), dtype=tf.float32, name="returns")
        nn = tf1.get_variable("nn", shape=(obs_dims, 1), dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        self._out_prediction = tf.squeeze(tf.matmul(self._in_observations, nn))
        loss = tf1.math.reduce_mean(tf.math.squared_difference(self._in_returns, self._out_prediction), name="mse_loss")
        self._train = tf1.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

        self._summary_writer = summary_writer
        logs = [tf1.summary.scalar("mean_baseline_return", tf.reduce_mean(self._in_returns)),
                tf1.summary.scalar("baseline_loss", loss),
                log_2d_tensor_as_img("baseline_nn", nn)]
        self._log_summary = tf1.summary.merge(logs)
        self._cur_episode = 0

    def estimate(self, trj):
        return self._session.run(self._out_prediction,
                                 {self._in_observations: trj.observations,
                                  self._in_returns: trj.returns})

    def fit(self, trj):
        _, log = self._session.run([self._train, self._log_summary],
                                   {self._in_observations: trj.observations,
                                    self._in_returns: trj.returns})
        self._summary_writer.add_summary(log, self._cur_episode)
        self._cur_episode += 1
