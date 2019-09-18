try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")

import numpy as np

from tests.aux.policies import _log_2d_tensor_as_img


class FixedBaseline:
    def __init__(self, value):
        self.value = value

    def estimate(self, trj):
        return np.ones_like(trj.returns) * self.value

    def fit(self, trj):
        pass


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
                _log_2d_tensor_as_img("baseline_nn", nn)]
        self._log_summary = tf1.summary.merge(logs)
        self._cur_episode = 0

    def estimate(self, trj):
        r = trj.returns - np.ones_like(trj.returns)
        return self._session.run(self._out_prediction,
                                 {self._in_observations: trj.observations,
                                  self._in_returns: r})

    def fit(self, trj):
        r = trj.returns - np.ones_like(trj.returns)
        _, log = self._session.run([self._train, self._log_summary],
                                   {self._in_observations: trj.observations,
                                    self._in_returns: r})
        self._summary_writer.add_summary(log, self._cur_episode)
        self._cur_episode += 1


class FixedBaselinePrefab:
    def __init__(self, value):
        self.value = value

    def make(self):
        return FixedBaseline(self.value)

    def setup_for(self, _):
        pass


class ValueBaselinePrefab:
    def __init__(self, session, obs_dims, summary_writer, lr):
        self.session = session
        self.obs_dims = obs_dims
        self.summary_writer = summary_writer
        self.lr = lr

    def make(self):
        return ValueBaseline(self.session, self.obs_dims, self.summary_writer, self.lr)

    def setup_for(self, env):
        self.obs_dims = env.observation_space.shape[0]
