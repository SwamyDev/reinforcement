import numpy as np

try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")

from example.log_utils import log_2d_tensor_as_img
from reinforcement import tf_operations as tf_ops


class ParameterizedPolicy:
    def __init__(self, session, obs_dims, num_actions, summary_writer, lr=10):
        self._session = session
        self._lr = lr
        self._in_actions = tf1.placeholder(shape=(None,), dtype=tf.uint8, name="actions")
        self._in_returns = tf1.placeholder(shape=(None,), dtype=tf.float32, name="returns")
        self._in_observations = tf1.placeholder(shape=(None, obs_dims), dtype=tf.float32, name="observations")
        theta = tf1.get_variable(f"theta", shape=(obs_dims, num_actions), dtype=tf.float32,
                                 initializer=tf.glorot_uniform_initializer())
        self._out_probabilities = tf.nn.softmax(tf.matmul(self._in_observations, theta))
        self._train = None

        self._logs = [tf1.summary.scalar(f"mean_normalized_return", tf.reduce_mean(self._in_returns)),
                      log_2d_tensor_as_img(f"theta", theta)]
        self._log_summary = tf.no_op
        self._summary_writer = summary_writer
        self._cur_episode = 0

    def set_signal_calc(self, signal_calc):
        loss = -signal_calc(tf_ops, self._in_actions, self._out_probabilities, self._in_returns)
        self._train = tf1.train.GradientDescentOptimizer(learning_rate=self._lr).minimize(loss)
        self._session.run(tf1.global_variables_initializer())
        self._finish_logs(loss)

    def _finish_logs(self, loss):
        self._logs.append(tf1.summary.scalar("loss", loss))
        self._log_summary = tf1.summary.merge(self._logs)

    def estimate(self, observation):
        return np.squeeze(
            self._session.run(self._out_probabilities, {self._in_observations: np.array(observation).reshape(1, -1)}))

    def fit(self, trajectory):
        _, log = self._session.run([self._train, self._log_summary],
                                   {self._in_observations: trajectory.observations,
                                    self._in_actions: trajectory.actions,
                                    self._in_returns: trajectory.advantages})
        self._summary_writer.add_summary(log, self._cur_episode)
        self._cur_episode += 1
