import numpy as np
import tensorflow as tf
from tensorflow._api.v1.compat import v1 as tf1

from reinforcement import np_operations as np_ops, tf_operations as tf_ops


class LinearPolicy:
    def __init__(self, plotter, lr=1):
        self._lr = lr
        self._m = np.random.normal(1, 0.001)  # high slope avoids local maxima - use negative m for demonstration
        self._probs = []
        self._get_signal = None
        self._plotter = plotter(self._m)

    def set_signal_calc(self, signal_calc):
        def np_signal(acts, ps, rs):
            return signal_calc(np_ops, acts, ps, rs)

        self._get_signal = np_signal

    def estimate(self, observation):
        y = (np.tanh(self._m * ((observation[0] * 2) - 1)) + 1) / 2
        p = np.array([y, 1 - y], dtype=np.float)
        self._probs.append(p)
        return p

    def fit(self, trajectory):
        signal = self._get_signal(trajectory.actions, np.array(self._probs, dtype=np.float), trajectory.returns)
        self._m += signal * self._lr
        self._probs.clear()
        self._plotter.record(self._m, trajectory.returns, signal)

    def plot(self):
        self._plotter.show()

    class MatPlotter:
        def __init__(self, m):
            self._m_history = [m]
            self._ret_med_history = [0.0]
            self._signal_history = [0]

        def record(self, m, returns, signal):
            self._m_history.append(m)
            self._signal_history.append(signal)
            self._ret_med_history.append(float(np.mean(returns)))

        def show(self):
            import matplotlib.pyplot as plt
            fig, (mp, lp, rp) = plt.subplots(3, 1)
            fig.suptitle("Numpy Agent")
            mp.plot(self._m_history)
            mp.set_title("slope")
            lp.plot(self._signal_history)
            lp.set_title("signal")
            rp.plot(self._ret_med_history)
            rp.set_title("median return")
            plt.show()

    class NullPlotter:
        def __init__(self, *args, **kwargs):
            pass

        def record(self, *args, **kwargs):
            pass

        def show(self):
            pass


def _log_2d_tensor_as_img(name, mat):
    return tf1.summary.image(name, tf.reshape(mat, shape=(1, mat.shape[0].value, mat.shape[1].value, 1)))


INDEX = 0


class ParameterizedPolicy:
    def __init__(self, session, obs_dims, num_actions, summary_writer, lr=10):
        global INDEX
        self._session = session
        self._lr = lr
        self._in_actions = tf1.placeholder(shape=(None,), dtype=tf.uint8, name="actions")
        self._in_returns = tf1.placeholder(shape=(None,), dtype=tf.float32, name="returns")
        self._in_observations = tf1.placeholder(shape=(None, obs_dims), dtype=tf.float32, name="observations")
        theta = tf1.get_variable(f"theta_{INDEX}", shape=(obs_dims, num_actions), dtype=tf.float32,
                                 initializer=tf.glorot_uniform_initializer())
        self._out_probabilities = tf.nn.softmax(tf.matmul(self._in_observations, theta))
        self._train = None

        self._logs = [tf1.summary.scalar(f"mean_normalized_return_{INDEX}", tf.reduce_mean(self._in_returns)),
                      _log_2d_tensor_as_img(f"theta_{INDEX}", theta)]
        self._log_summary = tf.no_op
        self._summary_writer = summary_writer
        self._cur_episode = 0
        INDEX += 1

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


class ParameterizedPolicyPrefab:
    def __init__(self, session, obs_dims, num_actions, summary_writer, lr):
        self.session = session
        self.obs_dims = obs_dims
        self.num_actions = num_actions
        self.summary_writer = summary_writer
        self.lr = lr

    def make(self):
        return ParameterizedPolicy(self.session, self.obs_dims, self.num_actions, self.summary_writer, self.lr)

    def setup_for(self, env):
        self.obs_dims = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
