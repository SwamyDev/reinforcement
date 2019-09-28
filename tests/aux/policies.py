from example.policies import ParameterizedPolicy

try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")

import numpy as np

from reinforcement import np_operations as np_ops


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
