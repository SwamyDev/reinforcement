from example.baselines import ValueBaseline

try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")

import numpy as np


class FixedBaseline:
    def __init__(self, value):
        self.value = value

    def estimate(self, trj):
        return np.ones_like(trj.returns) * self.value

    def fit(self, trj):
        pass


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
