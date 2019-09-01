import numpy as np


class Trajectory:
    def __init__(self, observations, actions, returns):
        self.observations = observations
        self.actions = actions
        self.returns = returns


class TrajectoryBuilder:
    def __init__(self):
        self._history = list()

    def add(self, observation, action, reward):
        self._history.append((observation, action, reward))

    def to_trajectory(self):
        obs, ats, rws = list(zip(*self._history))
        return Trajectory(np.array(obs), np.array(ats), np.array(rws, dtype=np.float32))


class Trajectories:
    def __init__(self):
        self._trajectories = list()

    def add(self, trajectory):
        self._trajectories.append(trajectory)

    @property
    def returns(self):
        for t in self._trajectories:
            yield t.returns

    @property
    def actions(self):
        for t in self._trajectories:
            yield t.actions
