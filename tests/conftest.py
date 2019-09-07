import itertools

import numpy as np
import pytest

from reinforcement.trajectories import history_to_trajectory


class TrajectoryBuilder:
    def __init__(self):
        self._history = list()
        self._current_record = None

    def add(self, observation, action, reward):
        self._history.append((observation, action, reward))

    def to_trajectory(self):
        return history_to_trajectory(self._history)


@pytest.fixture
def observation_factory():
    return lambda: np.random.uniform(size=(2, 3))


@pytest.fixture
def observation(observation_factory):
    return observation_factory()


@pytest.fixture
def make_trajectory_builder():
    return TrajectoryBuilder


@pytest.fixture
def make_trajectory(make_trajectory_builder):
    def factory(actions=None, observations=None, returns=None):
        t = make_trajectory_builder()
        for a, o, r in zip(actions or itertools.repeat(0),
                           observations or itertools.repeat(np.zeros((3, 2))),
                           returns or range(0, 3)):
            t.add(o, a, r)
        return t.to_trajectory()

    return factory
