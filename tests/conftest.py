try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")

import itertools

import numpy as np
import pytest

from reinforcement.trajectories import history_to_trajectory


def pytest_addoption(parser):
    parser.addoption(
        "--log-tensorboard", action="store_true", default=False, help="log tensorboard information for certain tests"
    )
    parser.addoption(
        "--plot-policy", action="store_true", default=False, help="plots policy parameter information for certain tests"
    )


class TrajectoryBuilder:
    def __init__(self):
        self._history = list()
        self._current_record = None

    def add(self, action, observation, reward):
        self._history.append((action, observation, reward))

    def to_trajectory(self):
        t = history_to_trajectory(self._history)
        self._history.clear()
        return t


@pytest.fixture
def log_tensorboard(request):
    return request.config.getoption("--log-tensorboard")


@pytest.fixture
def plot_policy(request):
    return request.config.getoption("--plot-policy")


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
        for a, o, r in zip(actions if actions is not None else itertools.repeat(0),
                           observations if observations is not None else itertools.repeat(np.zeros((3, 2))),
                           returns if returns is not None else range(0, 3)):
            t.add(a, o, r)
        return t.to_trajectory()

    return factory


@pytest.fixture
def session():
    tf1.reset_default_graph()
    with tf1.Session() as s:
        yield s


@pytest.fixture
def summary_writer(session, log_tensorboard, request):
    def writer():
        if log_tensorboard:
            import shutil
            ld = f"{request.node.name}_log"
            shutil.rmtree(ld, ignore_errors=True)
            return tf.summary.FileWriter(ld, session=session)
        else:
            class _NoLog:
                def add_summary(self, *args, **kwargs):
                    pass

                def add_graph(self, *args, **kwargs):
                    pass

            return _NoLog()

    w = writer()
    yield w
    w.add_graph(session.graph)
