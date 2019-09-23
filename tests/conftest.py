try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")

import contextlib
import itertools

import numpy as np
import pytest
from _pytest.runner import runtestprotocol

from reinforcement.trajectories import history_to_trajectory


def pytest_addoption(parser):
    parser.addoption(
        "--log-tensorboard", action="store_true", default=False, help="log tensorboard information for certain tests"
    )
    parser.addoption(
        "--plot-policy", action="store_true", default=False, help="plots policy parameter information for certain tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "stochastic(sample_size, max_samples=None): mark a test as stochastic running it `sample_size` times"
                   " and providing access to statistical analysis of the runs via the injected `stochastic_run` "
                   "fixture. Optionally specify `max_samples`. By default it is equal to `sample_size`. If it is "
                   "bigger, then additional samples are drawn if the test fails after taking `sample_size` samples, up"
                   "to the specified `max_samples`. If it is smaller than `sample_size` it is capped to `sample_size`")


class StochasticRunRecorder:
    def __init__(self):
        self._results = list()

    @contextlib.contextmanager
    def current_run(self):
        self._results.clear()
        yield

    def record(self, result):
        self._results.append(result)

    def count(self, result_equals):
        return self._results.count(result_equals)

    def __len__(self):
        return len(self._results)


_RECORDER = StochasticRunRecorder()


def pytest_runtest_protocol(item, nextitem):
    m = _get_marker(item)
    if m is None:
        return None

    reports = None
    with _RECORDER.current_run():
        n, max_n = _get_sample_range(m)
        s = 0
        while s < n:
            item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
            reports = runtestprotocol(item, nextitem=nextitem, log=False)
            s += 1
            if s == n and _has_failed(reports):
                n = min(n + 1, max_n)

    _report_last_run(item, reports)
    return True


def _get_marker(item):
    try:
        return item.get_closest_marker("stochastic")
    except AttributeError:
        return item.get_marker("stochastic")


def _get_sample_range(m):
    min_n = m.kwargs.get('sample_size', m.args[0])
    max_n = max(m.kwargs.get('max_samples', min_n), min_n)
    return min_n, max_n


def _has_failed(reports):
    return any(r.failed for r in reports if r.when == 'call')


def _report_last_run(item, reports):
    for r in reports:
        item.ihook.pytest_runtest_logreport(report=r)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)


@pytest.fixture
def stochastic_run(request):
    return _RECORDER


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
