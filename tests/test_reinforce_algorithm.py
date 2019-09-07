import numpy as np
import pytest

from reinforcement.algorithm.reinforce import Reinforce
from reinforcement.trajectories import Trajectories

CALL_ORDER = list()


class OperationsSim:
    @staticmethod
    def reduce_mean(array):
        return np.mean(array)

    @staticmethod
    def log(array):
        return np.log(array).T


class PolicySpy:
    def __init__(self):
        self.received_returns = None
        self.received_loss_signal = None
        self._calc = None

    def set_signal_calc(self, calc):
        self._calc = calc

    def fit(self, batch):
        CALL_ORDER.append(self.fit)
        self.received_returns = list(batch.returns)
        sim = OperationsSim()
        losses = [self._calc(sim, a, r) for a, r in zip(batch.actions, batch.returns)]
        self.received_loss_signal = np.mean(losses)


class PolicySim(PolicySpy):
    def __init__(self):
        super().__init__()
        self._estimations = {}
        self._calc = None

    def estimate(self, obs):
        roll = np.random.uniform(size=(1, 5))
        key = str(obs)
        self._estimations[key] = np.exp(roll) / np.sum(np.exp(roll))
        return self._estimations[key]

    def estimate_for(self, obs):
        try:
            return self._estimations[str(obs)]
        except KeyError:
            raise AssertionError(f"The requested observation has never used in any estimations:\n"
                                 f"{obs}\nEstimations: {self._estimations.keys()}")

    def __repr__(self):
        return f"ApproximationSim() - {len(self._estimations)} estimations done"


class TrajectoriesStub(Trajectories):
    def set_trajectories(self, *trajectories):
        self._trajectories.clear()
        for t in trajectories:
            self.add(t)


class BaselineStub:
    def __init__(self, size):
        self._estimates = np.random.uniform(size)

    def set_estimates(self, estimates):
        self._estimates = estimates

    def estimate(self, _):
        return self._estimates


class BaselineSpy(BaselineStub):
    def estimate(self, _):
        CALL_ORDER.append(self.estimate)
        return super().estimate(_)


@pytest.fixture(autouse=True)
def cleanup_call_order():
    CALL_ORDER.clear()


@pytest.fixture
def policy():
    return PolicySim()


@pytest.fixture
def trajectories(make_trajectory):
    t = TrajectoriesStub()
    t.set_trajectories(make_trajectory(), make_trajectory())
    return t


@pytest.fixture
def baseline():
    return BaselineSpy(size=(2, 3))


def test_sampling_returns_estimate_of_action_probabilities(policy, observation):
    alg = make_alg(policy)
    assert_estimates(alg.sample(observation), policy.estimate_for(observation))


def make_alg(policy, baseline=None, gamma=0.9):
    return Reinforce(policy, gamma, baseline)


def assert_estimates(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_reinforce_calculated_correct_reward_signal(policy, trajectories, baseline, make_trajectory):
    alg = make_alg(policy, baseline, gamma=0.5)
    baseline.set_estimates([
        [2, 3, 4],
        [5, 6]
    ])
    trajectories.set_trajectories(make_trajectory(returns=[3, 4, 5]),
                                  make_trajectory(returns=[6, 7]))
    alg.optimize(trajectories)
    assert_signals(policy.received_returns, [normalized([4.25, 3.5, 1.0]),
                                             normalized([4.50, 1.0])])


def normalized(advantages):
    return (np.array(advantages, dtype=np.float32) - np.mean(advantages)) / (np.std(advantages) + 1e-8)


def assert_signals(actual, expected):
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


def test_make_sure_baseline_estimation_is_done_before_model_fitting(policy, trajectories, baseline):
    alg = make_alg(policy, baseline)
    alg.optimize(trajectories)
    assert CALL_ORDER == [baseline.estimate, policy.fit]


def test_fitting_the_approximation_uses_the_correct_loss(policy, trajectories, baseline, make_trajectory):
    alg = make_alg(policy, baseline, gamma=0.9)
    baseline.set_estimates([[1, 1], [1, 1]])
    trajectories.set_trajectories(make_trajectory(actions=[[0.2, 0.8], [0.9, 0.1]], returns=[3, 5]),
                                  make_trajectory(actions=[[0.4, 0.6], [0.8, 0.2]], returns=[3, 5]))
    alg.optimize(trajectories)
    assert policy.received_loss_signal == np.mean([np.log([0.2, 0.8]) * 1, np.log([0.9, 0.1]) * -1,
                                                   np.log([0.4, 0.6]) * 1, np.log([0.8, 0.2]) * -1])
