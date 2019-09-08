import numpy as np
import pytest

import reinforcement.np_operations as np_ops
from reinforcement.algorithm.reinforce import Reinforce
from reinforcement.np_operations import softmax

CALL_ORDER = list()


class PolicySpy:
    def __init__(self):
        self.received_returns = None
        self._calc = None

    def set_signal_calc(self, calc):
        self._calc = calc

    def fit(self, trajectory):
        CALL_ORDER.append(self.fit)
        self.received_returns = list(trajectory.returns)


class PolicySim(PolicySpy):
    def __init__(self):
        super().__init__()
        self._estimations = {}
        self.num_actions = 5
        self.received_loss_signal = None

    def estimate(self, obs):
        roll = np.random.uniform(size=(self.num_actions,))
        key = str(obs)
        self._estimations[key] = softmax(roll)
        return self._estimations[key]

    def estimate_for(self, obs):
        try:
            return self._estimations[str(obs)]
        except KeyError:
            raise AssertionError(f"The requested observation has never used in any estimations:\n"
                                 f"{obs}\nEstimations: {self._estimations.keys()}")

    def fit(self, trajectory):
        super().fit(trajectory)
        probabilities = np.array([self.estimate(o) for o in trajectory.observations])
        self.received_loss_signal = self._calc(np_ops, trajectory.actions, probabilities, trajectory.returns)

    def __repr__(self):
        return f"ApproximationSim() - {len(self._estimations)} estimations done"


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
def trajectory(make_trajectory):
    return make_trajectory()


@pytest.fixture
def baseline():
    return BaselineSpy(size=(3,))


def test_sampling_returns_estimate_of_action_probabilities(policy, observation):
    alg = make_alg(policy)
    assert_estimates(alg.sample(observation), policy.estimate_for(observation))


def make_alg(policy, baseline=None, gamma=0.9):
    return Reinforce(policy, gamma, baseline)


def assert_estimates(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_reinforce_calculated_correct_reward_signal(policy, baseline, make_trajectory):
    alg = make_alg(policy, baseline, gamma=0.5)
    baseline.set_estimates([2, 3, 4])
    alg.optimize(make_trajectory(returns=[3, 4, 5]))
    assert_signals(policy.received_returns, normalized([4.25, 3.5, 1.0]))


def normalized(advantages):
    return (np.array(advantages, dtype=np.float32) - np.mean(advantages)) / (np.std(advantages) + 1e-8)


def assert_signals(actual, expected):
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


def test_make_sure_baseline_estimation_is_done_before_model_fitting(policy, trajectory, baseline):
    alg = make_alg(policy, baseline)
    alg.optimize(trajectory)
    assert CALL_ORDER == [baseline.estimate, policy.fit]


def test_fitting_the_approximation_uses_the_correct_loss(policy, baseline, make_trajectory):
    policy.num_actions = 2
    alg = make_alg(policy, baseline, gamma=0.9)
    baseline.set_estimates([1, 1])
    t = make_trajectory(actions=[1, 0], observations=[[0], [1]], returns=[3, 5])
    alg.optimize(t)
    assert policy.received_loss_signal == -np.mean([np.log(policy.estimate_for(t.observations[0])[1]) * 1,
                                                    np.log(policy.estimate_for(t.observations[1])[0]) * -1])
