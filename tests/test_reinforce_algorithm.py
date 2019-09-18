import numpy as np
import pytest

import reinforcement.np_operations as np_ops
from reinforcement.algorithm.reinforce import Reinforce
from reinforcement.np_operations import softmax


class PolicySpy:
    def __init__(self):
        self.fitted_onto = None
        self._calc = None

    def set_signal_calc(self, calc):
        self._calc = calc

    def fit(self, trajectory):
        self.fitted_onto = trajectory.advantages


class PolicySim(PolicySpy):
    def __init__(self):
        super().__init__()
        self._estimations = {}
        self.num_actions = 5
        self.received_signal = None

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
        self.received_signal = self._calc(np_ops, trajectory.actions, probabilities, trajectory.advantages)

    def __repr__(self):
        return f"ApproximationSim() - {len(self._estimations)} estimations done"


class BaselineStub:
    def __init__(self, size):
        self._estimates = np.random.uniform(size)

    def set_estimates(self, estimates):
        self._estimates = np.array(estimates, dtype=np.float32)

    def estimate(self, _):
        return self._estimates

    def fit(self, _):
        pass


class BaselineSpy(BaselineStub):
    def __init__(self, size):
        super().__init__(size)
        self.estimated_for = None
        self.fitted_onto = None

    def estimate(self, trj):
        self.estimated_for = trj.returns
        return super().estimate(trj)

    def fit(self, trj):
        super().fit(trj)
        self.fitted_onto = trj.returns


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


def make_alg(policy, baseline=None, gamma=0.9, num_train_trjaectories=1):
    return Reinforce(policy, gamma, baseline, num_train_trjaectories)


def assert_estimates(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_that_reinforce_adds_correct_advantages_to_trajectory(policy, baseline, make_trajectory):
    alg = make_alg(policy, baseline, gamma=0.5)
    baseline.set_estimates([2.0, 3.0, 4.0])
    alg.optimize(make_trajectory(returns=[1.0, 2.0, 3.0]))
    assert_signals(policy.fitted_onto, normalized([0.75, 0.5, -1.0]))


def normalized(values):
    a = np.array(values, dtype=np.float32)
    return (a - np.mean(a)) / (np.std(a) + 1e-8)


def assert_signals(actual, expected):
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


def test_that_reinforce_passes_accumulated_returns_to_baseline(policy, baseline, make_trajectory):
    alg = make_alg(policy, baseline, gamma=0.5)
    alg.optimize(make_trajectory(returns=[1.0, 2.0, 3.0]))
    assert_estimates(baseline.estimated_for, [2.75, 3.5, 3.0])


def test_reinforce_does_not_fit_policy_and_baseline_until_trajectory_batch_is_filled(policy, baseline, make_trajectory):
    alg = make_alg(policy, baseline, num_train_trjaectories=2)
    alg.optimize(make_trajectory(returns=[1.0, 2.0, 3.0]))
    assert not baseline.fitted_onto and not policy.fitted_onto


def test_reinforce_fits_baseline_on_concatenated_trajectory_batch_returns(policy, baseline, make_trajectory):
    alg = make_alg(policy, baseline, gamma=0.5, num_train_trjaectories=2)
    alg.optimize(make_trajectory(returns=[1.0, 2.0, 3.0]))
    alg.optimize(make_trajectory(returns=[4.0, 3.0, 2.0]))
    assert_signals(baseline.fitted_onto, concat([2.75, 3.5, 3.0], [6.0, 4.0, 2.0]))


def concat(*args):
    return np.concatenate(args)


def test_reinforce_fits_policy_on_concatenated_trajectory_batch_advantages(policy, baseline, make_trajectory):
    alg = make_alg(policy, baseline, gamma=0.5, num_train_trjaectories=2)
    baseline.set_estimates([2.0, 3.0, 4.0])
    alg.optimize(make_trajectory(returns=[1.0, 2.0, 3.0]))
    alg.optimize(make_trajectory(returns=[4.0, 3.0, 2.0]))
    assert_signals(policy.fitted_onto, concat(normalized([0.75, 0.5, -1.0]), normalized([4.0, 1.0, -2.0])))


def test_reinforce_uses_clean_trajectory_batches(policy, baseline, make_trajectory):
    baseline.set_estimates([0, 0, 0])
    alg = make_alg(policy, baseline, gamma=0.5, num_train_trjaectories=2)
    fill_trajectory_batch(alg, make_trajectory)
    alg.optimize(make_trajectory(returns=[1.0, 2.0, 3.0]))
    alg.optimize(make_trajectory(returns=[4.0, 3.0, 2.0]))
    assert_signals(policy.fitted_onto, concat(normalized([2.75, 3.5, 3.0]), normalized([6.0, 4.0, 2.0])))


def fill_trajectory_batch(alg, make_trajectory):
    # noinspection PyProtectedMember
    for _ in range(alg._num_train_trajectories):
        alg.optimize(make_trajectory())


def test_fitting_the_approximation_calculates_the_correct_signal(policy, baseline, make_trajectory):
    policy.num_actions = 2
    alg = make_alg(policy, baseline, gamma=0.5)
    baseline.set_estimates([-2.0, 2.0])
    t = make_trajectory(actions=[1, 0], observations=[[0], [1]], returns=[2.0, 4.0])
    alg.optimize(t)
    assert policy.received_signal == np.mean([np.log(policy.estimate_for(t.observations[0])[1]) * 1.0,
                                              np.log(policy.estimate_for(t.observations[1])[0]) * -1.0])
