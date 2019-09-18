import numpy as np

from reinforcement.trajectories import concatenate


class Reinforce:
    def __init__(self, policy, gamma, baseline, num_train_trajectories=100):
        self._policy = policy
        self._gamma = gamma
        self._baseline = baseline
        self._policy.set_signal_calc(self._calc_signal)
        self._trajectories = list()
        self._num_train_trajectories = num_train_trajectories

    @staticmethod
    def _calc_signal(ops, in_actions, out_probs, in_returns):
        one_hot = ops.one_hot(in_actions, out_probs.shape[1])
        selected_prob = ops.reduce_sum(one_hot * out_probs, axis=1)
        return ops.reduce_mean(ops.log(selected_prob) * in_returns)

    def sample(self, observation):
        return self._policy.estimate(observation)

    def optimize(self, trajectory):
        self._update_rewards_to_return_signals(trajectory)
        self._trajectories.append(trajectory)
        if len(self._trajectories) == self._num_train_trajectories:
            total = concatenate(self._trajectories)
            self._baseline.fit(total)
            self._policy.fit(total)
            self._trajectories.clear()

    def _update_rewards_to_return_signals(self, trajectory):
        trj_len = len(trajectory)
        for tp in range(trj_len, 0, -1):
            prev = trajectory.returns[tp] if tp < trj_len else 0
            trajectory.returns[tp - 1] += self._gamma * prev
        trajectory.advantages = trajectory.returns - self._baseline.estimate(trajectory)
        self._normalize(trajectory.advantages)

    @staticmethod
    def _normalize(rs):
        std = np.std(rs)
        rs -= np.mean(rs)
        rs /= std + 1e-8
