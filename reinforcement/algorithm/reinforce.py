import numpy as np


def _calc_signal(ops, action_dist, returns):
    return ops.reduce_mean(ops.log(action_dist) * returns)


class Reinforce:
    def __init__(self, policy, gamma, baseline):
        self._policy = policy
        self._gamma = gamma
        self._baseline = baseline
        self._policy.set_signal_calc(_calc_signal)

    def sample(self, observation):
        return self._policy.estimate(observation)

    def optimize(self, trajectories):
        self._update_rewards_to_return_signals(trajectories)
        self._policy.fit(trajectories)

    def _update_rewards_to_return_signals(self, trajectories):
        bases = self._baseline.estimate(trajectories)
        for rewards, bs in zip(trajectories.returns, bases):
            trj_len = len(rewards)
            for tp in range(trj_len, 0, -1):
                prev = rewards[tp] if tp < trj_len else 0
                rewards[tp - 1] += self._gamma * prev
            rewards -= bs
            self._normalize(rewards)

    @staticmethod
    def _normalize(rs):
        std = np.std(rs)
        rs -= np.mean(rs)
        rs /= std + 1e-8
