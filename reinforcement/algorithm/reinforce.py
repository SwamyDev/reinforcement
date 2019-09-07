import numpy as np



class Reinforce:
    def __init__(self, policy, gamma, baseline):
        self._policy = policy
        self._gamma = gamma
        self._baseline = baseline
        self._policy.set_signal_calc(self._calc_signal)

    @staticmethod
    def _calc_signal(ops, in_actions, out_probs, in_returns):
        one_hot = ops.one_hot(in_actions, out_probs.shape[1])
        selected_prob = ops.reduce_sum(one_hot * out_probs, axis=1)
        return -ops.reduce_mean(ops.log(selected_prob) * in_returns)

    def sample(self, observation):
        return self._policy.estimate(observation)

    def optimize(self, trajectory):
        self._update_rewards_to_return_signals(trajectory)
        self._policy.fit(trajectory)

    def _update_rewards_to_return_signals(self, trajectory):
        bases = self._baseline.estimate(trajectory)
        trj_len = len(trajectory)
        for tp in range(trj_len, 0, -1):
            prev = trajectory.returns[tp] if tp < trj_len else 0
            trajectory.returns[tp - 1] += self._gamma * prev
        trajectory.returns -= bases
        self._normalize(trajectory.returns)

    @staticmethod
    def _normalize(rs):
        std = np.std(rs)
        rs -= np.mean(rs)
        rs /= std + 1e-8
