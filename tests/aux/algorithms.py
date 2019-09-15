from reinforcement.algorithm.reinforce import Reinforce


class ReinforcePrefab:
    def __init__(self, policy_prefab, baseline_prefab, gamma, num_train_trajectories):
        self.policy_prefab = policy_prefab
        self.baseline_prefab = baseline_prefab
        self.gamma = gamma
        self.num_train_trajectories = num_train_trajectories

    def make(self):
        return Reinforce(self.policy_prefab.make(), self.gamma, self.baseline_prefab.make(),
                         self.num_train_trajectories)

    def setup_for(self, env):
        self.policy_prefab.setup_for(env)
        self.baseline_prefab.setup_for(env)
