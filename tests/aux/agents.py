from reinforcement.agents.basis import AgentInterface, BatchAgent


class RandomAgent(AgentInterface):
    def __init__(self, action_space):
        self._action_space = action_space

    def next_action(self, _):
        return self._action_space.sample()

    def signal(self, _):
        pass

    def train(self):
        pass


class BatchAgentPrefab:
    def __init__(self, alg_prefab):
        self.alg_prefab = alg_prefab

    def make(self):
        return BatchAgent(self.alg_prefab.make())

    def setup_for(self, env):
        self.alg_prefab.setup_for(env)
