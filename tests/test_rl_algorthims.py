import numpy as np
import pytest

NUM_SESSIONS = 1000


class DiscreteSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(self.n)


class SimpleDeterministic:
    def __init__(self):
        self.action_space = DiscreteSpace(2)
        self.observation_space = DiscreteSpace(2)
        self.len_episode = 100
        self.max_total_return = 100
        self._current_state = None
        self._episode = None

    def reset(self):
        self._episode = 0
        self._current_state = self.observation_space.sample()
        return self._current_state

    def step(self, action):
        self._episode += 1
        if self._is_valid(action):
            self._move_state()
            return self._current_state, 1, self._is_done(), None
        else:
            return self._current_state, 0, self._is_done(), None

    def _is_valid(self, action):
        return self._current_state != action

    def _move_state(self):
        self._current_state = (self._current_state + 1) % self.observation_space.n

    def _is_done(self):
        return self._episode == self.len_episode


class RandomAgent:
    def __init__(self, action_space):
        self._action_space = action_space

    def next_action(self, _):
        return self._action_space.sample()


class ReinforceAgent:
    def __init__(self):
        pass

    def next_action(self, observation):
        pass


@pytest.fixture
def env():
    return SimpleDeterministic()


@pytest.fixture
def random_agent(env):
    return RandomAgent(env.action_space)


@pytest.fixture
def reinforce_agent(env):
    return ReinforceAgent()


def test_random_agent_only_achieves_random_expected_total_return(env, random_agent):
    avg_return = run_sessions_with(env, random_agent)
    assert avg_return == pytest.approx(env.max_total_return / env.action_space.n, rel=1)


def run_sessions_with(env, random_agent):
    avg_return = 0
    for _ in range(NUM_SESSIONS):
        obs = env.reset()
        done = False
        total_r = 0
        while not done:
            obs, r, done, _ = env.step(random_agent.next_action(obs))
            total_r += r
        avg_return += total_r / NUM_SESSIONS
    return avg_return


@pytest.mark.skip
def test_reinforce_agent_achieves_higher_return_than_random(env, reinforce_agent):
    avg_return = run_sessions_with(env, reinforce_agent)
    assert avg_return > (env.max_total_return / env.action_space.n) * 1.1
