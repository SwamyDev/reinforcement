import numpy as np
import pytest
import tensorflow as tf

import reinforcement.tf_operations as tf_ops
from reinforcement.agents.basis import BatchAgent, AgentInterface
from reinforcement.algorithm.reinforce import Reinforce

NUM_SESSIONS = 100


class DiscreteSpace:
    def __init__(self, n):
        self.n = n
        self.shape = (1,)

    def sample(self):
        return np.random.randint(self.n)

    def __repr__(self):
        return f"DescreteSpace(n={self.n})"


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
        return np.array([self._current_state])

    def step(self, action):
        self._episode += 1
        if self._is_valid(action):
            self._move_state()
            return np.array([self._current_state]), 1, self._is_done(), None
        else:
            return np.array([self._current_state]), 0, self._is_done(), None

    def _is_valid(self, action):
        return action is not None and self._current_state != action

    def _move_state(self):
        self._current_state = (self._current_state + 1) % self.observation_space.n

    def _is_done(self):
        return self._episode == self.len_episode

    def __repr__(self):
        return f"SimpleDeterministic(action_space={repr(self.action_space)}, " \
               f"observation_space={repr(self.observation_space)}, len_episode={self.len_episode}," \
               f"max_total_return={repr(self.max_total_return)})"


class RandomAgent(AgentInterface):
    def __init__(self, action_space):
        self._action_space = action_space

    def next_action(self, _):
        return self._action_space.sample()

    def signal(self, _):
        pass

    def train(self):
        pass


class SimplePolicy:
    def __init__(self, session, obs_dims, num_actions, lr=0.1):
        self._session = session
        self._lr = lr
        self._in_actions = tf.compat.v1.placeholder(shape=(None), dtype=tf.uint8, name="actions")
        self._in_returns = tf.compat.v1.placeholder(shape=(None), dtype=tf.float32, name="returns")
        self._in_observations = tf.compat.v1.placeholder(shape=(None, obs_dims), dtype=tf.float32, name="observations")
        theta = tf.compat.v1.get_variable("theta", shape=(obs_dims, num_actions), dtype=tf.float32,
                                          initializer=tf.glorot_uniform_initializer())
        self._out_probabilities = tf.nn.softmax(tf.matmul(self._in_observations, theta))
        self._train = None

    def set_signal_calc(self, signal_calc):
        loss = signal_calc(tf_ops, self._in_actions, self._out_probabilities, self._in_returns)
        self._train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self._lr).minimize(loss)
        self._session.run(tf.compat.v1.global_variables_initializer())

    def estimate(self, observation):
        return np.squeeze(
            self._session.run(self._out_probabilities, {self._in_observations: np.array(observation).reshape(1, -1)}))

    def fit(self, trajectory):
        self._session.run(self._train, {self._in_observations: trajectory.observations,
                                        self._in_actions: trajectory.actions,
                                        self._in_returns: trajectory.returns})


class FixedBaseline:
    def __init__(self, value):
        self.value = value

    def estimate(self, _):
        return self.value


@pytest.fixture
def env():
    return SimpleDeterministic()


@pytest.fixture
def random_agent(env):
    return RandomAgent(env.action_space)


@pytest.fixture
def session():
    tf.compat.v1.enable_eager_execution()
    with tf.compat.v1.Session() as s:
        yield s


@pytest.fixture
def policy(session, env):
    return SimplePolicy(session, env.observation_space.shape[0], env.action_space.n)


@pytest.fixture
def baseline():
    return FixedBaseline(0)


@pytest.fixture
def reinforce(policy, baseline):
    return Reinforce(policy, 0.99, baseline)


@pytest.fixture
def reinforce_agent(reinforce):
    return BatchAgent(reinforce)


def test_random_agent_only_achieves_random_expected_total_return(env, random_agent):
    avg_return = run_sessions_with(env, random_agent)
    assert avg_return == pytest.approx(env.max_total_return / env.action_space.n, rel=1)


def run_sessions_with(env, agent):
    avg_return = 0
    for _ in range(NUM_SESSIONS):
        obs = env.reset()
        done = False
        total_r = 0
        while not done:
            obs, r, done, _ = env.step(agent.next_action(obs))
            agent.signal(r)
            total_r += r
        agent.train()
        avg_return += total_r / NUM_SESSIONS
    return avg_return


@pytest.mark.skip("WIP: still not performing on a consistent level")
def test_reinforce_agent_achieves_higher_return_than_random(env, reinforce_agent):
    avg_return = run_sessions_with(env, reinforce_agent)
    print(avg_return)
    assert avg_return > (env.max_total_return / env.action_space.n) * 1.1
