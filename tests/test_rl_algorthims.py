import numpy as np
import pytest
import tensorflow as tf
from pytest import approx

import reinforcement.np_operations as np_ops
import reinforcement.tf_operations as tf_ops
from reinforcement.agents.basis import BatchAgent, AgentInterface
from reinforcement.algorithm.reinforce import Reinforce


class Space:
    pass


class Discrete(Space):
    def __init__(self, n):
        self.n = n
        self.shape = (1,)

    def sample(self):
        return np.random.randint(self.n)

    def __repr__(self):
        return f"Discrete(n={self.n})"


class Box(Space):
    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return f"Box(shape={self.shape})"


class BinaryBox(Box):
    def __init__(self):
        super().__init__(shape=(2,))

    @staticmethod
    def sample():
        r = np.random.randint(1)
        return np.array([r, 1 - r])


class SimpleSwitchingMDP:
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = BinaryBox()
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
            self._flip_state()
            return self._current_state, np.random.normal(1, 0.1), self._is_done(), None
        else:
            return self._current_state, np.random.normal(-1, 0.1), self._is_done(), None

    def _is_valid(self, action):
        return action is not None and self._current_state[0] != action

    def _flip_state(self):
        self._current_state = np.array([self._current_state[1], self._current_state[0]])

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


class DeterministicAgent(AgentInterface):
    def next_action(self, observation):
        return 1 - observation[0]

    def signal(self, reward):
        pass

    def train(self):
        pass


class NpSwitchingPolicy:
    def __init__(self, plotter, lr=1):
        self._lr = lr
        self._m = np.random.normal(1, 0.001)  # high m avoids local maxima - use negative m for demonstration
        self._probs = []
        self._get_loss = None

        self._plotter = plotter(self._m)
        self._m_history = [self._m]
        self._ret_med_history = [0.0]
        self._loss_history = [0]

    def set_signal_calc(self, signal_calc):
        def np_signal(acts, ps, rs):
            return signal_calc(np_ops, acts, ps, rs)

        self._get_loss = np_signal

    def estimate(self, observation):
        y = (np.tanh(self._m * ((observation[0] * 2) - 1)) + 1) / 2
        p = np.array([y, 1 - y], dtype=np.float)
        self._probs.append(p)
        return p

    def fit(self, trajectory):
        loss = self._get_loss(trajectory.actions, np.array(self._probs, dtype=np.float), trajectory.returns)
        self._m -= loss * self._lr
        self._probs.clear()
        self._plotter.record(self._m, trajectory.returns, loss)

    def plot(self):
        self._plotter.show()

    class MatPlotter:
        def __init__(self, m):
            self._m_history = [m]
            self._ret_med_history = [0.0]
            self._loss_history = [0]

        def record(self, m, returns, loss):
            self._m_history.append(m)
            self._loss_history.append(loss)
            self._ret_med_history.append(float(np.mean(returns)))

        def show(self):
            import matplotlib.pyplot as plt
            fig, (mp, lp, rp) = plt.subplots(3, 1)
            fig.suptitle("Numpy Agent")
            mp.plot(self._m_history)
            mp.set_title("m")
            lp.plot(self._loss_history)
            lp.set_title("loss")
            rp.plot(self._ret_med_history)
            rp.set_title("median return")
            plt.show()

    class NullPlotter:
        def __init__(self, *args, **kwargs):
            pass

        def record(self, *args, **kwargs):
            pass

        def show(self):
            pass


class TfSwitchingPolicy:
    def __init__(self, session, obs_dims, num_actions, make_sum_writer, lr=10):
        self._session = session
        self._make_sum_writer = make_sum_writer
        self._lr = lr
        self._in_actions = tf.compat.v1.placeholder(shape=(None,), dtype=tf.uint8, name="actions")
        self._in_returns = tf.compat.v1.placeholder(shape=(None,), dtype=tf.float32, name="returns")
        self._in_observations = tf.compat.v1.placeholder(shape=(None, obs_dims), dtype=tf.float32, name="observations")
        theta = tf.compat.v1.get_variable("theta", shape=(obs_dims, num_actions), dtype=tf.float32,
                                          initializer=tf.glorot_uniform_initializer())
        self._out_probabilities = tf.nn.softmax(tf.matmul(self._in_observations, theta))
        self._train = None

        self._logs = [tf.summary.scalar("mean normalized return", tf.reduce_mean(self._in_returns)),
                      self._log_2d_tensor_as_img("theta", theta)]
        self._log_summary = tf.no_op
        self._sum_writer = None
        self._cur_iteration = 0

    @staticmethod
    def _log_2d_tensor_as_img(name, mat):
        return tf.summary.image(name, tf.reshape(mat, shape=(1, mat.shape[0].value, mat.shape[1].value, 1)))

    def set_signal_calc(self, signal_calc):
        loss = signal_calc(tf_ops, self._in_actions, self._out_probabilities, self._in_returns)
        self._train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self._lr).minimize(loss)
        self._session.run(tf.compat.v1.global_variables_initializer())
        self._finish_logs(loss)

    def _finish_logs(self, loss):
        self._logs.append(tf.summary.scalar("loss", loss))
        self._log_summary = tf.summary.merge(self._logs)
        self._sum_writer = self._make_sum_writer(self._session.graph)

    def estimate(self, observation):
        return np.squeeze(
            self._session.run(self._out_probabilities, {self._in_observations: np.array(observation).reshape(1, -1)}))

    def fit(self, trajectory):
        _, log = self._session.run([self._train, self._log_summary],
                                   {self._in_observations: trajectory.observations,
                                    self._in_actions: trajectory.actions,
                                    self._in_returns: trajectory.returns})
        self._sum_writer.add_summary(log, self._cur_iteration)
        self._cur_iteration += 1


class FixedBaseline:
    def __init__(self, value):
        self.value = value

    def estimate(self, trj):
        return np.ones_like(trj.returns) * self.value


@pytest.fixture
def train_length():
    return 100


@pytest.fixture
def eval_length():
    return 100


@pytest.fixture
def simple_env():
    return SimpleSwitchingMDP()


@pytest.fixture
def random_agent(simple_env):
    return RandomAgent(simple_env.action_space)


@pytest.fixture
def dt_agent():
    return DeterministicAgent()


@pytest.fixture
def session():
    with tf.compat.v1.Session() as s:
        yield s


@pytest.fixture
def make_sum_writer(session, log_tensorboard, request):
    def writer(data):
        if log_tensorboard:
            import shutil
            ld = f"{request.node.name}_log"
            shutil.rmtree(ld, ignore_errors=True)
            return tf.summary.FileWriter(ld, data, session=session)
        else:
            class _NoLog:
                def add_summary(self, *args, **kwargs):
                    pass

            return _NoLog()

    return writer


@pytest.fixture
def baseline():
    return FixedBaseline(0.0)


@pytest.fixture
def np_policy(simple_env, plot_policy):
    p = NpSwitchingPolicy(NpSwitchingPolicy.MatPlotter if plot_policy else NpSwitchingPolicy.NullPlotter)
    yield p
    p.plot()


@pytest.fixture
def np_reinforce(np_policy, baseline):
    return Reinforce(np_policy, 0.0, baseline)


@pytest.fixture
def np_reinforce_agent(np_reinforce):
    return BatchAgent(np_reinforce)


@pytest.fixture
def tf_policy(session, simple_env, make_sum_writer):
    return TfSwitchingPolicy(session, simple_env.observation_space.shape[0], simple_env.action_space.n, make_sum_writer)


@pytest.fixture
def tf_reinforce(tf_policy, baseline, session):
    return Reinforce(tf_policy, 0.99, baseline)


@pytest.fixture
def tf_reinforce_agent(tf_reinforce):
    return BatchAgent(tf_reinforce)


def test_random_agent_only_achieves_random_expected_total_return(simple_env, random_agent):
    avg_return = run_sessions_with(simple_env, random_agent, 1000)
    assert avg_return == approx(0.0, abs=2)


def run_sessions_with(env, agent, num_sessions):
    avg_total_return = 0
    for e in range(num_sessions):
        obs = env.reset()
        done = False
        total_r = 0
        while not done:
            obs, r, done, _ = env.step(agent.next_action(obs))
            agent.signal(r)
            total_r += r
        agent.train()
        avg_total_return += total_r / num_sessions
    return avg_total_return


def test_deterministic_agent_achieves_max_return(simple_env, dt_agent, eval_length):
    avg_return = run_sessions_with(simple_env, dt_agent, eval_length)
    assert avg_return == approx(simple_env.max_total_return, rel=1)


def test_reinforce_agent_achieves_near_optimal_solution(simple_env, np_reinforce_agent, train_length, eval_length):
    run_sessions_with(simple_env, np_reinforce_agent, train_length)
    avg_return = run_sessions_with(simple_env, np_reinforce_agent, eval_length)
    assert avg_return == approx(simple_env.max_total_return, abs=5)


def test_tf_reinforce_agent_learns_near_optimal_solution(simple_env, tf_reinforce_agent, train_length,
                                                         eval_length):
    run_sessions_with(simple_env, tf_reinforce_agent, train_length)
    avg_return = run_sessions_with(simple_env, tf_reinforce_agent, eval_length)
    assert avg_return > simple_env.max_total_return * 0.9
