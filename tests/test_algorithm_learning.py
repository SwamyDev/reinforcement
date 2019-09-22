from tests.aux.agents import RandomAgent, BatchAgentPrefab
from tests.aux.algorithms import ReinforcePrefab
from tests.aux.baselines import FixedBaselinePrefab, ValueBaselinePrefab
from tests.aux.environments import SwitchingMDP, OneDimWalkMDP
from tests.aux.policies import LinearPolicy, ParameterizedPolicyPrefab

try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")

import pytest
from pytest import approx

from reinforcement.agents.basis import BatchAgent, AgentInterface
from reinforcement.algorithm.reinforce import Reinforce


class OptimalAgent(AgentInterface):
    def next_action(self, observation):
        return 1 - observation[0]

    def signal(self, reward):
        pass

    def train(self):
        pass


@pytest.fixture
def train_length():
    return 100


@pytest.fixture
def eval_length():
    return 100


@pytest.fixture
def switching_env():
    return SwitchingMDP()


@pytest.fixture
def one_dim_walk_env():
    return OneDimWalkMDP()


@pytest.fixture
def random_agent(switching_env):
    return RandomAgent(switching_env.action_space)


@pytest.fixture
def optimal_agent():
    return OptimalAgent()


@pytest.fixture
def baseline_prefab():
    return FixedBaselinePrefab(0.0)


@pytest.fixture
def baseline(baseline_prefab):
    return baseline_prefab.make()


@pytest.fixture
def value_baseline_prefab(session, switching_env, summary_writer):
    return ValueBaselinePrefab(session, switching_env.observation_space.shape[0], summary_writer, lr=0.1)


@pytest.fixture
def value_baseline(value_baseline_prefab):
    return value_baseline_prefab.make()


@pytest.fixture
def linear_policy(switching_env, plot_policy):
    p = LinearPolicy(LinearPolicy.MatPlotter if plot_policy else LinearPolicy.NullPlotter)
    yield p
    p.plot()


@pytest.fixture
def reinforce_linear(linear_policy, baseline):
    return Reinforce(linear_policy, 0.0, baseline, 1)


@pytest.fixture
def policy_parameterized_prefab(session, switching_env, summary_writer):
    return ParameterizedPolicyPrefab(session, switching_env.observation_space.shape[0], switching_env.action_space.n,
                                     summary_writer, 10)


@pytest.fixture
def policy_parameterized(policy_parameterized_prefab):
    return policy_parameterized_prefab.make()


@pytest.fixture
def reinforce_parameterized(policy_parameterized, baseline):
    return Reinforce(policy_parameterized, 0.99, baseline, num_train_trajectories=1)


@pytest.fixture(params=['reinforce_linear', 'reinforce_parameterized'])
def reinforce_agents(request):
    alg = request.getfixturevalue(request.param)
    return BatchAgent(alg)


@pytest.fixture
def reinforce_prefab(policy_parameterized_prefab, baseline_prefab):
    return ReinforcePrefab(policy_parameterized_prefab, baseline_prefab, 0.99, 1)


@pytest.fixture
def reinforce_agent_prefab(reinforce_prefab):
    return BatchAgentPrefab(reinforce_prefab)


@pytest.fixture
def reinforce_parameterized_agent(reinforce_agent_prefab):
    return reinforce_agent_prefab.make()


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_random_agent_only_achieves_random_expected_return(switching_env, random_agent):
    avg_return = run_sessions_with(switching_env, random_agent, 1000)
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


def test_optimal_agent_achieves_max_return(switching_env, optimal_agent, eval_length):
    avg_return = run_sessions_with(switching_env, optimal_agent, eval_length)
    assert avg_return == approx(switching_env.avg_max_reward, rel=1)


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_reinforce_agents_learn_near_optimal_solution(switching_env, reinforce_agents, train_length,
                                                      eval_length):
    run_sessions_with(switching_env, reinforce_agents, train_length)
    avg_return = run_sessions_with(switching_env, reinforce_agents, eval_length)
    assert avg_return == approx(switching_env.avg_max_reward, abs=10)


@pytest.mark.stochastic(10, max_samples=100)
def test_reinforce_agent_has_reduced_learning_without_baseline_and_high_variance_rewards(one_dim_walk_env,
                                                                                         reinforce_agent_prefab,
                                                                                         train_length,
                                                                                         eval_length, stochastic_run):
    reinforce_agent_prefab.setup_for(one_dim_walk_env)
    reinforce_agent_prefab.alg_prefab.gamma = 1

    agent = reinforce_agent_prefab.make()
    run_sessions_with(one_dim_walk_env, agent, 600)
    stochastic_run.record(run_sessions_with(one_dim_walk_env, agent, 10))

    min_r = one_dim_walk_env.avg_min_reward
    assert stochastic_run.count(result_equals=approx(min_r, abs=1)) >= int(len(stochastic_run) * 0.7)


@pytest.mark.stochastic(10, max_samples=100)
def test_reinforce_agent_and_learning_baseline_is_robust_towards_variance(one_dim_walk_env, reinforce_agent_prefab,
                                                                          value_baseline_prefab, train_length,
                                                                          eval_length, stochastic_run):
    reinforce_agent_prefab.alg_prefab.baseline_prefab = value_baseline_prefab
    reinforce_agent_prefab.setup_for(one_dim_walk_env)
    reinforce_agent_prefab.alg_prefab.gamma = 1
    agent = reinforce_agent_prefab.make()

    run_sessions_with(one_dim_walk_env, agent, 300)
    stochastic_run.record(run_sessions_with(one_dim_walk_env, agent, 10))

    max_r = one_dim_walk_env.avg_max_reward
    assert stochastic_run.count(result_equals=approx(max_r, abs=1)) >= int(len(stochastic_run) * 0.7)
