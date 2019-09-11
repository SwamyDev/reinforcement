import numpy as np
import pytest
from pytest import approx
from reinforcement.np_operations import one_hot

from reinforcement.agents.basis import AgentInterface, BatchAgent
from reinforcement.trajectories import TrajectoryError


# noinspection PyAbstractClass
class MissingNextAction(AgentInterface):
    def signal(self, reward):
        pass

    def train(self):
        pass


# noinspection PyAbstractClass
class MissingSignal(AgentInterface):
    def next_action(self, observation):
        pass

    def train(self):
        pass


# noinspection PyAbstractClass
class MissingTrain(AgentInterface):
    def signal(self, reward):
        pass

    def next_action(self, observation):
        pass


class AlgorithmStub:
    def __init__(self):
        self.actions_probability = [0.2, 0.8]

    def sample(self, observation):
        return self.actions_probability

    def optimize(self, trajectory):
        pass


class AlgorithmSpy(AlgorithmStub):
    def __init__(self):
        super().__init__()
        self.received_trajectory = None

    def optimize(self, trajectory):
        self.received_trajectory = trajectory


@pytest.fixture
def algorithm():
    return AlgorithmSpy()


@pytest.fixture
def batch_agent(algorithm):
    return BatchAgent(algorithm)


@pytest.fixture
def episode(batch_agent, algorithm, observation_factory, make_trajectory_builder):
    class _Episode:
        @staticmethod
        def run():
            b = make_trajectory_builder()
            for r in range(-1, 2):
                o = observation_factory()
                a = batch_agent.next_action(o)
                batch_agent.signal(r)
                b.add(a, o, r)
            return b.to_trajectory()

    return _Episode()


@pytest.mark.parametrize('invalid_agent_type', [MissingNextAction, MissingSignal, MissingTrain])
def test_missing_agent_interface_throws(invalid_agent_type):
    with pytest.raises(TypeError):
        invalid_agent_type()


def test_batch_agent_returns_action(batch_agent, observation):
    action = batch_agent.next_action(observation)
    assert action == 0 or action == 1


def test_batch_agents_samples_its_algorithm_for_next_action(batch_agent, algorithm, observation):
    def do_transition():
        a = batch_agent.next_action(observation)
        batch_agent.signal(0)
        return a

    assert on_average(do_transition) == approx(algorithm.actions_probability, abs=0.05)


def on_average(producer):
    return np.mean([one_hot([producer()], 2)[0] for _ in range(1000)], axis=0)


def test_a_batch_agent_starts_out_with_an_empty_trajectory(batch_agent):
    with pytest.raises(TrajectoryError):
        batch_agent.train()


def test_batch_agent_records_multiple_transitions(algorithm, batch_agent, episode):
    trajectory = episode.run()
    batch_agent.train()
    assert algorithm.received_trajectory == trajectory


def test_batch_agent_trains_on_trajectories_from_different_episodes(batch_agent, algorithm, episode):
    episode.run()
    batch_agent.train()
    trajectory = episode.run()
    batch_agent.train()
    assert algorithm.received_trajectory == trajectory
