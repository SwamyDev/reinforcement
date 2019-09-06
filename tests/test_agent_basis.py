from abc import ABC, abstractmethod

import numpy as np
import pytest
from pytest import approx


class AgentInterface(ABC):
    @abstractmethod
    def next_action(self, observation):
        pass

    @abstractmethod
    def observe(self, observation):
        pass


class BatchAgent(AgentInterface):
    def __init__(self, algorithm):
        self._algorithm = algorithm

    def observe(self, observation):
        pass

    def next_action(self, observation):
        p = self._algorithm.sample(observation)
        return np.eye(1, len(p), k=np.random.choice(len(p), p=p))[0]


# noinspection PyAbstractClass
class MissingNextAction(AgentInterface):
    def observe(self, reward):
        pass


# noinspection PyAbstractClass
class MissingObserve(AgentInterface):
    def next_action(self, observation):
        pass


class AlgorithmStub:
    def __init__(self):
        self.returns_sample_probability = [0.5, 0.5]

    def sample(self, observation):
        return self.returns_sample_probability


@pytest.fixture
def algorithm():
    return AlgorithmStub()


@pytest.fixture
def batch_agent(algorithm):
    return BatchAgent(algorithm)


@pytest.mark.parametrize('invalid_agent_type', [MissingNextAction, MissingObserve])
def test_missing_agent_interface_throws(invalid_agent_type):
    with pytest.raises(TypeError):
        invalid_agent_type()


def test_batch_agent_returns_action_as_one_hot_vector(batch_agent, observation):
    action = batch_agent.next_action(observation)
    assert (action == np.array([1, 0])).all() or (action == np.array([0, 1])).all()


def test_batch_agents_samples_its_algorithm_for_next_action(batch_agent, algorithm, observation):
    algorithm.returns_sample_probability = [0.8, 0.2]
    assert on_average(lambda: batch_agent.next_action(observation)) == approx([0.8, 0.2], abs=0.05)


def on_average(producer):
    return np.mean([producer() for _ in range(1000)], axis=0)


def test_a_batch_agent_starts_out_with_zero_trajectories(batch_agent, algorithm, observation):
    batch_agent.next_action(batch_agent)
