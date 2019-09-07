from abc import ABC, abstractmethod

import numpy as np

from reinforcement.trajectories import TrajectoryBuilder


class AgentInterface(ABC):
    @abstractmethod
    def next_action(self, observation):
        pass

    @abstractmethod
    def signal(self, reward):
        pass

    @abstractmethod
    def train(self):
        pass


class BatchAgent(AgentInterface):
    def __init__(self, algorithm):
        self._algorithm = algorithm
        self._trajectory_record = TrajectoryBuilder()

    def signal(self, reward):
        self._trajectory_record = self._trajectory_record.finish_with(reward)

    def next_action(self, observation):
        p = self._algorithm.sample(observation)
        a = np.random.choice(len(p), p=p)
        self._trajectory_record = self._trajectory_record.add_action(a).given(observation)
        return a

    def train(self):
        trj = self._trajectory_record.to_trajectory()
        self._algorithm.optimize(trj)
