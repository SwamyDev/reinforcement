from abc import ABC, abstractmethod

import numpy as np

from reinforcement.trajectories import TrajectoryRecorder


class AgentInterface(ABC):
    @abstractmethod
    def next_action(self, observation):
        raise NotImplementedError

    @abstractmethod
    def signal(self, reward):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError


class BatchAgent(AgentInterface):
    def __init__(self, algorithm):
        self._algorithm = algorithm
        self._recorder = TrajectoryRecorder()
        self._record = self._recorder.start()

    def signal(self, reward):
        self._record = next(self._record.add_reward(reward))

    def next_action(self, observation):
        p = self._algorithm.sample(observation)
        a = np.random.choice(len(p), p=p)
        self._record = next(self._record.add_action(a))
        self._record = next(self._record.add_observation(observation))
        return a

    def train(self):
        trj = self._recorder.to_trajectory()
        self._algorithm.optimize(trj)
