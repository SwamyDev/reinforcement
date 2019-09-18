from functools import reduce

import numpy as np


class Trajectory:
    ATTRIBUTES_TO_CONCATENATE = ['observations', 'actions', 'returns', 'advantages']

    def __init__(self, actions, observations, returns):
        self.observations = observations
        self.actions = actions
        self.returns = returns

    def __len__(self):
        return len(self.returns)

    def __eq__(self, other):
        return np.array_equal(self.actions, other.actions) and \
               np.array_equal(self.observations, other.observations) and \
               np.array_equal(self.returns, other.returns)

    def concatenate(self, other):
        for atr in self.ATTRIBUTES_TO_CONCATENATE:
            lhs = getattr(self, atr, None)
            rhs = getattr(other, atr, None)
            if lhs is not None and rhs is not None:
                setattr(self, atr, np.concatenate([lhs, rhs]))
        return self

    def __repr__(self):
        return f"Trajectory:\n{str(self)}"

    def __str__(self):
        return "\n".join(
            f"pi -> {a} given: \n{o}\n R: {r}" for a, o, r in zip(self.actions, self.observations, self.returns))


def history_to_trajectory(history):
    if len(history) == 0:
        raise TrajectoryError("Attempt to create trajectory from empty history.\nRecord some data first.")
    ats, obs, rws = list(zip(*history))
    return Trajectory(np.array(ats), np.array(obs, dtype=np.float32), np.array(rws, dtype=np.float32))


def concatenate(trajectories):
    def concat_two(lhs, rhs):
        lhs.concatenate(rhs)
        return lhs

    return reduce(concat_two, trajectories)


class TrajectoryRecorder:
    def __init__(self):
        self._history = list()
        self._action = None
        self._observation = None
        self._reward = None

    def start(self):
        return self._ActionRecord(self)

    def record_next(self):
        self._history.append((self._action, self._observation, self._reward))

    def to_trajectory(self):
        t = history_to_trajectory(self._history)
        self._history.clear()
        return t

    class _Record:
        def __init__(self, recorder):
            self._recorder = recorder

    class _ActionRecord(_Record):
        def add_action(self, a):
            self._recorder._action = a
            return self

        def __next__(self):
            # noinspection PyProtectedMember
            return self._recorder._ObservationRecord(self._recorder)

    class _ObservationRecord(_Record):
        def add_observation(self, o):
            self._recorder._observation = o
            return self

        def __next__(self):
            # noinspection PyProtectedMember
            return self._recorder._RewardRecord(self._recorder)

    class _RewardRecord(_Record):
        def add_reward(self, r):
            self._recorder._reward = r
            self._recorder.record_next()
            return self

        def __next__(self):
            # noinspection PyProtectedMember
            return self._recorder._ActionRecord(self._recorder)


class TrajectoryError(ValueError):
    pass
