import numpy as np


class Trajectory:
    def __init__(self, observations, actions, returns):
        self.observations = observations
        self.actions = actions
        self.returns = returns

    def __len__(self):
        return len(self.returns)

    def __eq__(self, other):
        return np.array_equal(self.observations, other.observations) and \
               np.array_equal(self.actions, other.actions) and \
               np.array_equal(self.returns, other.returns)

    def __repr__(self):
        return f"Trajectory:\n{str(self)}"

    def __str__(self):
        return "\n".join(
            f"pi -> {a} given: \n{o}\n R: {r}" for a, o, r in zip(self.actions, self.observations, self.returns))


class TrajectoryBuilder:
    def __init__(self):
        self._history = list()
        self._current_record = None

    def add(self, observation, action, reward):
        self._history.append((observation, action, reward))

    def add_action(self, action):
        return self._ActionRecord(self, action)

    class _ActionRecord:
        def __init__(self, builder, action):
            self.builder = builder
            self.action = action

        def given(self, observation):
            return self._ObservationRecord(self, observation)

        class _ObservationRecord:
            def __init__(self, action_rec, observation):
                self._action_rec = action_rec
                self._observation = observation

            def finish_with(self, reward):
                self._action_rec.builder.add(self._observation, self._action_rec.action, reward)
                return self._action_rec.builder

    def to_trajectory(self):
        if len(self._history) == 0:
            raise TrajectoryError("Attempt to create trajectory from empty history.\nRecord some data first.")
        obs, ats, rws = list(zip(*self._history))
        self._history.clear()
        return Trajectory(np.array(obs), np.array(ats), np.array(rws, dtype=np.float32))


class TrajectoryError(ValueError):
    pass


class Trajectories:
    _RETURNS = 2
    _ACTIONS = 1

    def __init__(self):
        self._trajectories = list()

    def add(self, trajectory):
        self._trajectories.append(trajectory)

    @property
    def returns(self):
        for t in self._trajectories:
            yield t.returns

    @property
    def actions(self):
        for t in self._trajectories:
            yield t.actions
