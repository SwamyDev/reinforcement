import numpy as np


class Space:
    pass


class Discrete(Space):
    def __init__(self, n):
        self.n = n

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


class NormalDistribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.normal(self.mean, self.std)


class SwitchingMDP:
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = BinaryBox()
        self.len_episode = 100
        self.adjust_signals(reward=1, penalty=-1, std=0.1)
        self._current_state = None
        self._episode = None

    # noinspection PyAttributeOutsideInit
    def adjust_signals(self, reward, penalty, std):
        self._reward = NormalDistribution(reward, std)
        self._penalty = NormalDistribution(penalty, std)
        self.avg_max_reward = self.len_episode * self._reward.mean

    def reset(self):
        self._episode = 0
        self._current_state = self.observation_space.sample()
        return self._current_state

    def step(self, action):
        self._episode += 1
        if self._is_valid(action):
            self._flip_state()
            return self._current_state, self._reward.sample(), self._is_done(), None
        else:
            return self._current_state, self._penalty.sample(), self._is_done(), None

    def _is_valid(self, action):
        return action is not None and self._current_state[0] != action

    def _flip_state(self):
        self._current_state = np.array([self._current_state[1], self._current_state[0]])

    def _is_done(self):
        return self._episode == self.len_episode

    def __repr__(self):
        return f"SimpleDeterministic(action_space={repr(self.action_space)}, " \
               f"observation_space={repr(self.observation_space)}, len_episode={self.len_episode}," \
               f"avg_max_reward={repr(self.avg_max_reward)})"


class OneDimWalkMDP:
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Box(shape=(7, 1))
        self._max_len = 7 * 2
        self._reward = 1
        self._penalty = -1
        self.avg_max_reward = self._reward - (self.observation_space.shape[0] // 2) + 1
        self.avg_min_reward = self._penalty * self._max_len
        self._current_position = None
        self._episode = None

    def reset(self):
        self._episode = 0
        self._current_position = self.observation_space.shape[0] // 2
        return self._make_state()

    def _make_state(self):
        return np.eye(1, self.observation_space.shape[0], k=self._current_position)[0]

    def step(self, action):
        self._episode += 1
        if self._is_move_left(action):
            self._current_position -= 1
        else:
            self._current_position += 1

        if self._current_position < 0:
            self._current_position = 0
        r = self._reward if self._is_right_most() else self._penalty
        return self._make_state(), r, self._is_done(), None

    @staticmethod
    def _is_move_left(action):
        return action == 0

    def _is_right_most(self):
        return self._current_position == self.observation_space.shape[0] - 1

    def _is_done(self):
        return self._is_right_most() or self._episode == self._max_len

    def __repr__(self):
        return f"SimpleDeterministic(action_space={repr(self.action_space)}, " \
               f"observation_space={repr(self.observation_space)}," \
               f"avg_max_reward={repr(self.avg_max_reward)})"
