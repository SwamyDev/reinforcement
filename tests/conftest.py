import numpy as np
import pytest


@pytest.fixture
def observation_factory():
    return lambda: np.random.uniform(size=(2, 3))


@pytest.fixture
def observation(observation_factory):
    return observation_factory()
