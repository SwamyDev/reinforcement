import numpy as np
import pytest


@pytest.fixture
def observation():
    return np.random.uniform(size=(2, 3))
