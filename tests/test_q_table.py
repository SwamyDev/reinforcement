import pytest

from reinforcement.reward_functions.q_function import InvalidAction
from reinforcement.reward_functions.q_table import QTable


@pytest.fixture
def table():
    """Returns a default QTable with actions 1, 2, 4 and no initializer"""
    return QTable([1, 2, 4])


def test_default_initial_value(table):
    assert table[[1, 2, 3], 1] == 0


def test_using_initializer():
    table = QTable(range(0, 2), lambda: 1)
    assert table[[1, 0], 0] == 1


def test_multiple_action_values(table):
    table[[2, 3], 2] = 7
    table[[2, 3], 1] = 10
    assert table[[2, 3], 1] == 10
    assert table[[2, 3], 2] == 7


def test_update_actions_value(table):
    table[[2, 3], 2] = 7
    table[[2, 3], 1] = 10

    table[[2, 3], 1] = 0

    assert table[[2, 3], 1] == 0


def test_scalar_state(table):
    table[3, 1] = 6
    assert table[3, 1] == 6


def test_setting_action_out_of_space_throws_exception(table):
    with pytest.raises(InvalidAction):
        unused = table[[2, 3], 0]


def test_state_is_unaltered(table):
    s = [1, 2]
    table[s, 1] = 10.0
    assert s == [1, 2]
    unused = table[s, 1]
    assert s == [1, 2]
    unused = table[s, 2]
    assert s == [1, 2]
