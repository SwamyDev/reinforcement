import re
import subprocess

import pytest


class Examples:
    def __init__(self, module_path):
        self.reinforce = module_path / 'example/reinforce.py'


@pytest.fixture
def examples(request):
    return Examples(request.session.fspath)


def test_reinforce_example_converges_with_default_setting(examples):
    assert _run(examples.reinforce, episodes=500, log_frq=100) > 50


@pytest.mark.slow
@pytest.mark.flaky(reruns=1, reruns_delay=1)
def test_reinforce_example_closes_in_on_optimal_solution(examples):
    assert _run(examples.reinforce, episodes=3000, log_frq=100) > 195


def _run(example, **kwargs):
    args = [[f"--{k.replace('_', '-')}", str(v)] for k, v in kwargs.items()]
    args = [a for pair in args for a in pair]
    r = subprocess.run(['python', example] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = r.stdout.decode('utf-8').splitlines()
    return float(re.findall(r'[-+]?[0-9]*\.?[0-9]+', lines[-1])[-1])
