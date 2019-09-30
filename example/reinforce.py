import argparse
import logging

import gym
import numpy as np

try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
except ImportError:
    raise ImportError("reinforcement requires tensorflow 1.14")

from example.baselines import ValueBaseline
from example.log_utils import NoLog
from example.policies import ParameterizedPolicy
from reinforcement.algorithm.reinforce import Reinforce
from reinforcement.agents.basis import BatchAgent

logger = logging.getLogger(__name__)


class Reporter:
    def __init__(self, cfg):
        self.cfg = cfg

    def should_render(self, e):
        return _matches_frequency(e, self.cfg.render_frq)

    def should_log(self, e):
        return _matches_frequency(e, self.cfg.log_frq)

    def report(self, e, rs):
        r, lf = rs[-1], self.cfg.log_frq
        return f"Episode {e}: reward={r}; mean reward of last {lf} episodes: {np.mean(rs[-lf:])}"


def _matches_frequency(e, f):
    return f > 0 and e % f == 0


def run_reinforce(config):
    reporter, env, rewards = Reporter(config), gym.make('CartPole-v0'), []
    with tf1.Session() as session:
        agent = _make_agent(config, session, env)
        for episode in range(1, config.episodes + 1):
            reward = _run_episode(env, episode, agent, reporter)
            rewards.append(reward)
            if reporter.should_log(episode):
                logger.info(reporter.report(episode, rewards))
    env.close()


def _make_agent(config, session, env):
    p = ParameterizedPolicy(session, env.observation_space.shape[0], env.action_space.n, NoLog(), config.lr_policy)
    b = ValueBaseline(session, env.observation_space.shape[0], NoLog(), config.lr_baseline)
    alg = Reinforce(p, config.gamma, b, config.num_trajectories)
    return BatchAgent(alg)


def _run_episode(env, episode, agent, report):
    obs = env.reset()
    done, reward = False, 0
    while not done:
        if report.should_render(episode):
            env.render()
        obs, r, done, _ = env.step(agent.next_action(obs))
        agent.signal(r)
        reward += r

    agent.train()
    return reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run reinforce example.')
    parser.add_argument('-e', '--episodes', type=int, default=3000, help='number of episodes to be run')
    parser.add_argument('-n', '--num-trajectories', type=int, default=10,
                        help='number of trajectories used in training of agent')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='gamma used for reward accumulation')
    parser.add_argument('--lr-policy', type=float, default=50, help='learning rate of policy ANN')
    parser.add_argument('--lr-baseline', type=float, default=0.01, help='learning rate of baseline ANN')
    parser.add_argument('--render-frq', type=int, default=0, help='render every x episode')
    parser.add_argument('--log-frq', type=int, default=100, help='log every x episode')
    parser.add_argument('--log-lvl', type=str, default="info", help='log level (default: info)')

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_lvl.upper()))
    run_reinforce(args)
