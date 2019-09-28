[![Build Status](https://travis-ci.org/SwamyDev/reinforcement.svg?branch=master)](https://travis-ci.org/SwamyDev/reinforcement) [![Coverage Status](https://coveralls.io/repos/github/SwamyDev/reinforcement/badge.svg?branch=master)](https://coveralls.io/github/SwamyDev/reinforcement?branch=master) [![PyPI version](https://badge.fury.io/py/reinforcement.svg)](https://badge.fury.io/py/reinforcement)

# Reinforcement
The reinforcement module aims to provide simple implementations for basic reinforcement learning algorithms, using Test Driven Development and other principles of Software Engineering in an attempt to minimize defects and improve reproducibility of these algorithms.

## Installation
The library can simply be installed using pip:
```bash
pip install reinforcement
```

## Example Implementation
This section shows how to implement a REINFORCE agent and benchmark it on the 'CartPole' gym environment.

[embedmd]:# (example/reinforce.py python /def run_reinforce/ /env.close\(\)/)
```python
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
```
This is the main function setting up the boiler plate code, like creating the tensorflow session, logging reports, and creating the reinforce agent. The `Reporter` class is just a helper to make logging at a certain frequency more convenient

[embedmd]:# (example/reinforce.py python /def _make_agent/ /return BatchAgent\(alg\)/)
```python
def _make_agent(config, session, env):
    p = ParameterizedPolicy(session, env.observation_space.shape[0], env.action_space.n, NoLog(), config.lr_policy)
    b = ValueBaseline(session, env.observation_space.shape[0], NoLog(), config.lr_baseline)
    alg = Reinforce(p, config.gamma, b, config.num_trajectories)
    return BatchAgent(alg)
```

This is the factory function creating the REINFORCE agent. It uses a parameterized policy to learn and estimate proper actions and a parameterized baseline to calculate the advantage. The `BatchAgent` type records interactions between the agent and the environment as trajectories which are then used to optimize the policy and the baseline. The `NoLog` class is a Null-Object implementing the tensorboard `FileWriter` interface.

[embedmd]:# (example/reinforce.py python /def _run_episode/ /return reward/)
```python
def _run_episode(env, episode, agent, report):
    obs = env.reset()
    done, reward = False, 0
    while not done:
        if report.should_render(episode):
            env.render()
        obs, r, done, _ = env.step(agent.next_action(obs))
        agent.signal(r)
        reward += r

    return reward
```

The actual meat of the implementation, implementing a run through an episode of the environment. The observations of the environment are passed to the `next_action` interface of the agent and the estimated actions of the agents are passed to the environment. The agent is then trained at the end of the episode, because we want to train it on whole trajectories.

You can find the full implementation in [examples/reinforce.py](example/reinforce.py). The [example folder](example/) also contains some additional utility classes and functions that are used in the implementation.

## Running an Example
Running the REINFORCE agent example with default settings:
```bash
python example/reinforce.py
```

After a few 1000 episodes it should get very close to the highest achievable reward:
```
...
$ INFO:__main__:Episode 2800: reward=200.0; mean reward of last 100 episodes: 199.71
$ INFO:__main__:Episode 2900: reward=200.0; mean reward of last 100 episodes: 199.36
$ INFO:__main__:Episode 3000: reward=200.0; mean reward of last 100 episodes: 198.09
```
