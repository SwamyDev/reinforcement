[![Build Status](https://travis-ci.org/SwamyDev/reinforcement.svg?branch=master)](https://travis-ci.org/SwamyDev/reinforcement) [![Coverage Status](https://coveralls.io/repos/github/SwamyDev/reinforcement/badge.svg?branch=master)](https://coveralls.io/github/SwamyDev/reinforcement?branch=master) [![PyPI version](https://badge.fury.io/py/reinforcement.svg)](https://badge.fury.io/py/reinforcement)

# Reinforcement
The Reinforcement module aims to provide simple implementations for various reinforcement learning algorithms. The module tries to be agnostic about its use cases, but implements different solutions for policy selection, value- and q-function approximations as well as different agents for reinforcement learning algorithms. 

The project is in its early stage and currently only provides an n-step temporal difference learning agent. The main purpose of the project is to facilitate my own understanding of reinforcement learning, with no particular application in mind. 

## Module structure
The module is organises in 3 main parts. Policies, reward functions and agents, each providing necessary components to construct a reinforcement learning agent. Components should have a low dependency amongst each other and share a simple common interface to facilitate modular construction of agents.

### Agents
This module contains the actual agents implementing the reinforcement learning algorithm using a policy component and a reward function component. Currently only a n-step temporal difference agent is implemented.

### Policies
This module contains action selection policies used by reinforcement learning agents. Available policies: *epsilon greedy*; *normalized epsilon greedy*.

### Reward Functions
This module contains implementations of reward functions, which are used by reinforcement learning agents. Available reward functions: *value table*, *q table*, *q neural network*

## Models
Reinforcement also contains neural network implementation which can be used as non-linear reward function approximiations. Currently there are 2 regression models implemented, one using [Keras](https://keras.io/) and one using pure [Tensorflow](www.tensorflow.org). 

## Architecture
This software is crafted using Test Driven Development and tries to adhere to the SOLID principle as far as it lies in the abilities of the author.
