# RL Atari Player

The RL Atari Player uses reinforcement learning to obtain information about Atari 2600 games from the screen and produce optimal moves.

### Introduction

A reinforcement learning (RL) agent uses rewards and observations from the environment to adapt and make decisions. 
In a Markov Decision Process, the agent can choose its next action purely based on the current state and the set of possible next states.
In the discrete space (for example, a 10 x 10 grid), each state-action pair can be converted into a table (called the Q-table) that updates after every time the agent traverses through the environment from a predetermined 'start' to 'finish'.
This is called Q-Learning, and it is a problem that can be solved using dynamic programming.

When scaling to a much larger and more complex environments, such as in video games, the traditional Q-table becomes less feasible; since there can be a very large number of states, it becomes unrealistic to store every piece of information in a table.
This problem can be solved by using a neural network to estimate the next set of actions at any given state. 
Called the Deep Q Network (DQN), this approach allows the agent to calculate "trajectories" between time steps and apply the knowledge to make the next decision.

A significant improvement that can be made to a DQN is through the *epsilon-greedy* strategy. This hyperparameter controls the "exploration vs. exploitation" factor.
An intuitive role of the epsilon is a probability (between 0.0 and 1.0) that the agent will choose a random action over one that it knows is "safe". This allows the algorithm
to escape potential local minima and expand its understanding of the environment.

### Gameplay

The DQN included in this repository learns to "play" a variety of Atari 2600 games by observing the game screen. 
Although a popular method uses a Convolutional Neural Network (CNN) to process the 210 x 160 pixel image, the algorithm described here uses the [OpenAI Gym](https://gym.openai.com/envs/#atari) environments with 128 bytes of RAM as input.

### Files

The main DQN class can be found in `dqn.py`. An example usage of the class can be found under `demo.py`, where a number of different hyperparameters including the number of iterations and learning rate can be controlled.
