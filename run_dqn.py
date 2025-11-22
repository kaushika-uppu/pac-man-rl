from algorithms.dqn import DQNAgent
from learning_environment.ghost_env import GhostEnv
import numpy as np
import random

test_maze = np.loadtxt("maze1_test.txt", dtype=int)
test_smaller = np.loadtxt("maze_small.txt", dtype=int)

ghost_start = (3, 5)
pacman_start = (7, 5)

random.seed(42)

env = GhostEnv(test_smaller, ghost_start, pacman_start)

# input: (ghost_r, ghost_c, pacman_r, pacman_c, rel_r, rel_c, 3x3 grid around ghost (dim 9))
state_dim = 15
# output: number of possible movement directions (up, down, left, right)
action_dim = 4

agent = DQNAgent(state_dim, action_dim)
catch_list, steps_list = agent.train_dqn(env, episodes=2000)

agent.plot_training_steps(catch_list, steps_list)