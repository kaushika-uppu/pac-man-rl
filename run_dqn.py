from algorithms.dqn import DQNAgent
from learning_environment.ghost_env import GhostEnv
import numpy as np
import random

# maze from ui
test_maze = np.loadtxt("maze1_test.txt", dtype=int)
# small maze for testing
test_smaller = np.loadtxt("maze_small.txt", dtype=int)
# small maze with portals for testing
test_portal = np.loadtxt("maze_small_portal.txt", dtype=int)

# maze_smaller start positions
# ghost_start = (3,5)
# pacman_start = (7,5)

# test_maze start positions
pacman_start = (26,15)
ghost_start = (14,15)

# portal locations for test_portal
# portals = {(3,13): (5,0), (5,0): (3,13)}

# portal locations for test_maze
portals = {(17,0): (17,27), (17,27): (17,0)}

random.seed(42)

env = GhostEnv(test_maze, ghost_start, pacman_start, portals)

# input: (ghost_r, ghost_c, pacman_r, pacman_c, rel_r, rel_c, 3x3 grid around ghost (dim 9))
state_dim = 15
# output: number of possible movement directions (up, down, left, right)
action_dim = 4

agent = DQNAgent(state_dim, action_dim)
catch_list, steps_list = agent.train_dqn(env, episodes=2000)

agent.plot_training_steps(catch_list, steps_list)