from algorithms.q_learning import QLearningAgent
from learning_environment.ghost_env import GhostEnv
import numpy as np
import random

test_maze = np.loadtxt("maze1_test.txt", dtype=int)
test_smaller = np.loadtxt("maze_small.txt", dtype=int)

# start ghost at random intersecton
ghost_start = (3, 5)
pacman_start = (7,5)

random.seed(42)

# tested with ghost_env_modified file
env = GhostEnv(test_smaller, ghost_start, pacman_start)
# print(env.get_intersections())
agent = QLearningAgent()
catch_list, step_list = agent.train_q_learning(env, episodes=2000)
agent.plot_training_steps(catch_list, step_list)