# Use the same maze and start positions as the learning experiments
import random
import numpy as np
from algorithms.baseline import Baseline
from learning_environment.ghost_env import GhostEnv


test_maze = np.loadtxt("maze1_test.txt", dtype=int)
test_smaller = np.loadtxt("maze_small.txt", dtype=int)

ghost_start = (3, 5)
pacman_start = (7, 5)

random.seed(42)

env = GhostEnv(test_smaller, ghost_start, pacman_start)

agent = Baseline()
avg_reward, avg_steps = agent.run_greedy_baseline(env, episodes=100)

print(f"Average reward: {avg_reward:.2f}")
print(f"Average steps: {avg_steps:.2f}")