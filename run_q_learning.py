from algorithms.q_learning import QLearningAgent
from learning_environment.ghost_env import GhostEnv
import numpy as np
import random
import json

def save_QL_policy(policy, filename="QL_policy.json"):
    with open(filename, "w") as f:
        print("Saving policy to file")
        json.dump({str(k): v for k, v in policy.items()}, f)

def load_QL_policy(filename="QL_policy.json"):
    with open(filename, "r") as f:
        raw = json.load(f)
    return {eval(k): tuple(v) for k, v in raw.items()}

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
# print(env.get_intersections())
agent = QLearningAgent()
catch_list, step_list = agent.train_q_learning(env, episodes=5000)
best_policy = agent.get_best_policy(env)
print(best_policy.get(((14,15), (26,15))))
save_QL_policy(best_policy)



agent.plot_training_steps(catch_list, step_list)

