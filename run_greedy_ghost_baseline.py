import numpy as np
import random

from learning_environment.ghost_env_modified import GhostEnv


def blinky_style_action(env, state, last_move):
    """
    Approximate the actual game's Blinky chase behavior (the most aggressive ghost) as a baseline for performance
    """
    ghost_pos, pac_pos = state
    valid_moves = env._get_valid_moves_from_position(ghost_pos)

    # Avoid reversing direction if there is at least one other option
    if last_move is not None:
        rev = (-last_move[0], -last_move[1])
        non_reverse = [m for m in valid_moves if m != rev]
        if non_reverse:
            valid_moves = non_reverse

    best_move = None
    best_dist_sq = float("inf")

    for dr, dc in valid_moves:
        new_r = ghost_pos[0] + dr
        new_c = ghost_pos[1] + dc
        dist_sq = (new_r - pac_pos[0]) ** 2 + (new_c - pac_pos[1]) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_move = (dr, dc)

    # Fallback: if something went wrong, don't move
    if best_move is None:
        best_move = (0, 0)

    return best_move


def run_greedy_baseline(episodes: int = 1):
    # Use the same maze and start positions as the learning experiments
    test_maze = np.loadtxt("maze1_test.txt", dtype=int)
    test_smaller = np.loadtxt("maze_small.txt", dtype=int)

    ghost_start = (3, 5)
    pacman_start = (7, 5)

    random.seed(42)
    np.random.seed(42)

    env = GhostEnv(test_smaller, ghost_start, pacman_start)

    episode_rewards = []
    episode_steps = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        last_move = None
        while not done:
            action = blinky_style_action(env, state, last_move)
            last_move = action
            next_state, reward, done, caught = env.step(action)
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)
        episode_steps.append(env.steps)

        print(
            f"Episode {ep + 1}/{episodes} | Reward={total_reward:.2f} | "
            f"Steps={env.steps} | Caught={caught}"
        )

    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    avg_steps = float(np.mean(episode_steps)) if episode_steps else 0.0

    print("\nGreedy-towards-Pacman ghost baseline over",
          f"{episodes} episode(s):")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average steps:  {avg_steps:.2f}")

    return avg_reward, avg_steps


if __name__ == "__main__":
    run_greedy_baseline(episodes=1)

