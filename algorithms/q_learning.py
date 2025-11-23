from learning_environment.ghost_env import GhostEnv, SimplePacmanAI
import random
from collections import defaultdict
import time
import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.5, epsilon_min=0.05, epsilon_decay=0.9995):
        self.Q = defaultdict(float)
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon 
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def format_state_action(self, state, action):
        """
        Input:  state -> ghost position tuple (r, c), pacman position (r, c)
                action (r, c)
        Output: Tuple(ghost row, ghost column, pacman row, pacman column, action row, action column)
        """
        return (state[0][0], state[0][1], state[1][0], state[1][1], action[0], action[1])


    def get_q_value(self, state, action):
        """Get the q value of state and action"""
        return self.Q[self.format_state_action(state, action)]
    
    def get_q_table(self):
        return self.Q

    def find_best_action(self, state, valid_actions):
        """
        Returns the best action among the valid ones
        Output: best action, best Q value
        """
        best_action = None
        best_q = -float('inf')

        for a in valid_actions:
            q = self.get_q_value(state, a)
            if q > best_q or best_action is None:
                best_action, best_q = a, q

        return best_action, best_q

    def choose_greedy_action(self, state, valid_actions):
        """Chooses an action using ε-greedy"""

        # Explore a random action
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Pick the best action
        a, q = self.find_best_action(state, valid_actions)
        return a

    def update(self, state, action, reward, next_state, next_valid_actions):
        """
        Perform the Q-learning update
        Q(S,A)←Q(S,A)+α(R+γQ(S′,A′)−Q(S,A))
        """
        state_action = self.format_state_action(state, action)

        # no more future rewards if goal is reached (no next valid actions)
        if not next_valid_actions:
            target = reward
        else:
            a, next_q = self.find_best_action(next_state, next_valid_actions)
            target = reward + self.gamma * next_q
       

        self.Q[state_action] += self.alpha * (target - self.Q[state_action])

    def plot_training_steps(self, catch_list, steps_list):
        episodes = len(steps_list)
        episodes_axis = np.arange(1, episodes + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(episodes_axis, steps_list, color='blue', alpha=0.7, label='Steps per Episode')
        # Mark episodes where Pac-Man was caught
        caught_eps = [i + 1 for i, c in enumerate(catch_list) if c == 1]
        caught_steps = [steps_list[i] for i, c in enumerate(catch_list) if c == 1]
        # plt.scatter(caught_eps, caught_steps, color='red', marker='o', s=40, label='Pac-Man Caught')
        plt.xlabel('Episode')
        plt.ylabel('Steps per Episode', color='blue')
        plt.title('Q-learning Training: Steps per Episode')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        plt.show()

    def print_maze(self, maze, ghost_pos, pac_pos):
        rows, cols = maze.shape
        for r in range(rows):
            for c in range(cols):
                if (r, c) == ghost_pos:
                    print("\033[91mG\033[0m", end=" ")  # red
                elif (r, c) == pac_pos:
                    print("\033[93mP\033[0m", end=" ")  # yellow
                elif maze[r, c] == 0:
                    print("#", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

    def train_q_learning(self, env, episodes, debug=False):
        """
        Train ghost agent using Q-learning 
        Input: GhostEnv, QLearningAgent, number of episodes 
        """
        steps_list = []
        catch_list = []
        rewards_list = []
        for ep in range(episodes):
            print('Running episode ', ep+1)
            caught = False

            # reset environment
            state = env.reset()
            total_reward = 0
            step = 0

            while True:
                ghost_position, pacman_position = state
                valid_actions = env._get_valid_moves_from_position(ghost_position)

                # print(valid_actions)

                # epsilon greedy selection
                action = self.choose_greedy_action(state, valid_actions)

                # execute one step of environment
                next_state, reward, done, caught = env.step(action)
                total_reward += reward
                # print('Reward: ', total_reward)

                if debug and ep % 1000 == 0:
                    self.print_maze(env.maze_layout, env.ghost_pos, next_state[1])

                # check next valid actions
                next_valid = env._get_valid_moves_from_position(next_state[0]) if not done else []

                # update q-value
                self.update(state, action, reward, next_state, next_valid)

                # Update current state
                state = next_state

                if done:
                    break
            if caught:
                print("caught")
                catch_list.append(1)
            else:
                catch_list.append(0)

            rewards_list.append(total_reward)
            steps_list.append(env.steps)

            # Log running averages over the last 50 episodes
            if len(steps_list) >= 50:
                avg_steps_50 = np.mean(steps_list[-50:])
                avg_reward_50 = np.mean(rewards_list[-50:])
                print(f"  Avg over last 50 episodes -> steps: {avg_steps_50:.2f}, reward: {avg_reward_50:.2f}")

            # epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return catch_list, steps_list
    
    def get_best_policy(self, env):
        """
        Computes the best policy from the Q-table
        Output: dict (ghost tuple, pacman tuple) -> best action tuple  
        """ 
        policy = {}
        for key, q_val in self.Q.items():
            ghost_r, ghost_c, pac_r, pac_c, dr, dc = key
            state = ((ghost_r, ghost_c), (pac_r, pac_c))
            action = (dr, dc)
            if state not in policy or q_val > self.get_q_value(state, policy[state]):
                policy[state] = action
        return policy






                

                

        





