from learning_environment.ghost_env_modified import GhostEnv, SimplePacmanAI
import random
from collections import deque
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

MOVE_TO_ACTION = {
    (-1, 0): 0,  # up
    (1, 0): 1,   # down
    (0, -1): 2,  # left
    (0, 1): 3    # right
}
ACTION_TO_MOVE = {v: k for k, v in MOVE_TO_ACTION.items()}

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        hidden_dim = 32
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
class DQNAgent:
    def __init__(self, state_dim, action_dim, alpha = 1e-3, gamma = 0.995, epsilon = 0.5, epsilon_min = 0.01, epsilon_decay = 0.99999, batch_size = 64, buffer_size = 100000, n_step = 8):
        self.device = torch.device("cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.memory = deque(maxlen = buffer_size)
        self.update_target_network()

        self.n_step = n_step
        self.n_step_buffer = deque(maxlen = n_step)
        
        self.total_train_steps = 0


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if done and reward > 50:
            # append successful transition 10 times to drastically increase its sampling rate
            for _ in range(10): 
                self.memory.append((state, action, reward, next_state, done))

    def get_state(self, state, shape, maze_layout):
        rows, cols = shape
        ghost_pos, pacman_pos = state
        ghost_r, ghost_c = ghost_pos
        pac_r, pac_c = pacman_pos
        rel_r = pac_r - ghost_r
        rel_c = pac_c - ghost_c

        # adding 3x3 grid around ghost maze info
        local_maze = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = ghost_r + dr, ghost_c + dc
                # 0 for wall, 1 for path/pellet (normalized)
                is_walkable = maze_layout[r, c] != 0 if 0 <= r < rows and 0 <= c < cols else 0
                local_maze.append(is_walkable)

        return np.array(
            [ghost_r / rows, ghost_c / cols, pac_r / rows, pac_c / cols, rel_r / rows, rel_c / cols] + local_maze, 
            dtype=np.float32
        )
    
    def choose_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)[0]
            # create mask of q-values for all actions, but setting invalid ones very low
            q_mask = torch.full_like(q_values, float('-inf'))
            q_mask[torch.tensor(valid_actions, dtype=torch.long)] = q_values[torch.tensor(valid_actions, dtype=torch.long)]
            best_action_id = torch.argmax(q_mask).item()
            return best_action_id
        
    def shape_reward(self, state, next_state, raw_reward):
        """Currently a pass-through: use the environment's raw reward.

        Kept as a separate method so reward shaping can be reintroduced easily
        if desired, but by default we train purely on env-defined returns.
        """
        return raw_reward

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # using double dqn
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def plot_training_steps(self, catch_list, steps_list):
        episodes = len(steps_list)
        x = np.arange(1, episodes + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(x, steps_list, color = 'blue', label = 'Steps per Episode', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Steps per Episode')
        plt.title('DQN Training Progress')
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

    def train_dqn(self, env, episodes, target_update = 500, debug = False):
        catch_list = []
        all_rewards = []
        moving_avg_rewards = []
        episode_steps = []

        # Early stopping variables
        best_avg_reward = float('-inf')
        stall_counter = 0
        stall_threshold = 10
        best_policy_state = None

        for ep in range(episodes):
            total_reward = 0
            state = self.get_state(env.reset(), env.maze_layout.shape, env.maze_layout)
            done = False
            caught = False

            while not done:
                valid_moves = env._get_valid_moves_from_position(env.ghost_pos)
                valid_actions = [MOVE_TO_ACTION[m] for m in valid_moves]  # convert to action ids
                action_id = self.choose_action(state, valid_actions)
                action = ACTION_TO_MOVE[action_id]  # convert back to (dr, dc)
                
                next_state_raw, reward, done, caught = env.step(action)
                next_state = self.get_state(next_state_raw, env.maze_layout.shape, env.maze_layout)

                shaped_reward = self.shape_reward(state, next_state, reward)
                total_reward += shaped_reward

                if debug and ep % 1000 == 0:
                    print(f"Step {env.steps}:")
                    print(f"  Ghost Pos: {env.ghost_pos}, Pac-Man Pos: {next_state_raw[1]}")
                    print(f"  Chosen Action: {action}, Reward: {reward}, Shaped Reward: {shaped_reward}")
                    print(f"  Done: {done}, Caught: {caught}")
                    print(f"  n-step buffer length: {len(self.n_step_buffer)}")
                    self.print_maze(env.maze_layout, env.ghost_pos, next_state_raw[1])
                    print("-"*30)
                
                self.n_step_buffer.append((state, action_id, shaped_reward, next_state, done))
                if len(self.n_step_buffer) == self.n_step:
                    # compute n-step return
                    G = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
                    s0, a0, _, _, d0 = self.n_step_buffer[0]
                    _, _, _, n_step_next_state, d_final = self.n_step_buffer[-1]
                    self.remember(s0, a0, G, n_step_next_state, d_final)
                    self.n_step_buffer.popleft()

                self.replay()
                self.total_train_steps += 1
                if self.total_train_steps % target_update == 0:
                    self.update_target_network()
                state = next_state

            while self.n_step_buffer:
                G = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(len(self.n_step_buffer))])
                s0, a0, _, _, d0 = self.n_step_buffer[0]
                _, _, _, n_step_next_state, d_final = self.n_step_buffer[-1]
                self.remember(s0, a0, G, n_step_next_state, d_final)
                self.n_step_buffer.popleft()

            # if env.steps % target_update == 0:
            #     self.update_target_network()

            catch_list.append(1 if caught else 0)
            episode_steps.append(env.steps)

            # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if ep % 100 == 0 and ep > 0:
                avg_reward_100 = np.mean(all_rewards[-100:])
                avg_steps_100 = np.mean(episode_steps[-100:])
                print(f"Episode {ep}/{episodes} | Epsilon={self.epsilon:.3f} | "
                      f"Avg Reward (100ep)={avg_reward_100:.2f} | Avg Steps (100ep)={avg_steps_100:.1f} | "
                      f"Caught={caught} | Replay size={len(self.memory)}")

            # Early stopping check every 50 episodes
            if ep % 50 == 0 and ep > 0:
                avg_reward_50 = np.mean(all_rewards[-50:])
                avg_steps_50 = np.mean(episode_steps[-50:])
                if avg_reward_50 > best_avg_reward:
                    best_avg_reward = avg_reward_50
                    best_avg_steps = avg_steps_50
                    stall_counter = 0
                    best_policy_state = self.policy_net.state_dict().copy()
                    print(f"  → New best avg reward (50ep): {best_avg_reward:.2f} | Avg steps (50ep): {best_avg_steps:.1f}")
                else:
                    stall_counter += 1
                    print(f"  → Stalled. Avg reward (50ep): {avg_reward_50:.2f} | Avg steps (50ep): {avg_steps_50:.1f} | Stall count: {stall_counter}/{stall_threshold}")
                    if stall_counter >= stall_threshold:
                        print(f"\nEarly stopping triggered after {ep} episodes (stalled {stall_threshold} times)")
                        break

            all_rewards.append(total_reward)

            avg_reward = np.mean(all_rewards[-100:])
            moving_avg_rewards.append(avg_reward)

        np.savez(os.path.join("results", "ghost_dqn_results.npz"),
                 rewards = np.array(all_rewards),
                 moving_avg = np.array(moving_avg_rewards),
                 steps = np.array(episode_steps)
                 )

        # Restore best policy before returning
        if best_policy_state is not None:
            self.policy_net.load_state_dict(best_policy_state)
            if 'best_avg_steps' in locals():
                print(f"\nBest policy restored (avg reward: {best_avg_reward:.2f}, avg steps (50ep): {best_avg_steps:.1f})")
            else:
                print(f"\nBest policy restored (avg reward: {best_avg_reward:.2f})")

        return catch_list, episode_steps