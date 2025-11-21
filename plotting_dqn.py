import numpy as np
import matplotlib.pyplot as plt

data = np.load("results/ghost_dqn_results.npz")

rewards = data["rewards"]
moving_avg = data["moving_avg"]
steps = data["steps"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ----- subplot 1: rewards -----
ax1.plot(rewards, label="Reward per Episode", alpha=0.5)
ax1.plot(moving_avg, label="100-Episode Moving Avg", linewidth=2)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.set_title("Pac-Man Ghost DQN Reward")
ax1.grid(alpha=0.3)
ax1.legend()

# ----- subplot 2: steps -----
ax2.plot(steps, label="Steps per Episode", color='orange')
ax2.set_xlabel("Episode")
ax2.set_ylabel("Steps")
ax2.set_title("Pac-Man Ghost DQN Steps")
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()