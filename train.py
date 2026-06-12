# for episode in range(500):
#     state = env.reset()
#     while not done:
#         action = agent.select_action(state)
#         next_state, reward, done = env.step(action)
#         buffer.store(...)
#         agent.train_step()
#     decay epsilon
#     log rewardimport sys


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from env.warehouse_env import WarehouseEnv
from agent.dqn_agent import DQNAgent

# ─────────────────────────────────────────
#  Hyperparameters
# ─────────────────────────────────────────
EPISODES         = 600
MAX_STEPS        = 200
BATCH_SIZE       = 64
LR               = 0.001
GAMMA            = 0.99
EPSILON_START    = 1.0
EPSILON_END      = 0.01
EPSILON_DECAY    = 0.995
TARGET_UPDATE    = 10       # Update target net every N episodes
BUFFER_CAPACITY  = 10000
SAVE_PATH        = "models/dqn_model.pth"
PRINT_EVERY      = 50       # Print progress every N episodes

def train():
    env   = WarehouseEnv(grid_size=8, max_steps=MAX_STEPS)
    agent = DQNAgent(
        state_size       = env.state_size,
        n_actions        = env.n_actions,
        lr               = LR,
        gamma            = GAMMA,
        epsilon_start    = EPSILON_START,
        epsilon_end      = EPSILON_END,
        epsilon_decay    = EPSILON_DECAY,
        batch_size       = BATCH_SIZE,
        buffer_capacity  = BUFFER_CAPACITY,
        target_update_freq = TARGET_UPDATE,
    )

    rewards_history   = []
    steps_history     = []
    success_history   = []
    best_avg_reward   = -float('inf')

    print("=" * 55)
    print("   Warehouse Robot DQN Training Started")
    print("=" * 55)
    print(f"Grid: 8x8 | Episodes: {EPISODES} | Max Steps: {MAX_STEPS}")
    print(f"LR: {LR} | Gamma: {GAMMA} | Batch: {BATCH_SIZE}")
    print("=" * 55)

    for episode in range(1, EPISODES + 1):
        state   = env.reset()
        total_reward = 0
        done    = False
        reached_goal = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()

            state = next_state
            total_reward += reward

            if reward == 100:
                reached_goal = True

        # Post-episode updates
        agent.decay_epsilon()
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        rewards_history.append(total_reward)
        steps_history.append(env.steps)
        success_history.append(1 if reached_goal else 0)

        # Save best model
        if episode >= 50:
            avg_reward = np.mean(rewards_history[-50:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(SAVE_PATH)

        # Print progress
        if episode % PRINT_EVERY == 0:
            avg_r      = np.mean(rewards_history[-PRINT_EVERY:])
            avg_steps  = np.mean(steps_history[-PRINT_EVERY:])
            success_rt = np.mean(success_history[-PRINT_EVERY:]) * 100
            avg_loss   = np.mean(agent.losses[-100:]) if agent.losses else 0

            print(f"Ep {episode:4d}/{EPISODES} | "
                  f"Avg Reward: {avg_r:7.1f} | "
                  f"Avg Steps: {avg_steps:5.1f} | "
                  f"Success: {success_rt:5.1f}% | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")

    print("\n" + "=" * 55)
    print(f"Training Complete! Best Avg Reward: {best_avg_reward:.1f}")
    print(f"Model saved to: {SAVE_PATH}")
    print("=" * 55)

    # Save training history for plotting
    np.save("models/rewards_history.npy", np.array(rewards_history))
    np.save("models/steps_history.npy",   np.array(steps_history))
    np.save("models/success_history.npy", np.array(success_history))
    print("Training history saved to models/")

    return rewards_history, steps_history, success_history


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()