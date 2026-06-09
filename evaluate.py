import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from env.warehouse_env import WarehouseEnv
from agent.dqn_agent import DQNAgent

MODEL_PATH = "models/dqn_model.pth"


def run_episode(agent, env, greedy=True, render_steps=False):
    """Run one full episode. If greedy=True, always pick best action."""
    state = env.reset()
    done  = False
    total_reward = 0
    original_epsilon = agent.epsilon

    if greedy:
        agent.epsilon = 0.0  # Pure exploitation

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

        if render_steps:
            env.render()
            input("Press Enter for next step...")

    agent.epsilon = original_epsilon
    return total_reward, env.steps, env.get_path(), reward == 100


def visualize_path(env, path, title="Robot Path in Warehouse"):
    """Draw the warehouse grid with the robot's path."""
    fig, ax = plt.subplots(figsize=(8, 8))
    grid = env.fixed_grid
    n    = env.grid_size

    # Color map
    colors = {
        'free':     '#F0F4F8',
        'obstacle': '#2D3748',
        'path':     '#90CDF4',
        'start':    '#48BB78',
        'goal':     '#F6AD55',
        'robot':    '#FC8181',
    }

    # Draw grid cells
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                color = colors['obstacle']
            elif (r, c) in path:
                color = colors['path']
            else:
                color = colors['free']

            rect = plt.Rectangle([c, n - 1 - r], 1, 1,
                                  facecolor=color,
                                  edgecolor='#CBD5E0',
                                  linewidth=1)
            ax.add_patch(rect)

    # Mark start
    sr, sc = env.start_pos
    ax.add_patch(plt.Rectangle([sc, n - 1 - sr], 1, 1,
                                facecolor=colors['start'],
                                edgecolor='#CBD5E0', linewidth=1))
    ax.text(sc + 0.5, n - 0.5 - sr, 'S', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')

    # Mark goal
    gr, gc = env.goal_pos
    ax.add_patch(plt.Rectangle([gc, n - 1 - gr], 1, 1,
                                facecolor=colors['goal'],
                                edgecolor='#CBD5E0', linewidth=1))
    ax.text(gc + 0.5, n - 0.5 - gr, 'G', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')

    # Draw path arrows
    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        ax.annotate("",
            xy=(c2 + 0.5, n - 0.5 - r2),
            xytext=(c1 + 0.5, n - 0.5 - r1),
            arrowprops=dict(arrowstyle="->", color='#2B6CB0', lw=1.5)
        )

    # Grid lines
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.grid(True, color='#CBD5E0', linewidth=0.5)
    ax.set_aspect('equal')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['start'],    label='Start (S)'),
        mpatches.Patch(facecolor=colors['goal'],     label='Goal (G)'),
        mpatches.Patch(facecolor=colors['path'],     label='Path taken'),
        mpatches.Patch(facecolor=colors['obstacle'], label='Obstacle'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=10, framealpha=0.9)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("models/path_visualization.png", dpi=150)
    plt.show()
    print("Path visualization saved → models/path_visualization.png")


def compare_random_vs_trained(env, agent):
    """Side-by-side path comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    n = env.grid_size

    # --- Random agent path ---
    state = env.reset()
    done  = False
    while not done:
        action = np.random.randint(0, 4)
        _, _, done = env.step(action)
    random_path  = env.get_path()
    random_steps = env.steps

    # --- Trained agent path ---
    reward, trained_steps, trained_path, success = run_episode(agent, env)

    for ax, path, title, steps in zip(
        axes,
        [random_path, trained_path],
        ["Random Agent", f"Trained DQN Agent ({'✓ Goal Reached' if success else '✗ Failed'})"],
        [random_steps, trained_steps]
    ):
        grid = env.fixed_grid
        for r in range(n):
            for c in range(n):
                color = '#2D3748' if grid[r][c] == 1 else \
                        ('#90CDF4' if (r, c) in path else '#F0F4F8')
                ax.add_patch(plt.Rectangle([c, n - 1 - r], 1, 1,
                             facecolor=color, edgecolor='#CBD5E0', lw=0.8))

        # Start & Goal
        sr, sc = env.start_pos
        gr, gc = env.goal_pos
        ax.add_patch(plt.Rectangle([sc, n-1-sr], 1, 1, facecolor='#48BB78', edgecolor='#CBD5E0'))
        ax.text(sc+0.5, n-0.5-sr, 'S', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
        ax.add_patch(plt.Rectangle([gc, n-1-gr], 1, 1, facecolor='#F6AD55', edgecolor='#CBD5E0'))
        ax.text(gc+0.5, n-0.5-gr, 'G', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

        ax.set_xlim(0, n); ax.set_ylim(0, n)
        ax.set_xticks(range(n+1)); ax.set_yticks(range(n+1))
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.grid(True, color='#CBD5E0', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\nSteps taken: {steps}", fontsize=12, fontweight='bold')

    plt.suptitle("Random Agent vs Trained DQN Agent", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("models/comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison saved → models/comparison.png")


def evaluate(n_eval_episodes=20):
    env   = WarehouseEnv(grid_size=8, max_steps=200)
    agent = DQNAgent(state_size=env.state_size, n_actions=env.n_actions)

    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}. Train first with: python train.py")
        return

    agent.load(MODEL_PATH)

    print("\nRunning evaluation episodes...")
    successes, all_steps, all_rewards = [], [], []

    for ep in range(n_eval_episodes):
        reward, steps, path, success = run_episode(agent, env, greedy=True)
        successes.append(success)
        all_steps.append(steps)
        all_rewards.append(reward)
        status = "✓" if success else "✗"
        print(f"  Episode {ep+1:2d}: {status}  Steps: {steps:3d}  Reward: {reward:7.1f}")

    print(f"\n{'='*45}")
    print(f"Success Rate : {np.mean(successes)*100:.1f}%  ({sum(successes)}/{n_eval_episodes})")
    print(f"Avg Steps    : {np.mean(all_steps):.1f}")
    print(f"Avg Reward   : {np.mean(all_rewards):.1f}")
    print(f"{'='*45}")

    # Visualize best path
    _, _, best_path, _ = run_episode(agent, env, greedy=True)
    visualize_path(env, best_path, title=f"Trained DQN — Path to Goal ({env.steps} steps)")

    # Compare random vs trained
    compare_random_vs_trained(env, agent)


if __name__ == "__main__":
    evaluate()