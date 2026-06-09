import numpy as np
import matplotlib.pyplot as plt
import os


def smooth(data, window=20):
    """Moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_training_results():
    paths = {
        "rewards": "models/rewards_history.npy",
        "steps":   "models/steps_history.npy",
        "success": "models/success_history.npy",
    }

    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"Missing: {path}. Run train.py first.")
            return

    rewards = np.load(paths["rewards"])
    steps   = np.load(paths["steps"])
    success = np.load(paths["success"])
    episodes = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Warehouse Robot DQN — Training Results", fontsize=16, fontweight='bold')

    colors = ['#4299E1', '#48BB78', '#ED8936', '#9F7AEA']

    # ── Plot 1: Reward per episode ──
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color=colors[0], linewidth=0.8)
    ax.plot(episodes[19:], smooth(rewards, 20), color=colors[0], linewidth=2, label='Avg (20 ep)')
    ax.set_title("Total Reward per Episode", fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Plot 2: Steps per episode ──
    ax = axes[0, 1]
    ax.plot(episodes, steps, alpha=0.3, color=colors[1], linewidth=0.8)
    ax.plot(episodes[19:], smooth(steps, 20), color=colors[1], linewidth=2, label='Avg (20 ep)')
    ax.set_title("Steps per Episode (lower = better)", fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Plot 3: Success rate (rolling) ──
    ax = axes[1, 0]
    window = 50
    rolling_success = [np.mean(success[max(0, i-window):i+1]) * 100 for i in range(len(success))]
    ax.plot(episodes, rolling_success, color=colors[2], linewidth=2)
    ax.fill_between(episodes, rolling_success, alpha=0.2, color=colors[2])
    ax.set_title(f"Success Rate (rolling {window} ep)", fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3)

    # ── Plot 4: Summary stats bar chart ──
    ax = axes[1, 1]
    first_half  = rewards[:len(rewards)//2]
    second_half = rewards[len(rewards)//2:]
    bars = ax.bar(
        ['First Half\n(avg reward)', 'Second Half\n(avg reward)'],
        [np.mean(first_half), np.mean(second_half)],
        color=[colors[3], colors[0]], width=0.4, edgecolor='white'
    )
    ax.set_title("Avg Reward: Early vs Late Training", fontweight='bold')
    ax.set_ylabel("Average Reward")
    ax.grid(alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f'{h:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("models/training_results.png", dpi=150)
    plt.show()
    print("Training plots saved → models/training_results.png")

    # Print summary
    print(f"\n{'='*45}")
    print(f"Total Episodes : {len(rewards)}")
    print(f"Final Success  : {np.mean(success[-50:])*100:.1f}% (last 50 ep)")
    print(f"Best Reward    : {max(rewards):.1f}")
    print(f"Avg Steps (last 50): {np.mean(steps[-50:]):.1f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    plot_training_results()