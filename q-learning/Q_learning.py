import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    grid_size = env.grid_size
    q_table = np.zeros((grid_size, grid_size, env.action_space.n))

    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0

        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, done, reward, info = env.step(action)
            next_state_tuple = tuple(next_state)
            total_reward += reward

            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[next_state_tuple]) - q_table[state][action]
            )

            state = next_state_tuple
            env.render()

            if done or info["Life"] <= 0:
                env.render()
                env.render()
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode+1}/{no_episodes} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    env.close()

    # Backup old Q-table
    if os.path.exists(q_table_save_path):
        backup_path = q_table_save_path.replace(".npy", "_old.npy")
        shutil.copy(q_table_save_path, backup_path)
        print(f"Old Q-table backed up to: {backup_path}")

    np.save(q_table_save_path, q_table)
    print(f"Training finished. New Q-table saved to: {os.path.abspath(q_table_save_path)}")


def visualize_q_table(q_values_path="q_table.npy",
                      actions=["↑", "↓", "→", "←"],
                      food_coordinates=None,
                      danger_coordinates=None,
                      goal_coordinates=None):
    try:
        q_table = np.load(q_values_path)
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            data = q_table[:, :, i].copy()
            mask = np.zeros_like(data, dtype=bool)

            # Mark and mask food cells
            if food_coordinates:
                for fx, fy in food_coordinates:
                    data[fx, fy] = np.nan
                    mask[fx, fy] = True
                    ax.text(fy + 0.5, fx + 0.5, 'F', ha='center', va='center',
                            color='orange', fontsize=12, weight='bold')

            # Mark and mask danger zones
            if danger_coordinates:
                for dx, dy in danger_coordinates:
                    data[dx, dy] = np.nan
                    mask[dx, dy] = True
                    ax.text(dy + 0.5, dx + 0.5, 'H', ha='center', va='center',
                            color='red', fontsize=12, weight='bold')

            # Mark and mask goal cell
            if goal_coordinates:
                gx, gy = goal_coordinates
                data[gx, gy] = np.nan
                mask[gx, gy] = True
                ax.text(gy + 0.5, gx + 0.5, 'G', ha='center', va='center',
                        color='green', fontsize=12, weight='bold')

            # Plot heatmap
            sns.heatmap(data, annot=True, fmt=".2f", cmap="viridis", cbar=False,
                        ax=ax, linewidths=0.5, linecolor='white', mask=mask)

            ax.set_title(f"Action: {action}")

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(" Q-table not found. Train the agent first.")
