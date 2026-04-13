from JungleEscapeEnv import JungleEscapeEnv
from Q_learning import train_q_learning, visualize_q_table

# === Settings ===
train = True
visualize_results = True

# Enable random initialization to avoid local minima
random_initialization = True
fast_mode = True  # Fast mode skips pygame wait

# Q-learning hyperparameters
learning_rate = 0.01
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
no_episodes = 1000

# Goal and special cell coordinates
goal_coordinates = (5, 5)
danger_coordinates = [(2, 2), (3, 4)]
food_coordinates = [(0, 1), (1, 0), (1, 2), (2, 1), (3, 0)]

# === Execution ===
if train:
    print("Starting training...")
    env = JungleEscapeEnv(
        grid_size=6,
        fast_mode=fast_mode,
        random_initialization=random_initialization
    )

    train_q_learning(
        env=env,
        no_episodes=no_episodes,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        alpha=learning_rate,
        gamma=gamma,
        q_table_save_path="q_table.npy"
    )

if visualize_results:
    visualize_q_table(
        q_values_path="q_table.npy",
        food_coordinates=food_coordinates,
        danger_coordinates=danger_coordinates,
        goal_coordinates=goal_coordinates
    )
