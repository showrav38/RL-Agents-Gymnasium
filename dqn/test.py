import torch
import numpy as np
from env import ContinuousMazeEnv
from DQN_model import DQN
from utils import select_action

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_consecutive_dqn(num_episodes=100):
    # Initialize environment with rendering
    env = ContinuousMazeEnv(render_mode="human")
    
    # Load trained model
    policy_net = DQN(input_dim=2, output_dim=4).to(DEVICE)
    try:
        policy_net.load_state_dict(torch.load("dqn_model.pth"))
    except FileNotFoundError:
        print("Error: dqn_model.pth not found. Ensure training completed and the file is in the directory.")
        return
    policy_net.eval()

    consecutive_successes = 0
    episode_results = []

    # Log file for test results
    with open("test_consecutive_log.txt", "w") as f:
        f.write("Episode,Total Reward,Steps,Success\n")

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Select action with epsilon=0.0 (greedy policy)
            action = select_action(state, policy_net, 0.0, env.action_space, DEVICE)
            state, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            print(f"Episode {episode+1}, Step {steps}, State: {state}, Reward: {reward:.2f}, Done: {done}")
            env.clock.tick(10)  # 10 FPS for visibility

        # Check if episode was successful (reached goal within radius)
        success = info["distance_to_goal"] <= env.goal_radius
        if success:
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        episode_results.append((total_reward, steps, success))
        print(f"Test Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {steps}, Success = {success}, Consecutive Successes = {consecutive_successes}")

        # Log results
        with open("test_consecutive_log.txt", "a") as f:
            f.write(f"{episode+1},{total_reward:.2f},{steps},{success}\n")

        if consecutive_successes >= 100:
            print(f"Achieved {consecutive_successes} consecutive successes!")
            break

    # Summary
    success_rate = sum(1 for _, _, success in episode_results if success) / len(episode_results)
    print(f"Summary: Success Rate = {success_rate*100:.2f}%, Consecutive Successes = {consecutive_successes}")
    
    env.close()

if __name__ == "__main__":
    test_consecutive_dqn(num_episodes=100)
