from minigrid.wrappers import RGBImgObsWrapper
from minigrid_simple_env import SimpleEnv
from scipy.special import softmax
import numpy as np

# -- Policy Gradient functions
def update_params_with_softmax(params, selected_index, reward, lr=0.1):
    probs = softmax(params)
    grad = np.zeros_like(params)
    grad[selected_index] = 1  # sólo la seleccionada obtiene refuerzo
    params += lr * (grad - probs) * reward
    return params

# Initialize parameters
def init_params(size, n_actions):
    params = {}
    for i in range(size):
        for j in range(size):
            for d in range(4):  # 4 directions
                params[(i, j, d)] = np.zeros(shape=n_actions)
    return params

# Training function
def train(env, params, EPISODES, STEPS, ACTIONS, LR):
    # Exploration parameters
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.001

    success_count = 0
    rewards_per_episode = []

    for episode in range(1, EPISODES + 1):
        total_reward = 0
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        obs, _ = env.reset()
        terminated = False

        for step in range(STEPS):
            # Get current position and direction 
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            # Action selection
            if np.random.uniform(0, 1) < epsilon:
                # Random selection
                action = np.random.randint(ACTIONS)
            else:
                # Softmax choice
                action =np.random.choice(range(ACTIONS), size=1, p=softmax(params[current_state]))[0]

            # Execute step with selected action
            obs, reward, terminated, truncated, info = env.step(action)

            reward -= 0.001  # Add a small penalty for each step
            total_reward += reward

            # Get new state
            new_pos = tuple(env.unwrapped.agent_pos)
            new_dir = env.unwrapped.agent_dir
            next_state = (new_pos[0], new_pos[1], new_dir)

            # Apply qlearning eq for update
            params[current_state] = update_params_with_softmax(params[current_state], action, total_reward, LR)

            if terminated or truncated:
                if terminated:
                    success_count += 1
                break

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode}/{EPISODES} — ε={epsilon:.3f} — Reward={total_reward:.2f}")

    print(f"\nSuccess rate: {success_count}/{EPISODES}")
    return params

# Testing function
def test(env, params, STEPS, SIZE, EPISODES):
    print("\nEvaluando agente entrenado...\n")

    env = SimpleEnv(size=SIZE, render_mode="human")
    env = RGBImgObsWrapper(env)

    actions_names = ["Left", "Right", "Forward"]
    success_count = 0

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0
        steps = 0

        while not terminated and steps < STEPS:
            # Get current position and direction = State
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            # Softmax choice
            action =np.random.choice(range(len(params[current_state])), size=1, p=softmax(params[current_state]))[0]

            print(f"Step {steps}: pos={current_state}, action={actions_names[action]}")

            # Get new state
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                # If the mission was completed then break the episode
                break

        if terminated:
            success_count += 1

        print(f"Test Episode {episode} — Success={terminated}, Total Reward={total_reward:.2f}, Steps={steps}")

    print(f"\nSuccess rate during test: {success_count}/{EPISODES}")
