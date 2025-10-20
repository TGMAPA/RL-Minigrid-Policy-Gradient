from minigrid.wrappers import RGBImgObsWrapper
from minigrid_simple_env import SimpleEnv
import numpy as np


def init_Q_Table(size, n_actions):
    q_table = {}
    for i in range(size):
        for j in range(size):
            for d in range(4):  # 4 directions
                q_table[(i, j, d)] = np.zeros(shape=n_actions)
    return q_table


def q_learning_eq(lr, reward, discount_factor, Qk, maxQ):
    return Qk + lr * (reward + discount_factor * maxQ - Qk)


def Qmax_state(Qtable, current_state):
    q_values = Qtable[current_state]
    max_idx = np.argmax(q_values)
    max_val = q_values[max_idx]
    return max_val, max_idx

def train(env, qTable, EPISODES, STEPS, ACTIONS, LR, DISCOUNT_FACTOR):
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
                # MaxState
                _, action = Qmax_state(qTable, current_state)

            # Execute step with selected action
            obs, reward, terminated, truncated, info = env.step(action)

            reward -= 0.001  # Add a small penalty for each step
            total_reward += reward

            # Get new state
            new_pos = tuple(env.unwrapped.agent_pos)
            new_dir = env.unwrapped.agent_dir
            next_state = (new_pos[0], new_pos[1], new_dir)

            # Apply qlearning eq for update
            qmax = np.max(qTable[next_state])
            qTable[current_state][action] = q_learning_eq(LR, reward, DISCOUNT_FACTOR, qTable[current_state][action], qmax)

            if terminated or truncated:
                if terminated:
                    success_count += 1
                break

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode}/{EPISODES} — ε={epsilon:.3f} — Reward={total_reward:.2f}")

    print(f"\nSuccess rate: {success_count}/{EPISODES}")
    return qTable


def test(env, qTable, STEPS, SIZE, EPISODES):
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

            # Get max Q action
            _, q_action = Qmax_state(qTable, current_state)

            print(f"Step {steps}: pos={current_state}, action={actions_names[q_action]}")

            # Get new state
            obs, reward, terminated, truncated, info = env.step(q_action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                # If the mission was completed then break the episode
                break

        if terminated:
            success_count += 1

        print(f"Test Episode {episode} — Success={terminated}, Total Reward={total_reward:.2f}, Steps={steps}")

    print(f"\nSuccess rate during test: {success_count}/{EPISODES}")
