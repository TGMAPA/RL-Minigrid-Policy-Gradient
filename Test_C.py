from minigrid.wrappers import RGBImgObsWrapper
from minigrid_simple_env import SimpleEnv
from scipy.special import softmax
import numpy as np


# --- Función de actualización tipo Policy Gradient (REINFORCE paso a paso)
def update_params_with_softmax(params, selected_index, reward, lr=0.1):
    """
    Aplica el gradiente del log de la política con el reward actual (tipo REINFORCE).
    params: vector de parámetros del estado actual
    selected_index: acción seleccionada
    reward: recompensa obtenida después de ejecutar la acción
    """
    probs = softmax(params)
    grad = np.zeros_like(params)
    grad[selected_index] = 1.0  # vector one-hot de la acción tomada
    # Actualización tipo policy gradient: ∇θ log π(a|s) * R
    params += lr * (grad - probs) * reward
    return params


# --- Inicialización
def init_params(size, n_actions):
    params = {}
    for i in range(size):
        for j in range(size):
            for d in range(4):  # 4 direcciones posibles
                params[(i, j, d)] = np.zeros(shape=n_actions)
    return params


# --- Entrenamiento
def train(env, params, EPISODES, STEPS, ACTIONS, LR):
    # Exploración epsilon-greedy sobre softmax
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
            # Estado actual
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            # Selección de acción
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(ACTIONS)
            else:
                action = np.random.choice(range(ACTIONS), size=1, p=softmax(params[current_state]))[0]

            # Ejecutar acción
            obs, reward, terminated, truncated, info = env.step(action)
            reward -= 0.001  # penalización por paso
            total_reward += reward

            # Actualizar parámetros de la política en el mismo paso
            # (usamos reward inmediato o una forma de retorno estimado simple)
            params[current_state] = update_params_with_softmax(
                params[current_state], selected_index=action, reward=reward, lr=LR
            )

            # Terminar episodio si llega al objetivo o se trunca
            if terminated or truncated:
                if terminated:
                    success_count += 1
                break

        rewards_per_episode.append(total_reward)

        if episode%10 == 0:
            print(f"Episode {episode}/{EPISODES} — ε={epsilon:.3f} — Reward={total_reward:.2f}")

    print(f"\nSuccess rate: {success_count}/{EPISODES}")
    return params


# --- Prueba
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
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            # Acción determinista en test
            action = np.random.choice(range(len(params[current_state])), size=1, p=softmax(params[current_state]))[0]

            print(f"Step {steps}: pos={current_state}, action={actions_names[action]}")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        if terminated:
            success_count += 1

        print(f"Test Episode {episode} — Success={terminated}, Total Reward={total_reward:.2f}, Steps={steps}")

    print(f"\nSuccess rate during test: {success_count}/{EPISODES}")


# --- Main
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    SIZE = 7
    ACTIONS = 3
    EPISODES = 2000
    STEPS = 400
    LR = 0.02

    env = SimpleEnv(size=SIZE, render_mode=None)
    env = RGBImgObsWrapper(env)

    params = init_params(SIZE, ACTIONS)
    params = train(env, params, EPISODES, STEPS, ACTIONS, LR)

    test(env, params, STEPS, SIZE, EPISODES=5)
