# Importación de librerías
from minigrid.wrappers import RGBImgObsWrapper
from minigrid_simple_env import SimpleEnv
import numpy as np
from scipy.special import softmax

# Inicializa los parámetros de la política para cada estado que tiene un vector de pesos que define la probabilidad de cada acción
def init_params(size, n_actions):
    params = {}
    for i in range(size):
        for j in range(size):
            for d in range(4):  # Direcciones posibles
                # Inicializar los parámetros
                params[(i, j, d)] = np.zeros(n_actions)
    return params

# Selecciona una acción según los parametros actuales
def select_action(params, state):
    probs = softmax(params[state])
    action = np.random.choice(len(probs), p=probs)
    return action, probs

# Calcula las recompensas (reward-to-go) como : G_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
def compute_rtgo(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

# Entrenamiento del agente mediante REINFORCE (Policy Gradient)
def train(env, policy, EPISODES, STEPS, LR, DISCOUNT_FACTOR):
    print("\nEntrenamiento del agente...\n")

    success_count = 0
    rewards_per_episode = []

    # Entrenamiento
    for episode in range(1, EPISODES + 1):
        # Listas para almacenar trayectoria del episodio
        states, actions, rewards = [], [], []

        obs, _ = env.reset()
        terminated = False
        total_reward = 0

        for step in range(STEPS):
            # Obtener el estado actual
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            # Seleccionar acción según política
            action, probs = select_action(policy, current_state)

            # Ejecutar acción en el entorno
            obs, reward, terminated, truncated, info = env.step(action)

            # Penalización leve por paso
            reward -= 0.001  
            total_reward += reward

            # Guardar trayectoria
            states.append(current_state)
            actions.append(action)
            rewards.append(reward)

            if terminated or truncated:
                if terminated:
                    success_count += 1
                break

        # Calcular recompensas
        rtgo = compute_rtgo(rewards, DISCOUNT_FACTOR)

        # Actualizar parámetros de la política
        for state, action, Gt in zip(states, actions, rtgo):
            probs = softmax(policy[state])
            grad_log = -probs
            grad_log[action] += 1.0  
            policy[state] += LR * Gt * grad_log 

        rewards_per_episode.append(total_reward)

        # Log
        if episode % 100 == 0:
            print(f"Episode {episode}/{EPISODES} — Reward={total_reward:.2f}")

    # Mostrar tasa de éxito final
    print(f"\nSuccess rate: {success_count}/{EPISODES}")

    return policy, rewards_per_episode


# Evaluación del agente entrenado sin exploración
def test(env, policy, STEPS, SIZE, EPISODES):
    print("\nTest del agente entrenado...\n")

    # Crear entorno con renderizado visual
    env = SimpleEnv(size=SIZE, render_mode="human")
    env = RGBImgObsWrapper(env)

    success_count = 0

    # Ejecutar episodios de prueba
    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0
        steps = 0

        # Ejecutar pasos hasta que el episodio termine
        while not terminated and steps < STEPS:
            # Obtener el estado actual
            pos = tuple(env.unwrapped.agent_pos)
            dir = env.unwrapped.agent_dir
            current_state = (pos[0], pos[1], dir)

            # Elegir la acción más probable según la política
            probs = softmax(policy[current_state])
            action = np.argmax(probs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                # Si llega al objetivo o se termina el episodio cortar el ciclo
                break

        if terminated:
            success_count += 1

        print(f"- Test Episode {episode} — Success={terminated}, Total Reward={total_reward:.2f}, Steps={steps}")

    # Mostrar resultados finales
    print(f"\nSuccess rate during test: {success_count}/{EPISODES}")
