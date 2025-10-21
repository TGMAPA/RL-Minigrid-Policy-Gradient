# PolicyGradient.py
from minigrid.wrappers import RGBImgObsWrapper
from minigrid_simple_env import SimpleEnv
from scipy.special import softmax
import numpy as np

# -------------------------
# Helpers
# -------------------------
def init_params(size, n_actions):
    """Inicializa un diccionario de parámetros por (x, y, dir)."""
    params = {}
    for i in range(size):
        for j in range(size):
            for d in range(4):
                params[(i, j, d)] = np.zeros(shape=n_actions, dtype=float)
    return params


def softmax_probs(logits):
    """Devuelve una distribución de probabilidad segura numéricamente."""
    p = softmax(logits)
    p = p / np.sum(p)  # normaliza por seguridad
    return p


def compute_returns(rewards, gamma=0.99):
    """Calcula los retornos acumulados (Return-to-go)."""
    T = len(rewards)
    returns = np.zeros(T, dtype=float)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


# -------------------------
# Train (REINFORCE)
# -------------------------
def train(env, params, EPISODES, STEPS, ACTIONS, LR, gamma=0.99, normalize_returns=True):
    """
    Entrena con el algoritmo REINFORCE (Monte Carlo Policy Gradient).
    """
    action_map = {
        0: env.actions.left,
        1: env.actions.right,
        2: env.actions.forward
    }

    success_count = 0
    rewards_per_episode = []

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        states, actions, rewards = [], [], []
        terminated = False

        # -------------------------
        # Interacción episodio
        # -------------------------
        for step in range(STEPS):
            pos = tuple(env.unwrapped.agent_pos)
            d = env.unwrapped.agent_dir
            state = (pos[0], pos[1], d)

            logits = params[state]
            probs = softmax_probs(logits)

            # Política estocástica (sampling)
            action_idx = np.random.choice(ACTIONS, p=probs)
            real_action = action_map[action_idx]

            obs, reward, terminated, truncated, info = env.step(real_action)
            reward -= 0.001  # penalización leve por paso

            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)

            if terminated or truncated:
                break

        # -------------------------
        # Cálculo de retornos
        # -------------------------
        returns = compute_returns(rewards, gamma=gamma)

        if normalize_returns and len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # -------------------------
        # Actualización de parámetros
        # -------------------------
        for s, a, Gt in zip(states, actions, returns):
            logits = params[s]
            probs = softmax_probs(logits)

            grad_log = np.zeros_like(logits)
            grad_log[a] = 1.0
            grad_log -= probs  # gradiente de log π(a|s)

            # Actualización ascendente (maximize expected return)
            new_logits = logits + LR * Gt * grad_log

            # ✅ Asegurar que sigue siendo numpy array y no se corrompe
            params[s] = np.array(new_logits, dtype=float)

        episode_reward = np.sum(rewards)
        rewards_per_episode.append(episode_reward)

        if terminated:
            success_count += 1

        # Logging
        if ep % 50 == 0 or ep == 1:
            avg_last = np.mean(rewards_per_episode[-50:]) if len(rewards_per_episode) >= 1 else 0.0
            print(f"Ep {ep}/{EPISODES} | R_ep={episode_reward:.2f} | avg50={avg_last:.3f} | successes={success_count}")

    print(f"\n✅ Training finished. Successes: {success_count}/{EPISODES}")
    return params, rewards_per_episode


# -------------------------
# Test (no learning)
# -------------------------
def test(env, params, STEPS, SIZE, EPISODES, render=True):
    print("\nEvaluating trained agent (deterministic argmax policy)\n")

    # Recrear entorno para renderizado
    env = SimpleEnv(size=SIZE, render_mode="human" if render else None)
    env = RGBImgObsWrapper(env)

    action_map = {
        0: env.actions.left,
        1: env.actions.right,
        2: env.actions.forward
    }
    actions_names = ["Left", "Right", "Forward"]

    success_count = 0

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0
        steps = 0

        while not terminated and steps < STEPS:
            pos = tuple(env.unwrapped.agent_pos)
            d = env.unwrapped.agent_dir
            s = (pos[0], pos[1], d)

            # Acción determinista (greedy)
            logits = params[s]
            action_idx = int(np.argmax(logits))
            real_action = action_map[action_idx]

            print(f"Step {steps}: pos={s}, action={actions_names[action_idx]}")

            obs, reward, terminated, truncated, info = env.step(real_action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        if terminated:
            success_count += 1
        print(f"Test Ep {ep} — Success={terminated}, Reward={total_reward:.2f}, Steps={steps}")

    print(f"\n✅ Test success rate: {success_count}/{EPISODES}")
