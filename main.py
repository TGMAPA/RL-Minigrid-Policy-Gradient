import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Test_C import *
from minigrid.wrappers import RGBImgObsWrapper


def main():
    SIZE = 7              # tamaño del entorno
    ACTIONS = 3           # Left, Right, Forward
    EPISODES = 1500       # más episodios para aprendizaje estable
    STEPS = 400
    LR = 0.05

    # Crear entorno
    env = SimpleEnv(size=SIZE, render_mode=None)
    env = RGBImgObsWrapper(env)

    # Inicializar parámetros (policy)
    params = init_params(SIZE, ACTIONS)

    # Entrenar agente
    params = train(env, params, EPISODES, STEPS, ACTIONS, LR)

    # ---- Graficar el "valor medio" de la política por celda ----
    table = np.zeros((SIZE, SIZE))
    """
    for i in range(SIZE):
        for j in range(SIZE):
            values = []
            for d in range(4):  # las 4 orientaciones del agente
                probs = softmax(params[(i, j, d)])
                values.append(np.max(probs))  # confianza en la acción más probable
            table[i, j] = np.mean(values)

    plt.figure(figsize=(6, 5))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Promedio de probabilidad máxima por celda (política entrenada)")
    plt.xlabel("Eje X del entorno")
    plt.ylabel("Eje Y del entorno")
    plt.show()
    """

    # ---- Prueba visual del agente ----
    test(env, params, STEPS, SIZE, EPISODES=5)


if __name__ == "__main__":
    main()
