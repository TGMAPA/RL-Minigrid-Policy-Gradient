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

    # ---- Prueba visual del agente ----
    test(env, params, STEPS, SIZE, EPISODES=5)


if __name__ == "__main__":
    main()
