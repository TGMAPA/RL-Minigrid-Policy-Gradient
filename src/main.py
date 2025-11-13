# Importación de librerias y modulos
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PolicyGradient import *  
from minigrid.wrappers import RGBImgObsWrapper

def main():
    # Tamaño del entorno
    SIZE = 10         
    # Número de acciones posibles     
    ACTIONS = 3           
    # Número de episodios para el entrenamiento
    EPISODES = 1500    
    # Número de pasos por episodio
    STEPS = 400          
    # Learning rate
    LR = 0.05         
    # Factor para el descuento 
    DISCOUNT_FACTOR = 0.98

    # Instancia del entorno
    env = SimpleEnv(size=SIZE, render_mode=None)
    env = RGBImgObsWrapper(env)

    # Inicializar parametros
    policy = init_params(SIZE, ACTIONS)

    # Entrenar agente con REINFORCE
    policy, rewards_per_episode = train(env, policy, EPISODES, STEPS, LR, DISCOUNT_FACTOR)

    # Mostrar evolución de las recompensas por episodio
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=np.arange(EPISODES), y=rewards_per_episode)
    plt.title("Recompensa total por episodio")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa total")
    plt.grid(True)
    plt.show()

    # Probar el aprendizaje del agente por 10 episodios
    test(env, policy, STEPS, SIZE, EPISODES=10)

main()
