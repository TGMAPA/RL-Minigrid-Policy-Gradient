import matplotlib.pyplot as plt
import seaborn as sns
from QL import *

def main():
    SIZE = 10
    ACTIONS = 3
    EPISODES = 5000
    STEPS = 400
    DISCOUNT_FACTOR = 0.98
    LR = 0.03

    env = SimpleEnv(size=SIZE, render_mode=None)
    env = RGBImgObsWrapper(env)

    qTable = init_Q_Table(SIZE, ACTIONS)

    # Train agent
    qTable = train(env, qTable, EPISODES, STEPS, ACTIONS, LR, DISCOUNT_FACTOR)

    # Graph Q-table with Q average per state
    q_values = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            vals = []
            for d in range(4):
                vals.append(np.max(qTable[(i, j, d)]))
            q_values[i, j] = np.mean(vals)

    plt.figure(figsize=(6, 5))
    sns.heatmap(q_values, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Promedio de Q-values por celda (promediando dirección)")
    plt.show()

    # Test agent
    test(env, qTable, STEPS, SIZE, 10)


main()