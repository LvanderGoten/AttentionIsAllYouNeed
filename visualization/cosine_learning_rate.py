import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
INITIAL_LR = 0.05
ALPHA = 0.1
DECAY_STEPS = 1000
M_MUL = 2
T_MUL = .9

num_cycles = 10
lrs = np.cumprod([INITIAL_LR] + [T_MUL] * num_cycles)

X = np.empty(shape=[num_cycles + 1, DECAY_STEPS])
Y = np.empty(shape=[num_cycles + 1, DECAY_STEPS])
for i, lr in enumerate(lrs):
    # t axis
    global_step = np.arange(DECAY_STEPS)

    # y axis
    cosine_decay = .5 * (1 + np.cos(np.pi * global_step/DECAY_STEPS))
    decayed = (1 - ALPHA) * cosine_decay + ALPHA
    decayed_learning_rate = lr * decayed

    X[i, :] = global_step + i * DECAY_STEPS
    Y[i, :] = decayed_learning_rate

X = X.reshape(-1)
Y = Y.reshape(-1)

sns.lineplot(x=X, y=Y)
plt.xlabel("Global step")
plt.ylim(0, 1.1 * INITIAL_LR)
plt.ylabel("Decayed learning rate")
plt.title('Cosine decay w. warm restarts')
plt.show()