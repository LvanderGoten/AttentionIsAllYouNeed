import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np

T, D = 100, 200
pos, i = np.meshgrid(np.arange(T), np.arange(D))
y_sin = np.sin(np.divide(pos, np.power(10000, np.divide(i, D))))
y_cos = np.cos(np.divide(pos, np.power(10000, np.divide(i, D))))
y_cos = np.roll(y_cos, shift=1, axis=0)
ix = np.mod(np.arange(D), 2) == 0
ix = np.tile(np.expand_dims(ix, axis=1), reps=[1, T])
y = np.where(ix, y_sin, y_cos)

fig, ax = plt.subplots(figsize=(10, 10), ncols=1, nrows=3)
ax[0].set_title("Sine function")
sns.heatmap(data=y_sin, vmin=-1, vmax=1, square=False, xticklabels=[], yticklabels=[], cbar=False, ax=ax[0])
ax[1].set_title("Cosine function")
sns.heatmap(data=y_cos, vmin=-1, vmax=1, square=False, xticklabels=[], yticklabels=[], cbar=False, ax=ax[1])
ax[2].set_title("Juxtaposed function")
sns.heatmap(data=y, vmin=-1, vmax=1, square=False, xticklabels=[], yticklabels=[], cbar=False, ax=ax[2])
plt.savefig("positional_embedding.png", transparent=True, bbox_inches="tight")