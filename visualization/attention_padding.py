import numpy as np
import tensorflow as tf; tf.enable_eager_execution()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def _plot_cube(voxels, colors=None, ax=None, fontsize=20):
    if colors is not None:
        colors = np.where(colors, "gray", "red")
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
    ax.set_aspect("equal")
    ax.invert_zaxis()
    ax.voxels(voxels, facecolors=colors, edgecolors="k")
    ax.set_xticks([])
    # ax.set_xticks(range(1, voxels.shape[0] + 1))
    ax.set_yticks([])
    #  ax.set_yticks(range(1, voxels.shape[1] + 1))
    ax.set_zticks([])
    # ax.set_zticks(range(1, voxels.shape[2] + 1))
    return ax


B = 4
T = 4
d = 10
t = [1, 2, 3, 4]
S = tf.constant(np.random.randn(B, T, d))
W = tf.nn.softmax(tf.matmul(S, tf.transpose(S, perm=[0, 2, 1])), axis=2)  # [B, T, T]

# Padding mask
T_mask = tf.sequence_mask(lengths=t, maxlen=T, dtype=tf.float64)  # [B, T]
T_mask = tf.matmul(tf.expand_dims(T_mask, axis=2), tf.expand_dims(T_mask, axis=1))      # [B, T, T]
W_masked = tf.multiply(T_mask, W)

W = tf.transpose(W > 0, perm=[1, 2, 0]).numpy()
W_masked = tf.transpose(W_masked > 0, perm=[1, 2, 0]).numpy()

fig = plt.figure(figsize=(10, 10))
for b in range(B):
    W_slice = np.copy(W)
    W_slice[:, :, :b] = False
    ax = fig.add_subplot(1, B, b + 1, projection='3d')
    ax = _plot_cube(voxels=W_slice, colors=W_masked, ax=ax)
plt.savefig("attention_sliced_batch.png", transparent=True, bbox_inches="tight")
