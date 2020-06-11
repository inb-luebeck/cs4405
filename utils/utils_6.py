import numpy as np
from matplotlib import pyplot as plt

def load_data(filepath):
    with np.load(filepath) as f:
        patterns = f['patterns']
        return (patterns)


def add_noise(patterns, noise):
    patterns_noise = []
    for pattern in patterns:
        mask = np.random.uniform(size=pattern.shape) < noise
        pattern_noise = pattern.copy()
        pattern_noise[mask] = -pattern_noise[mask]
        patterns_noise.append(pattern_noise)
    return np.stack(patterns_noise)


def plot(patterns, patterns_noise, patterns_fixpoint):

    n_patterns = len(patterns)

    fig, axes = plt.subplots(nrows=n_patterns,
                             ncols=3,
                             figsize=(6, 2*n_patterns))

    axes[0, 0].set_title('Original Pattern')
    axes[0, 1].set_title('Pattern with Noise')
    axes[0, 2].set_title('Hopfield Fixpoint')

    for i in range(n_patterns):
        axes[i, 0].imshow(patterns[i])
        axes[i, 1].imshow(patterns_noise[i])
        axes[i, 2].imshow(patterns_fixpoint[i])

    for axis in axes.flat:
        axis.set_axis_off()
