import numpy as np
from matplotlib import pyplot as plt

def load_data(filepath):
    with np.load(filepath) as f:
        hidden_weights = f['hidden_weights']
        output_weights = f['output_weights']
        return (hidden_weights, output_weights)

def plot_data(samples, labels):
    samples_pos = samples[labels == 1]
    samples_neg = samples[labels == -1]    
    plt.plot(samples_pos[:,0], samples_pos[:, 1], 'b.')
    plt.plot(samples_neg[:,0], samples_neg[:, 1], 'r.')
    plt.axis('scaled')

def plot_classlines(weights):
    for weight in weights:
        plot_classline(weight[1:], weight[0])

def plot_classline(weights, threshold):
    assert weights.any(), "Weights must not be the zero vector."
    ax = plt.gca()
    x_min, x_max = ax.get_xbound()
    y_min, y_max = ax.get_ybound()
    if weights[1] == 0:
        x_min = threshold / weights[0]
        x_max = x_min
    else:
        y_min = (threshold - weights[0] * x_min) / weights[1]
        y_max = (threshold - weights[0] * x_max) / weights[1]
    plt.plot([x_min, x_max], [y_min, y_max], lw=3)
