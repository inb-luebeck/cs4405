import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def load_data(filepath):
    with np.load(filepath) as f:
        samples = f['samples']
        labels = f['labels']
        return samples, labels

def activation_function(x, beta):
    return np.tanh(0.5 * beta * x)

def extend(tensor):
    if tensor.ndim == 1:
        return np.insert(tensor, 0, -1)
    else:
        thresholds = -np.ones(len(tensor))
        return np.column_stack((thresholds, tensor))

def classify_mlp(samples, hidden_weights, output_weights, beta):
    samples_extended = extend(samples)
    hidden_outputs = activation_function(np.matmul(samples_extended, hidden_weights.T), beta)
    hidden_outputs_extended = extend(hidden_outputs)
    outputs = activation_function(np.matmul(hidden_outputs_extended, output_weights.T), beta)    
    return outputs, hidden_outputs

def plot_data(samples, labels):
    samples_pos = samples[labels == 1]
    samples_neg = samples[labels == -1]    
    plt.plot(samples_pos[:, 0], samples_pos[:, 1], 'b.')
    plt.plot(samples_neg[:, 0], samples_neg[:, 1], 'r.')
    plt.axis('scaled')

class Animation:
    def __init__(self, samples, labels, hidden_neurons):
        self.samples = samples
        self.labels = labels
        self.hidden_neurons = hidden_neurons
        self.fig, self.ax = plt.subplots()
        plt.close()

    def get_classlines(self, weights):
        classlines = [self.get_classline(weight[1:], weight[0])
                      for weight in weights]
        return classlines

    def get_classline(self, weights, threshold):
        assert weights.any(), "Weights must not be the zero vector."
        x_min, x_max = self.ax.get_xbound()
        y_min, y_max = self.ax.get_ybound()
        if weights[1] == 0:
            x_min = threshold / weights[0]
            x_max = x_min
        else:
            y_min = (threshold - weights[0] * x_min) / weights[1]
            y_max = (threshold - weights[0] * x_max) / weights[1]
        return ([x_min, x_max], [y_min, y_max])

    def init_func(self):
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.axes.set_aspect('equal')
        self.samples_pos = self.ax.plot([], [], 'b.')[0]
        self.samples_neg = self.ax.plot([], [], 'r.')[0]
        self.classlines = [self.ax.plot([], [], 'g')[0] for _ in range(self.hidden_neurons)]
        return [self.samples_pos, self.samples_neg, *self.classlines]

    def func(self, data):
        hidden_weights = data['hidden_weights']
        output_weights = data['output_weights']        
        classifications, _ = classify_mlp(self.samples, hidden_weights, output_weights, beta=2)
        classifications = 2 * (classifications >= 0) - 1;
        samples_pos = self.samples[classifications == 1]
        samples_neg = self.samples[classifications == -1]
        self.samples_pos.set_data(samples_pos[:, 0], samples_pos[:, 1])
        self.samples_neg.set_data(samples_neg[:, 0], samples_neg[:, 1])
        classlines = self.get_classlines(hidden_weights)
        for idx, classline in enumerate(classlines):
            self.classlines[idx].set_data(classline)
        return [self.samples_pos, self.samples_neg, *self.classlines]

    def animate(self, generator, max_frames):
        anim = animation.FuncAnimation(fig=self.fig, 
                                       func=self.func,
                                       frames=generator,
                                       init_func=self.init_func,
                                       blit=True,
                                       save_count=max_frames)
        return HTML(anim.to_jshtml(default_mode='once'))
