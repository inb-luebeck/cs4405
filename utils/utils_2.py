import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from IPython.display import HTML

def load_data(filepath):
    with np.load(filepath) as f:
        samples = f['samples']
        labels = f['labels']
        return (samples, labels)

def intersection(x_min, x_max, y_min, y_max, weights, threshold):
    weight_x, weight_y = weights
    if weight_y == 0:
        x_min = threshold / weight_x
        x_max = x_min
    else:
        y_min = (threshold - weight_x * x_min) / weight_y
        y_max = (threshold - weight_x * x_max) / weight_y
    return x_min, x_max, y_min, y_max

def counter_clockwise_sort(points):
    mean = np.mean(points, axis=0)
    points_mean_free = points - mean
    angles = np.arctan2(points_mean_free[:, 0],
                        points_mean_free[:, 1])
    return points[np.argsort(angles), :]
    
def neuron_classify(samples, weights, threshold):
    classifications = np.matmul(samples, weights) - threshold
    classifications = (classifications >= 0) * 2 - 1
    return classifications

class Animation:
    
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.fig, self.ax = plt.subplots()
        plt.close()

    def init_func(self):
        samples_pos = self.samples[self.labels == 1]
        samples_neg = self.samples[self.labels == -1]
        self.ax.plot(samples_pos[:, 0], samples_pos[:, 1],
                     color='blue',
                     linestyle='',
                     marker='.')
        self.ax.plot(samples_neg[:, 0], samples_neg[:, 1],
                     color='red',
                     linestyle='',
                     marker='.')
        self.ax.axis('scaled')
        self.classline, = self.ax.plot([], [], 'k-')
        self.sample, = self.ax.plot([], [],
                                    color='green',
                                    linestyle='',
                                    marker='o',
                                    fillstyle='none')
        self.area_pos = Polygon(np.empty(shape=(0, 2)),
                                color='blue',
                                alpha=0.1)
        self.area_neg = Polygon(np.empty(shape=(0, 2)),
                                color='red',
                                alpha=0.1)
        self.ax.add_patch(self.area_pos)
        self.ax.add_patch(self.area_neg)
        return [self.classline, self.sample, self.area_pos, self.area_neg]

    def func(self, data):
        threshold = data['threshold']
        weights = data['weights']
        samples = data['samples']
        x_min, x_max = self.ax.get_xbound()
        y_min, y_max = self.ax.get_ybound()
        x_1, x_2, y_1, y_2 = intersection(x_min, x_max, y_min, y_max, weights, threshold)
        self.classline.set_data([x_1, x_2], [y_1, y_2])
        corners = np.array([[x_min, y_min],
                            [x_min, y_max],
                            [x_max, y_max],
                            [x_max, y_min]])
        classifications = neuron_classify(corners, weights, threshold)
        
        corners_pos = corners[classifications == 1]
        points_pos = np.concatenate([corners_pos, [[x_1, y_1], [x_2, y_2]]])
        points_pos = counter_clockwise_sort(points_pos)
        self.area_pos.set_xy(points_pos)

        corners_neg = corners[classifications == -1]
        points_neg = np.concatenate([corners_neg, [[x_1, y_1], [x_2, y_2]]])
        points_neg = counter_clockwise_sort(points_neg)
        self.area_neg.set_xy(points_neg)

        self.sample.set_data(np.stack(samples, axis=1)[-2:])
        return [self.classline, self.sample, self.area_pos, self.area_neg]

    def animate(self, generator, max_frames):
        anim = animation.FuncAnimation(fig=self.fig, 
                                       func=self.func,
                                       frames=generator,
                                       init_func=self.init_func,
                                       blit=True,
                                       save_count=max_frames)
        return HTML(anim.to_jshtml(default_mode='once'))
