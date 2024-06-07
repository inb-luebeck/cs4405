import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def load_data(filepath):
    with np.load(filepath) as f:
        samples = f['samples']
        return samples

class Animation:
    def __init__(self, samples, k):
        self.samples = samples
        self.k = k
        self.fig, self.ax = plt.subplots()
        plt.close()

    def init_func(self):
        self.ax.plot(self.samples[:, 0], self.samples[:, 1],
                     marker='.',
                     color='None')
        self.ax.axis('scaled')
        color_map = plt.get_cmap('Set1')
        self.centroids = [self.ax.plot([], [],
                                       marker='o',
                                       linestyle='',
                                       markeredgecolor='k',
                                       zorder=10,
                                       color=color_map(i))[0] for i in range(self.k)]
        self.clusters = [self.ax.plot([], [],
                                      marker='.',
                                      linestyle='',
                                      color=color_map(i))[0] for i in range(self.k)]
        return [*self.clusters, *self.centroids]

    def func(self, data):
        codebook_vectors = data['codebook_vectors']
        distances = euclidean_distances(self.samples, codebook_vectors)
        indexes = np.argmin(distances,
                            axis=1)
        for index in range(self.k):
            mask = (indexes == index)
            self.clusters[index].set_data(self.samples[mask, 0], self.samples[mask, 1])
            self.centroids[index].set_data(codebook_vectors[index, 0], codebook_vectors[index, 1])

        return [*self.clusters, *self.centroids]

    def animate(self, generator, max_frames):
        anim = animation.FuncAnimation(fig=self.fig, 
                                       func=self.func,
                                       frames=generator,
                                       init_func=self.init_func,
                                       blit=True,
                                       save_count=max_frames)
        return HTML(anim.to_jshtml(default_mode='once'))
