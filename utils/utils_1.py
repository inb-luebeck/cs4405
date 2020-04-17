import numpy as np

def load_data(filepath):
    with np.load(filepath) as f:
        samples = f['samples']
        labels = f['labels']
        return (samples, labels)
