import numpy as np
from matplotlib import pyplot as plt

def load_data(filepath, grayscale=False):
    with np.load(filepath) as f:
        image_1 = f['image_1']
        image_2 = f['image_2']
        if grayscale:
            image_1 = rgb_to_grayscale(image_1)
            image_2 = rgb_to_grayscale(image_2)
        return image_1, image_2

def rgb_to_grayscale(image):
    return np.dot(image, [0.2989, 0.5870, 0.1140])

def mix_images(image_1, image_2, coefficient=0.4):
    mixture_1 = coefficient * image_1 + (1 - coefficient) * image_2
    mixture_2 = coefficient * image_2 + (1 - coefficient) * image_1
    return np.stack([mixture_1, mixture_2],
                    axis=-1)

def show_image(image):
    n_subplots = image.shape[-1]
    fig, axes = plt.subplots(nrows=1,
                             ncols=n_subplots,
                             figsize=(n_subplots * 4, 4))
    for i, axis in enumerate(axes):
        axis.imshow(image[..., i],
                    cmap='gray')
        axis.axis('off')

def rotation_matrix(theta):
    theta = np.radians(theta)
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array(((cos, -sin),
                     (sin, cos)))

def kurtosis(x, axis=(0, 1)):
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    z = (x - mean) / std
    return np.mean(z ** 4, axis=axis)
