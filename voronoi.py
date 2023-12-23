"""
Voronoi diagram

https://en.wikipedia.org/wiki/Voronoi_diagram
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from timeit import default_timer

# For typing
num = int | float


def timer(func: Callable) -> Callable:
    """
    Decorator to time a function
    :param func: Function to be timed
    :return:
    """
    def wrapper(*args, **kwargs):
        s = default_timer()
        rv = func(*args, **kwargs)
        print(f"Time: {func.__name__}: {default_timer() - s:.4f} s")
        return rv
    return wrapper


def generate_seeds(n: int, w: int, h: int) -> np.ndarray:
    """
    Generates an array of seeds with random positions
    :param n:
    :param w:
    :param h:
    :return:
    """
    x_pos = np.random.randint(0, w, n)
    y_pos = np.random.randint(0, h, n)
    return np.vstack((x_pos, y_pos))


def euclidean_dist(x1: num, y1: num, x2: num, y2: num) -> num:
    """
    Euclidean distance between two points
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    return np.power(x2 - x1, 2) + np.power(y2 - y1, 2)
    # return np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))


def manhattan_dist(x1: num, y1: num, x2: num, y2: num) -> num:
    """
    Manhattan distance between two points
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    return np.abs(x1 - x2) + np.abs(y1 - y2)


def render_seeds(img: np.ndarray, seeds: np.ndarray, radius: num,
                 color: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    """
    Marks the seeds into the image
    :param img: Array of pixel coordinates
    :param seeds: Array of x- and y-coordinates for the seeds
    :param radius: Radius of the seeds
    :param color: Color for the seeds
    :return:
    """
    xmin, xmax = 0, img.shape[1]
    ymin, ymax = 0, img.shape[0]
    for x_s, y_s in zip(seeds[0], seeds[1]):
        x1 = max(x_s - radius, xmin)
        x2 = min(x_s + radius, xmax)
        y1 = max(y_s - radius, ymin)
        y2 = min(y_s + radius, ymax)
        for y in range(y1, y2):
            for x in range(x1, x2):
                if euclidean_dist(x, y, x_s, y_s) <= radius * radius:
                    img[y, x] = color

    return img


@timer
def render_voronoi(img: np.ndarray, seeds: np.ndarray, color_palette: np.ndarray,
                   dist_func: Callable = euclidean_dist) -> np.ndarray:
    """
    Renders the Voronoi
    :param img: Array of pixel coordinates
    :param seeds: Array of x- and y-coordinates for the seeds
    :param color_palette: Colors for the different areas of the diagram
    :param dist_func: Function for calculating the distance between the pixels
    :return:
    """
    h, w = img.shape[0], img.shape[1]
    n_colors = color_palette.shape[0]
    for y_p in range(h):
        for x_p in range(w):
            min_ind, min_dist = 0, h * w
            for i, (x_s, y_s) in enumerate(zip(seeds[0], seeds[1])):
                dist = dist_func(x_p, y_p, x_s, y_s)
                if dist <= min_dist:
                    min_ind = i
                    min_dist = dist

            img[y_p, x_p] = np.array(color_palette[min_ind % n_colors]) * (1 / 255)

    return img


def main() -> None:
    n_seeds = 20
    width, height = 800, 600
    seed_radius = 5
    # From Gruvbox theme
    color_palette = np.array([[146, 131, 116], [251, 73, 52], [184, 187, 38],
                              [250, 189, 47], [131, 165, 152], [211, 134, 155],
                              [142, 192, 124], [254, 128, 25]], dtype=object)
    seeds = generate_seeds(n_seeds, width, height)
    img = np.ones((height, width, 3))
    img = render_voronoi(img, seeds, color_palette, euclidean_dist)
    img = render_seeds(img, seeds, seed_radius)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
