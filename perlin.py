"""
Some kind of an effort to create Perlin noise, for now only in 2d

More info is for example in Wikipedia: https://en.wikipedia.org/wiki/Perlin_noise

This implementation however follows more along the lines of the one shown in:
https://www.cs.umd.edu/class/fall2018/cmsc425/Lects/lect14-perlin.pdf
"""

import numpy as np
import matplotlib.pyplot as plt

# Shortcut for typing
numeric = int | float


def _offsets_and_grads(grid: np.ndarray, point: np.ndarray, cell: tuple,
                       cell_x: int, cell_y: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the four offset vectors from the point to the cell's corners, and
    the gradient vectors located at the corners
    :param grid:
    :param point:
    :param cell:
    :param cell_x:
    :param cell_y:
    :return:
    """
    r, c = cell
    offsets = np.zeros(shape=(4, 2))
    grads = np.zeros(shape=(4, 2))
    ind = 0
    for j in range(r, r + 2):
        for i in range(c, c + 2):
            offset = point - np.array([i * cell_x, j * cell_y])
            offsets[ind] = offset
            grads[ind] = grid[j - 1, i - 1]
            ind += 1

    return offsets, grads


def _fade(t: numeric) -> numeric:
    """
    From https://en.wikipedia.org/wiki/Smoothstep (also shown in
    https://www.cs.umd.edu/class/fall2018/cmsc425/Lects/lect14-perlin.pdf)
    :param t:
    :return:
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def _jointfade(tx: numeric, ty: numeric) -> numeric:
    """
    Combines the calls of fade(x) and fade(y) via multiplication
    :param tx:
    :param ty:
    :return:
    """
    if not (0 <= tx <= 1):
        raise ValueError('Too large fading factor in the x-axis')
    if not (0 <= ty <= 1):
        raise ValueError('Too large fading factor in the y-axis')
    return _fade(tx) * _fade(ty)


def _interpolate_dots(grid: np.ndarray, point: np.ndarray, cell: tuple, cell_x: int,
                      cell_y: int) -> numeric:
    """
    Interpolates between the dot products of the four cell corners closest to
    the point
    :param point:
    :param cell:
    :param cell_x:
    :param cell_y:
    :return:
    """
    offsets, grads = _offsets_and_grads(grid=grid, point=point, cell=cell,
                                        cell_x=cell_x, cell_y=cell_y)
    wx = (point[0] - cell[1] * cell_x) / cell_x
    wy = (point[1] - cell[0] * cell_y) / cell_y
    top_left_dot = float(np.dot(offsets[0], grads[0]))
    top_right_dot = float(np.dot(offsets[1], grads[1]))
    bottom_left_dot = float(np.dot(offsets[2], grads[2]))
    bottom_right_dot = float(np.dot(offsets[3], grads[3]))
    top_left = _jointfade(tx=(1 - wx), ty=(1 - wy)) * top_left_dot
    top_right = _jointfade(tx=wx, ty=(1 - wy)) * top_right_dot
    bottom_left = _jointfade(tx=(1 - wx), ty=wy) * bottom_left_dot
    bottom_right = _jointfade(tx=wx, ty=wy) * bottom_right_dot
    return top_left + top_right + bottom_left + bottom_right


def _get_pixel_values(w: int, h: int, grid: np.ndarray) -> np.ndarray:
    """
    :param w: Width of the image in pixels
    :param h: Height of the image in pixels
    :param grid:
    :return:
    """
    img = np.zeros(shape=(h, w))
    grid_h, grid_w = grid.shape[:2]
    cell_y = h // grid_h  # Height of a grid cell in pixels
    cell_x = w // grid_w  # Width of a grid cell in pixels
    for y in range(h):
        for x in range(w):
            cell = int(np.floor(y / cell_y)), int(np.floor(x / cell_x))
            value = _interpolate_dots(grid=grid, point=np.array([x, y]), cell=cell,
                                      cell_x=cell_x, cell_y=cell_y)
            img[y, x] = value

    return img


def noise(width: int, height: int, octaves: int = 1, persistence: numeric = 1) \
        -> np.ndarray:
    """
    Generates the image of Perlin noise from the given image with the normalized
    gradient vectors
    :param width: Width of the image in pixels
    :param height: Height of the image in pixels
    :param octaves: The amount of octaves (maximum value is 8). Defaults to 1.
    :param persistence: An optional scaling factor on the range (0, 1) used to scale
        (dampen) the higher octaves. Defaults to 1. Is only applied if octaves > 1.
    :return:
    """
    # Assert that the number of octaves isn't too high
    limit = np.power(2, octaves)
    if limit > width or limit > height:
        msg = f'The number of octaves is too large. 2^octaves must be smaller than' \
              f'the width and height of the image.'
        raise ValueError(msg)

    grid_w, grid_h = 2, 2  # Amount of grid cells in both direction
    if octaves == 1:
        grid = np.random.random(size=(grid_h, grid_w, 2)) * 2 - 1
        grid = grid / np.linalg.norm(grid[:, :], axis=2, keepdims=True)
        return _get_pixel_values(w=width, h=height, grid=grid)
    img = np.zeros(shape=(height, width))
    for o in range(octaves):
        f = np.power(2, o)
        p = np.power(persistence, o)
        grid = np.random.random(size=(grid_h * f, grid_w * f, 2)) * 2 - 1
        grid = grid / np.linalg.norm(grid[:, :], axis=2, keepdims=True)
        img += p * _get_pixel_values(w=width, h=height, grid=grid)
    return img


def main():
    width, height = 256, 256  # Size of the image in pixels
    octaves, persistence = 8, 0.5
    np.random.seed(69420)
    cmap = 'jet'
    img = noise(width=width, height=height, octaves=octaves, persistence=persistence)
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
