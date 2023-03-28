"""
Some kind of an effort to create Perlin noise, for now only in 2d

More info is for example in Wikipedia: https://en.wikipedia.org/wiki/Perlin_noise

This implementation however follows more along the lines the one shown in:
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


def _get_values(img: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    :param img:
    :param grid:
    :return:
    """
    rows, cols = img.shape[:2]
    grid_h, grid_w = grid.shape[:2]
    cell_y = rows // grid_h  # Height of a grid cell in pixels
    cell_x = cols // grid_w  # Width of a grid cell in pixels
    for y in range(rows):
        for x in range(cols):
            cell = int(np.floor(y / cell_y)), int(np.floor(x / cell_x))
            value = _interpolate_dots(grid=grid, point=np.array([x, y]), cell=cell,
                                      cell_x=cell_x, cell_y=cell_y)
            img[y, x] = value

    return img


def gen_noise(width: int, height: int, grid_w: int, grid_h: int) -> np.ndarray:
    """
    Generates the image of Perlin noise from the given image with the normalized
    gradient vectors
    :param width: Width of the image in pixels
    :param height: Height of the image in pixels
    :param grid_w: The amount of grid cells along the x-axis
    :param grid_h: The amount of grid cells along the y-axis
    :return:
    """
    # Assert the grid divides the image evenly
    if width % grid_w != 0:
        msg = 'The number of grid cells does not divide the width of the ' \
              'image evenly'
        raise ValueError(msg)
    if height % grid_h != 0:
        msg = 'The number of grid cells does not divide the height of the ' \
              'image evenly'
        raise ValueError(msg)

    # Initialise a blank image with one color channel
    img = np.zeros(shape=(height, width))
    # Create the 2d grid with (normalised) gradient vectors at each cell corner
    grid = np.random.random(size=(grid_h, grid_w, 2)) * 2 - 1
    grid = grid / np.linalg.norm(grid[:, :], axis=2, keepdims=True)
    return _get_values(img=img, grid=grid)


def main():
    width, height = 256, 256  # Size of the image in pixels
    grid_w, grid_h = 32, 32  # Amount of grid cells in both direction
    cmap = 'gray'
    img = gen_noise(width=width, height=height, grid_w=grid_w, grid_h=grid_h)
    _, ax = plt.subplots()
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
