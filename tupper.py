"""
Tupper's self-referential formula

Sauce: https://en.wikipedia.org/wiki/Tupper%27s_self-referential_formula
"""

import math
import decimal
import traceback
import numpy as np
import matplotlib.pyplot as plt

from decimal import Decimal


def read_number(fname: str) -> int:
    """
    Reads the number from a text file
    :param fname:
    :return:
    """
    with open(fname, 'r') as file:
        return int(file.read())


def tupper_fun(x: int, y: int) -> bool:
    y_d = Decimal(y)
    try:
        return (1 / 2) < math.floor((y_d // Decimal(17) * Decimal(2)
                                     ** (Decimal(-17) * math.floor(x)
                                         - math.floor(y_d) % 17)) % 2)
    except decimal.DecimalException:
        with decimal.localcontext() as ctx:
            ctx.prec = len(y_d.as_tuple().digits) + 2
            try:
                return (1 / 2) < math.floor((y_d // Decimal(17) * Decimal(2)
                                             ** (Decimal(-17) * math.floor(x)
                                                 - math.floor(y_d) % 17)) % 2)
            except decimal.DecimalException:
                traceback.print_exc()


def color_img(img: np.ndarray, tupper_arr: list[list], color: tuple, grid_x: int,
              grid_y: int, margin: int) -> np.ndarray:
    """
    Updates image according to the given Tupper boolian array
    :param img:
    :param tupper_arr:
    :param color:
    :param grid_x:
    :param grid_y:
    :param margin:
    :return:
    """
    h, w = img.shape[0:2]
    ind_x = grid_x
    for col in tupper_arr[::-1]:
        ind_y = h - grid_y * margin
        for pixel in col[::-1]:
            if pixel:
                inc_x = ind_x + grid_x
                inc_y = ind_y + grid_y
                img[ind_y:inc_y, ind_x:inc_x] = color

            ind_y -= grid_y
        ind_x += grid_x
    return img


def main() -> None:
    # Define some constants
    x_range, y_range = 106, 17
    margin = 2
    grid_size_x, grid_size_y = 12, 18
    c_channels = 3
    color = (255, 0, 255)

    # Define a blank image
    width = (x_range + margin) * grid_size_x
    height = (y_range + margin) * grid_size_y
    img = np.zeros((height, width, c_channels))

    # Read the number from the file
    filename = 'tupper_big_af_k.txt'
    big_af_k = read_number(filename)

    # Apply the function
    x_vals = np.arange(0, x_range, 1)
    y_vals = np.arange(big_af_k, big_af_k + 17, 1)
    tupper_bool = [[tupper_fun(i, j) for j in y_vals]
                   for i in x_vals]

    # Color the image
    img = color_img(img, tupper_bool, color, grid_size_x, grid_size_y, margin)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
