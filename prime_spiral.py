"""
Prime spiral in polar coordinates
"""

import math
import matplotlib.pyplot as plt


def create_primes(n: int) -> list:
    """
    Creates primes up to n with the Sieve of Eratosthenes
    :param n:
    :return:
    """
    prime_bool = [True for _ in range(n + 1)]
    p = 2
    while p * p <= n:
        if prime_bool[p]:
            for i in range(p * 2, n + 1, p):
                prime_bool[i] = False

        p += 1
    prime_bool[0] = False
    prime_bool[1] = False
    prime_lst = [p for p in range(n + 1) if prime_bool[p]]
    return prime_lst


def polar2xy(r: int | float, omega: int | float) -> tuple:
    """
    Converts polar coordinates (radius and angle) into x and y coordinates
    :param r: Radius or distance from the origo
    :param omega: Angle between the radius and the positive
        x-axis
    :return:
    """
    x = r * math.cos(omega)
    y = r * math.sin(omega)
    return x, y


def spiral(n: int) -> None:
    """
    Creates and plots the spiral of prime numbers in spherical
    coordinates, using prime numbers up to n
    :param n:
    :return:
    """
    primes = create_primes(n)
    coords = [polar2xy(p, p) for p in primes]
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    plt.scatter(x, y, s=3, c='blue')
    plt.grid()
    plt.show()


def main():
    n = 10000
    spiral(n)


if __name__ == '__main__':
    main()
