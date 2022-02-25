from typing import Tuple
from numpy import ndarray
import numpy as np


def _d_mse_d_a(x: ndarray, y: ndarray, predictions: ndarray) -> float:
    n: float = float(len(x))
    return (-2 / n) * np.sum((y - predictions) * (x ** 2))


def _d_mse_d_b(x: ndarray, y: ndarray, predictions: ndarray) -> float:
    n: float = float(len(x))
    return (-2 / n) * np.sum((y - predictions) * x)


def _d_mse_d_c(x: ndarray, y: ndarray, predictions: ndarray) -> float:
    n: float = float(len(x))
    return (-2 / n) * np.sum(y - predictions)


def qm(x: ndarray, y: ndarray, learning_rate: float = 0.001, epochs: int = 1000) -> Tuple[float, float, float]:
    """quadratic regression with gradient descent

    Args:
        x (ndarray): given x values
        y (ndarray): known y values

    Returns:
        Tuple[float, float, float]: a, b, c for some fn ax^2 + bx + c
    """
    a: float = 0
    b: float = 0
    c: float = 0

    for _ in range(epochs):
        predictions: ndarray = (a * (x ** 2)) + (b * x) + c
        a -= (learning_rate * _d_mse_d_a(x, y, predictions))
        b -= (learning_rate * _d_mse_d_b(x, y, predictions))
        c -= (learning_rate * _d_mse_d_c(x, y, predictions))

    return (a, b, c)
