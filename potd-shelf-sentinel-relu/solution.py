import numpy as np


def relu_forward(x: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward and backward pass of ReLU.

    x: raw sensor readings, shape (n,).
    g: the upstream gradient from the next layer, shape (n,).

    Return (activated, grad):
      activated = max(0, x), elementwise.
      grad = g * 1[x > 0], elementwise (strict >, not >=).

    An all-negative x gives an all-zero activated and an all-zero grad.
    """
    mask = x > 0
    activated = x * mask
    grad = g * mask
    return activated, grad
