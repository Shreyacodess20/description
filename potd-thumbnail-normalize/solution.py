import numpy as np

def normalize_image(matrix: np.ndarray) -> tuple[float, float, np.ndarray]:
    mu = matrix.mean()
    sigma = matrix.std()
    if sigma == 0.0:
        normalized = np.zeros_like(matrix, dtype=float)
    else:
        normalized = (matrix - mu) / sigma
    return mu, sigma, normalized
