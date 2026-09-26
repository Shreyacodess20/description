import numpy as np

def gaussian_elimination_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(float).copy()
    b = b.astype(float).copy()
    n = a.shape[0]

    for k in range(n):
        pivot = np.argmax(np.abs(a[k:, k])) + k
        if pivot != k:
            a[[k, pivot]] = a[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        for i in range(k+1, n):
            factor = a[i, k] / a[k, k]
            a[i, k:] -= factor * a[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(a[i, i+1:], x[i+1:])) / a[i, i]
    return x

def ridge_regression_predict(
    input: np.ndarray, target: np.ndarray, lam: float, queries: np.ndarray
) -> np.ndarray:
    n, d = input.shape
    X_aug = np.hstack([input, np.ones((n, 1))])
    A = X_aug.T @ X_aug + lam * np.eye(d+1)
    b = X_aug.T @ target
    w = gaussian_elimination_solve(A, b)
    Q_aug = np.hstack([queries, np.ones((queries.shape[0], 1))])
    return Q_aug @ w
