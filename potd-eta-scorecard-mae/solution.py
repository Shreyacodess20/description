import numpy as np

def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return np.mean(np.abs(y - yhat))
