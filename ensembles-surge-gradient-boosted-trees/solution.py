import numpy as np

def build_regression_tree(
    X: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    lam: float,
    gamma: float,
    max_depth: int,
    depth: int = 0,
) -> dict:
    G, H = np.sum(g), np.sum(h)
    if depth == max_depth or len(X) < 2:
        return {"leaf": True, "weight": -G / (H + lam)}

    best_gain = -np.inf
    best_split = None
    n, d = X.shape
    for j in range(d):
        vals = np.unique(X[:, j])
        if len(vals) < 2:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2
        for t in thresholds:
            left = X[:, j] <= t
            right = ~left
            if not np.any(left) or not np.any(right):
                continue
            GL, HL = np.sum(g[left]), np.sum(h[left])
            GR, HR = np.sum(g[right]), np.sum(h[right])
            gain = 0.5 * (GL**2/(HL+lam) + GR**2/(HR+lam) - G**2/(H+lam)) - gamma
            if gain > best_gain or (gain == best_gain and (best_split is None or j < best_split[0] or (j == best_split[0] and t < best_split[1]))):
                best_gain = gain
                best_split = (j, t)
    if best_split is None or best_gain <= 0:
        return {"leaf": True, "weight": -G / (H + lam)}

    j, t = best_split
    left_mask = X[:, j] <= t
    right_mask = ~left_mask
    left_tree = build_regression_tree(X[left_mask], g[left_mask], h[left_mask], lam, gamma, max_depth, depth+1)
    right_tree = build_regression_tree(X[right_mask], g[right_mask], h[right_mask], lam, gamma, max_depth, depth+1)
    return {"leaf": False, "feature": j, "threshold": t, "left": left_tree, "right": right_tree}

def gbdt_predict(
    X: np.ndarray,
    y: np.ndarray,
    T: int,
    lam: float,
    gamma: float,
    eta: float,
    max_depth: int,
    base_score: float,
    queries: np.ndarray,
) -> np.ndarray:
    n = len(y)
    preds = np.full(n, base_score)
    trees = []
    for _ in range(T):
        g = preds - y
        h = np.ones_like(y)
        tree = build_regression_tree(X, g, h, lam, gamma, max_depth)
        trees.append(tree)
        for i in range(n):
            node = tree
            while not node["leaf"]:
                if X[i, node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            preds[i] += eta * node["weight"]

    results = []
    for q in queries:
        total = 0.0
        for tree in trees:
            node = tree
            while not node["leaf"]:
                if q[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            total += node["weight"]
        results.append(base_score + eta * total)
    return np.array(results)
