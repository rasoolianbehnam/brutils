import numpy as np


def ard_regression(X, y, max_iter=int(1e4), alpha_0=None, beta_0=None):
    n = y.shape[0]
    d = X.shape[1]
    alpha = alpha_0 if alpha_0 is not None else np.random.uniform(100, size=d)
    beta = beta_0 if beta_0 is not None else np.random.uniform() * 100
    mu = None
    Sigma = None

    for i in range(max_iter):
        params = np.hstack([alpha, beta])
        A = np.diag(alpha)
        AXt = A @ X.T
        C = X @ AXt + np.eye(n)
        Cinv = np.linalg.inv(C)
        mu = AXt @ Cinv @ y
        Sigma = A - AXt @ Cinv @ AXt.T
        # SigmaInv = np.linalg.inv(Sigma)

        alpha = np.diag(Sigma) + mu ** 2
        beta = n / (np.sum((y - X @ mu) ** 2) + np.einsum("ii", X.T @ X @ Sigma))
        if np.allclose(params, np.hstack([alpha, beta])):
            break

    return alpha, beta, mu, Sigma
