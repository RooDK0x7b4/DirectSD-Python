"""
Linear systems solver for DirectSD.

Port of MATLAB linsys.m: solves A*X = B using QR or SVD with optional
iterative refinement.
"""

import numpy as np


def linsys(A, B, method='qr', refine=False):
    """
    Solve the linear system A * X = B.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    B : np.ndarray, shape (m, k) or (m,)
    method : str
        'qr'  - QR decomposition (default)
        'svd' - SVD decomposition
    refine : bool
        If True, apply one step of iterative refinement.

    Returns
    -------
    X : np.ndarray or None
        Solution if found, else None.
    """
    A = np.atleast_2d(np.array(A, dtype=float))
    B = np.atleast_1d(np.array(B, dtype=float))
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    tol = np.sqrt(np.finfo(float).eps)
    m, n = A.shape

    if method == 'svd':
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        rank = np.sum(s > tol * s[0])
        if rank == 0:
            return None
        # Pseudo-inverse solution
        s_inv = np.where(s > tol * s[0], 1.0 / s, 0.0)
        X = Vt[:rank].T @ np.diag(s_inv[:rank]) @ U[:, :rank].T @ B

    elif method == 'qr':
        if m >= n:
            Q, R = np.linalg.qr(A)
            rank = np.sum(np.abs(np.diag(R)) > tol * max(np.abs(np.diag(R)).max(), 1.0))
            if rank < n:
                # Rank-deficient: fall back to SVD
                return linsys(A, B, method='svd', refine=refine)
            X = np.linalg.solve(R[:n, :n], Q[:, :n].T @ B)
        else:
            X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    else:
        raise ValueError(f"Unknown method '{method}'")

    if refine and X is not None:
        # Iterative refinement step
        r = B - A @ X
        dX, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
        X = X + dX

    return X
