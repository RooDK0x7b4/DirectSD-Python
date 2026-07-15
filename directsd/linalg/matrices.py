"""
Linear algebra utilities for DirectSD.

Ports of: toep (Toeplitz matrix), hank (Hankel matrix), linsys utilities.
"""

import numpy as np


def toep(a, r, c):
    """
    Build a Toeplitz matrix from polynomial coefficients.

    The matrix has the structure:
        | A(0)   0    0   ... |
        | A(1)  A(0)  0   ... |
        |  ...  A(1) A(0) ... |

    where A(0) is the leading (highest-degree) coefficient.

    Parameters
    ----------
    a : array-like or Poln
        Polynomial coefficients (highest degree first), or Poln object.
    r : int
        Number of rows.
    c : int
        Number of columns.

    Returns
    -------
    T : np.ndarray, shape (r, c)
    """
    from directsd.polynomial.poln import Poln

    if isinstance(a, Poln):
        a = a.coef

    a = np.atleast_1d(np.array(a, dtype=complex)).ravel()
    # MATLAB toep does fliplr(a) before filling: constant term goes at top of each column.
    a_asc = a[::-1]  # ascending (constant first), matches MATLAB's fliplr convention

    T = np.zeros((r, c), dtype=complex)
    n = len(a_asc)
    for j in range(c):
        for i in range(r):
            k = i - j
            if 0 <= k < n:
                T[i, j] = a_asc[k]

    return np.real_if_close(T, tol=1e6)


def hank(a, r, c):
    """
    Build a Hankel matrix from polynomial coefficients.

    Parameters
    ----------
    a : array-like
        Polynomial coefficients.
    r : int
        Number of rows.
    c : int
        Number of columns.

    Returns
    -------
    H : np.ndarray, shape (r, c)
    """
    a = np.atleast_1d(np.array(a, dtype=complex)).ravel()
    H = np.zeros((r, c), dtype=complex)
    for i in range(r):
        for j in range(c):
            k = i + j
            if k < len(a):
                H[i, j] = a[k]
    return np.real_if_close(H, tol=1e6)


def givens(a, b):
    """
    Compute Givens rotation parameters (c, s) such that
    [c  s] [a]   [r]
    [-s c] [b] = [0]
    """
    if b == 0:
        return 1.0, 0.0
    elif abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1 + tau ** 2)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1 + tau ** 2)
        s = c * tau
    return c, s


def house(x):
    """
    Compute Householder vector v and scalar beta such that
    (I - beta*v*v') * x = ||x|| * e1.
    """
    x = np.array(x, dtype=float).ravel()
    sigma = np.dot(x[1:], x[1:])
    v = x.copy()
    v[0] = 1.0
    if sigma == 0 and x[0] >= 0:
        beta = 0.0
    else:
        mu = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma / (x[0] + mu)
        beta = 2 * v[0] ** 2 / (sigma + v[0] ** 2)
        v = v / v[0]
    return v, beta


def schur_ordered(A, stable='lhp'):
    """
    Compute ordered Schur decomposition: T, Z such that A = Z T Z^H
    with stable eigenvalues first.

    Parameters
    ----------
    stable : str
        'lhp' for left-half-plane (continuous), 'unit' for inside unit circle (discrete).
    """
    import scipy.linalg as la

    if stable == 'lhp':
        sort_fn = lambda x: np.real(x) < 0
    else:
        sort_fn = lambda x: np.abs(x) < 1

    T, Z, _ = la.schur(A, output='complex', sort=sort_fn)
    return T, Z


def lyap(A, Q):
    """
    Solve continuous Lyapunov equation A*X + X*A' + Q = 0.
    """
    import scipy.linalg as la
    return la.solve_continuous_lyapunov(A, -Q)


def dlyap(A, Q):
    """
    Solve discrete Lyapunov equation A*X*A' - X + Q = 0.
    """
    import scipy.linalg as la
    return la.solve_discrete_lyapunov(A, Q)
