"""
Riccati-equation solvers robust to singular weights and unit-circle
eigenvalues.

Port of MATLAB ``dsdlinalg/dare1.m`` (K. Polyakov) — the extended-pencil
(Arnold & Laub 1984) DARE solver used by ``dsdsspace/h2reg.m``'s discrete
Chen-Francis branch. Unlike ``scipy.linalg.solve_discrete_are`` it

* accepts a SINGULAR control weight ``R`` (the pencil is compressed by a QR
  step instead of inverting R) — sampled-data lifted H2/L2 problems have
  ``R = D12'D12`` genuinely tiny/singular and ``D21 = 0``;
* handles pencil eigenvalues ON the unit circle (marginal integrator modes)
  by splitting each near-1 eigenvalue into a (1-eps, 1+eps) pair distributed
  between the stable/antistable deflating subspaces ("Polyakov 2005"), with
  a conditioning-driven retry over the split assignment (the permUcPoles
  search in dare1.m).

References
----------
[1] W.F. Arnold, A.J. Laub, "Generalized Eigenproblem Algorithms and
    Software for Algebraic Riccati Equations", Proc. IEEE 72 (1984).
[2] K. Polyakov, "Separation of eigenvalues on the unit circle", 2005.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

__all__ = ['dare1']

_SQRT_EPS = np.sqrt(np.finfo(float).eps)


def _select_patterns(n_uc):
    """Assignment patterns for the near-unit eigenvalues (True = antistable
    group). MATLAB's default is alternating starting 'inside' (dir=-1 →
    first gets 1-eps); permUcPoles then searches other distributions when
    X1 is ill-conditioned. We enumerate the practical patterns."""
    base = [False, True] * ((n_uc + 1) // 2)
    alt1 = base[:n_uc]                       # inside, outside, inside, ...
    alt2 = [not v for v in alt1]             # outside, inside, ...
    yield alt1
    if n_uc > 0:
        yield alt2
        yield [False] * n_uc
        yield [True] * n_uc


def dare1(A, B, Q, R, S=None, E=None, uc_tol=None):
    """
    Solve the discrete algebraic Riccati equation

        A'XA - E'XE - (A'XB + S)(R + B'XB)^{-1}(B'XA + S') + Q = 0

    via the extended matrix pencil + QZ, tolerating singular ``R`` and
    unit-circle pencil eigenvalues. Port of ``dare1.m``.

    Returns
    -------
    X : (n, n) ndarray
        Stabilizing (symmetric) solution.
    poles : ndarray
        Closed-loop pole set (the stable half of the pencil spectrum).
    err : float
        Relative residual of the Riccati equation (0 when R + B'XB is
        numerically singular and the residual cannot be formed — matching
        dare1.m).

    Raises
    ------
    numpy.linalg.LinAlgError
        If no eigenvalue split yields an invertible X1 (dare1.m returns
        X = inf / err = -1 in that case).
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    R = np.atleast_2d(np.asarray(R, float))
    n = A.shape[0]
    m = B.shape[1]
    if S is None:
        S = np.zeros((n, m))
    S = np.atleast_2d(np.asarray(S, float))
    if E is None:
        E = np.eye(n)
    E = np.atleast_2d(np.asarray(E, float))
    n2 = 2 * n

    if uc_tol is None:
        uc_tol = _SQRT_EPS

    # ── Scaling of the weights (dare1.m) ─────────────────────────────────
    scale = (np.linalg.norm(Q, 1) + np.linalg.norm(R, 1)
             + np.linalg.norm(S, 1))
    if scale < 1e-300:
        raise np.linalg.LinAlgError("dare1: Q, R, S are all zero")
    QQ, RR, SS = Q / scale, R / scale, S / scale

    # ── Extended pencil (2n+m)×(2n+m):  a·x = λ·b·x ─────────────────────
    a_ext = np.block([
        [E,                  np.zeros((n, n)), np.zeros((n, m))],
        [np.zeros((n, n)),   A.T,              np.zeros((n, m))],
        [np.zeros((m, n)),  -B.T,              np.zeros((m, m))],
    ])
    b_ext = np.block([
        [A,                  np.zeros((n, n)), B],
        [-QQ,                E.T,              -SS],
        [SS.T,               np.zeros((m, n)), RR],
    ])

    # ── Compression step (works even when R is singular): project onto the
    # orthogonal complement of b's last m columns ────────────────────────
    q_full, _ = np.linalg.qr(b_ext[:, n2:n2 + m], mode='complete')
    # MATLAB: q(:, 2n+m:-1:m+1)' — the complement columns (reversed order;
    # the reversal does not change the span but is kept for parity).
    qc = q_full[:, m:n2 + m][:, ::-1]
    a_c = qc.T @ a_ext[:, :n2]
    b_c = qc.T @ b_ext[:, :n2]

    # ── Eigenvalues of the compressed pencil (plain QZ first, to identify
    # the near-unit-circle ones deterministically) ───────────────────────
    aa0, bb0, alpha0, beta0, _, _ = la.ordqz(a_c, b_c, sort=lambda a_, b_:
                                             np.ones_like(np.atleast_1d(a_), dtype=bool),
                                             output='complex')
    with np.errstate(divide='ignore', invalid='ignore'):
        lam0 = np.where(np.abs(beta0) < 1e-10, np.inf, alpha0 / beta0)
    n_uc = int(np.sum(np.abs(lam0 - 1.0) < uc_tol))

    best = None   # (condX1, X1, X2, poles)
    for pattern in _select_patterns(n_uc):
        # Stateful selection: True → antistable (top-left) group.
        state = {'k': 0}

        def _sel(alpha, beta):
            alpha = np.atleast_1d(alpha)
            beta = np.atleast_1d(beta)
            out = np.empty(alpha.shape, dtype=bool)
            for i in range(len(alpha)):
                if abs(beta[i]) < 1e-10:
                    out[i] = True                     # λ = ∞ → antistable
                    continue
                lam = alpha[i] / beta[i]
                if abs(lam - 1.0) < uc_tol:
                    j = state['k']
                    out[i] = pattern[j] if j < len(pattern) else True
                    state['k'] = j + 1
                else:
                    out[i] = bool(abs(lam) > 1.0)
            return out

        try:
            aa, bb, alpha, beta, q_z, z = la.ordqz(a_c, b_c, sort=_sel,
                                                   output='complex')
        except Exception:
            continue

        X1 = z[:n, :n]
        X2 = z[n:n2, :n]
        try:
            cond_X1 = np.linalg.cond(X1)
        except Exception:
            continue
        if not np.isfinite(cond_X1):
            continue

        with np.errstate(divide='ignore', invalid='ignore'):
            lam_all = np.where(np.abs(beta) < 1e-10, np.inf, alpha / beta)
        poles = lam_all[n:]

        if best is None or cond_X1 < best[0]:
            best = (cond_X1, X1, X2, poles)
        if cond_X1 < 1e12:
            break

    if best is None:
        raise np.linalg.LinAlgError("dare1: QZ ordering failed")
    cond_X1, X1, X2, poles = best

    # ── X = X2 / X1 (dare1.m: LU with rcond check) ───────────────────────
    if cond_X1 > 1.0 / np.finfo(float).eps:
        raise np.linalg.LinAlgError(
            f"dare1: X1 is numerically singular (cond={cond_X1:.3g}); "
            f"no stabilizing solution found")
    X = np.real(X2 @ np.linalg.inv(X1))
    X = scale * (X + X.T) / 2.0

    # ── Residual (dare1.m: err = 0 when R + B'XB is singular) ────────────
    RBX = R + B.T @ X @ B
    err = 0.0
    if np.linalg.cond(RBX) < 1.0 / np.finfo(float).eps:
        res = (A.T @ X @ A - E.T @ X @ E
               - (A.T @ X @ B + S) @ np.linalg.solve(RBX, (B.T @ X @ A + S.T))
               + Q)
        err = float(np.linalg.norm(res) / max(1.0, np.linalg.norm(X)))

    return X, poles, err
