"""
Diophantine polynomial equation solvers for DirectSD.

  dioph     — X·A + Y·B = C  (basic, separate, or nocancel modes)
  dioph2    — X·A + X̃·B + Y·C = 0  (spectral, X̃ = reciprocal of X)
  diophsys  — system: X·A1+Y1·B1=C1, X·A2+Y2·B2=C2
  diophsys2 — system: X·A1+X̃·B1+Y1·C1=0, X·A2+X̃·B2+Y2·C2=0

Moved out of operations.py so all Diophantine solvers live in one
importable module. This also fixed a real bug: a second,
unrelated `diophsys` in polynomial/utils.py (returning a single shared Y for
both equations, `[X,Y,err]`) had zero callers anywhere in the codebase and no
test coverage, yet `directsd/__init__.py` exported *that* one as the public
`directsd.diophsys` — while the correct implementation (matching MATLAB's
diophsys.m exactly: `[X,Y1,Y2,err,condA]`, separate Y1/Y2, actually used
internally by _polquad's pipeline) was only reachable via a local aliased
import in design/polynomial.py. The wrong version has been deleted; this
diophsys is now the only one and is what `directsd.diophsys` resolves to.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from directsd.polynomial.poln import (
    Poln, coprime as _coprime,
    _strip_lz, _real_if_close,
)
from directsd.polynomial.operations import compat, deg, triple

_EPS = np.finfo(float).eps
_SQRT_EPS = np.sqrt(_EPS)


def _toep(coef: np.ndarray, r: int, c: int) -> np.ndarray:
    """
    Build lower-triangular Toeplitz (convolution) matrix.

    T[i,j] = coef_ascending[i-j]  (0-indexed)
    Input: descending polynomial coefficients (highest degree first).
    """
    from directsd.linalg.matrices import toep
    return toep(coef, r, c)


def _lstsq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SVD-based least-squares with iterative refinement (linsys.m 'svd','refine').

    Truncation tolerance follows MATLAB rank()/linsys.m: max(size(A))·eps
    relative to the largest singular value. A sqrt(eps)-relative cutoff is
    far too aggressive for the block-Sylvester systems built by
    dioph/diophsys: their column blocks are inherently unequally scaled
    (A-block carries the plant gain, often ~1e6; the B-block is a monic
    dLm, ~1), so the smallest genuinely-needed singular value sits ~7
    decades below the largest and gets truncated — the solver then returns
    a garbage minimiser of the wrong subspace even though an exact solution
    exists (root cause of spurious 'pol'→'ss' fallbacks in method
    selection).
    """
    rcond = max(A.shape) * _EPS
    sol, _, _, _ = la.lstsq(A, b, cond=rcond)
    # Iterative refinement (linsys.m lines 131-140: loop while improving)
    res = b - A @ sol
    res_norm = np.linalg.norm(res)
    b_norm = np.linalg.norm(b)
    while res_norm > _SQRT_EPS * b_norm:
        delta, _, _, _ = la.lstsq(A, res, cond=rcond)
        sol_new = sol + delta
        res_new = b - A @ sol_new
        if np.linalg.norm(res_new) >= res_norm:
            break
        sol, res, res_norm = sol_new, res_new, np.linalg.norm(res_new)
    return sol


def _to_coef(p) -> np.ndarray:
    """Return descending coefficient array for Poln or array-like."""
    if isinstance(p, Poln):
        return p.coef.astype(complex)
    return np.atleast_1d(np.asarray(p, dtype=complex)).ravel()


def _from_coef(c: np.ndarray, var: str) -> Poln:
    return Poln(_real_if_close(_strip_lz(np.asarray(c, dtype=complex))), var)


# ---------------------------------------------------------------------------
# Diophantine solver: X·A + Y·B = C
# ---------------------------------------------------------------------------

def dioph(
    a, b, c,
    dtype: str = 'normal',
    degX: int = -1,
    degY: int = -1,
) -> tuple:
    """
    Solve the polynomial Diophantine equation  X·A + Y·B = C.

    Parameters
    ----------
    a, b, c : Poln or array-like
        Polynomial coefficients (descending order) or Poln objects.
    dtype : {'normal', 'nocancel', 'separate'}
        * 'normal'   — cancel GCD(A,B,C) first, then coprime factors
        * 'separate' — cancel only coprime factors from (B,C) and (A,C)
        * 'nocancel' — no cancellation
    degX, degY : int
        Desired degrees. -1 means minimal.

    Returns
    -------
    x, y : Poln
        Solution polynomials.
    err : float
        Residual ||X·A + Y·B - C||.
    condA : float
        Condition number of the Sylvester matrix.
    """
    if dtype not in ('normal', 'nocancel', 'separate'):
        raise ValueError(f"Unknown dtype '{dtype}'")

    a, b, c = compat(a, b, c)
    var = a.var
    zero = Poln(np.array([0.0]), var)

    # Degenerate cases
    if a.norm() < _EPS:
        q, r = b.quorem(c) if isinstance(c, Poln) else b.quorem(Poln(c, var))
        return zero, q, r.norm(), 1.0 / _EPS
    if b.norm() < _EPS:
        q, r = a.quorem(c) if isinstance(c, Poln) else a.quorem(Poln(c, var))
        return q, zero, r.norm(), 1.0 / _EPS

    a0, b0, c0 = a, b, c

    # Extract cancellable common factors
    ac_fac = bc_fac = Poln(np.array([1.0]), var)
    if dtype == 'separate':
        b, c, bc_fac = _coprime(b, c)
        a, c, ac_fac = _coprime(a, c)
    elif dtype == 'normal':
        result = triple(a, b, c)
        if len(result) == 4:
            a, b, c, _ = result
        else:
            a, b, c = result
        b, c, bc_fac = _coprime(b, c)
        a, c, ac_fac = _coprime(a, c)

    ca, cb, cc = _to_coef(a), _to_coef(b), _to_coef(c)

    # Determine polynomial degrees of X and Y
    dA, dB, dC = deg(a), deg(b), deg(c)
    if degX < 0:
        degX = max(dB - 1, 0)
    if degY < 0:
        degY = max(max(dA + degX, dC) - dB, 0)
    if degX < 0:
        degX = 0
    if degY < 0:
        degY = 0

    m, n = degX + 1, degY + 1
    total = m + n

    # Sylvester system: [Toep(A)|Toep(B)] * [x_asc; y_asc] = c_asc
    Am = np.hstack([_toep(ca, total, m), _toep(cb, total, n)])
    Cm = _toep(cc, total, 1).ravel()

    condA = float(np.linalg.cond(Am)) if Am.size > 0 else 1.0
    sol = _lstsq(Am, Cm)

    # Extract solutions (ascending → descending via [::-1])
    xc = _strip_lz(sol[:m][::-1])
    yc = _strip_lz(sol[m:m+n][::-1])

    # Multiply back cancelled factors
    x = _from_coef(xc, var) * bc_fac
    y = _from_coef(yc, var) * ac_fac

    err = float((a0 * x + b0 * y - c0).norm())
    return x, y, err, condA


# ---------------------------------------------------------------------------
# Diophantine solver: X·A + X̃·B + Y·C = 0
# ---------------------------------------------------------------------------

def dioph2(a, b, c, degX: int = -1) -> tuple:
    """
    Solve  X·A + X̃·B + Y·C = 0  (spectral-type Diophantine equation).

    X̃ is the reciprocal polynomial (reversed coefficients).
    The leading coefficient of X is fixed to 1 (monic constraint).

    Parameters
    ----------
    a, b, c : Poln or array-like
    degX : int
        Degree of X. Default: deg(C).

    Returns
    -------
    x, y : Poln
    err : float
        Residual ||X·A + X̃·B + Y·C||.
    """
    a, b, c = compat(a, b, c)
    var = a.var

    ca, cb, cc = _to_coef(a), _to_coef(b), _to_coef(c)
    dA, dB, dC = deg(a), deg(b), deg(c)

    if degX < 0:
        degX = dC
    degY = max(dA, dB)
    degAll = max(degX + max(dA, dB), degY + dC)

    total = degAll + 1
    mX = degX + 1
    nY = degY + 1

    # Coefficient matrix for X: toep(A) + fliplr(toep(B))
    # fliplr(toep(B)) acts on the reversed-X coefficient vector
    # to produce the X̃·B contribution.
    AmX = _toep(ca, total, mX) + np.fliplr(_toep(cb, total, mX))
    AmY = _toep(cc, total, nY)

    # System: [AmY | AmX] * [y_asc; x_asc_without_leading] = -AmX[:, -1]
    # Fix highest coefficient of X to 1 (monic)
    Am = np.hstack([AmY, AmX])
    Cm = -Am[:, -1]
    Am = Am[:, :-1]

    sol = _lstsq(Am, Cm)
    sol = _strip_lz(sol.ravel())

    # Extract Y and X (ascending → descending)
    y_asc = sol[:nY]
    x_asc = np.concatenate([sol[nY:], [1.0]])  # append the fixed leading coef

    xc = _strip_lz(x_asc[::-1])
    yc = _strip_lz(y_asc[::-1])

    x = _from_coef(xc, var)
    y = _from_coef(yc, var)

    err = float((a * x + b * x.reciprocal() + c * y).norm())
    return x, y, err


# ---------------------------------------------------------------------------
# System of two Diophantine equations: X·Ai + Yi·Bi = Ci, i=1,2
# ---------------------------------------------------------------------------

def diophsys(a1, b1, c1, a2, b2, c2) -> tuple:
    """
    Solve the system:
        X·A1 + Y1·B1 = C1
        X·A2 + Y2·B2 = C2

    X is shared; Y1, Y2 are separate. X is of minimal degree.

    Solvability condition: (A2·C1 - A1·C2) must be divisible by GCD(B1, B2).

    Returns
    -------
    x, y1, y2 : Poln
    err : float
        ||X·A1+Y1·B1-C1|| + ||X·A2+Y2·B2-C2||
    condA : float
    """
    a1, b1, c1, a2, b2, c2 = compat(a1, b1, c1, a2, b2, c2)
    var = a1.var

    # Extract GCD of B1 and B2 to determine degree of X
    b1r, b2r, v = _coprime(b1, b2)  # b1 = b1r * v, b2 = b2r * v (sort of)
    # In MATLAB: [B1r,B2r,V] = coprime(PB1, PB2) → B1 = B1r*V, B2 = B2r*V

    # Check solvability (warn if not met, but proceed)
    p_check = a2 * c1 - a1 * c2
    if p_check.norm() > _SQRT_EPS:
        _, rem = p_check.quorem(v)
        if rem.norm() > 1e-6 * p_check.norm():
            import warnings
            warnings.warn(
                f"A2·C1-A1·C2 may not be divisible by GCD(B1,B2) (rel.err={rem.norm()/p_check.norm():.2e})"
            )

    ca1, cb1, cc1 = _to_coef(a1), _to_coef(b1), _to_coef(c1)
    ca2, cb2, cc2 = _to_coef(a2), _to_coef(b2), _to_coef(c2)

    dA1, dB1, dC1 = deg(a1), deg(b1), deg(c1)
    dA2, dB2, dC2 = deg(a2), deg(b2), deg(c2)

    # Degree of X is deg(B1*B2r) - 1 = deg(lcm(B1,B2)) - 1
    # b = B1 * B2r has degree dB1 + deg(B2r)
    b_lcm = b1 * b2r  # lcm of B1 and B2
    degX = max(deg(b_lcm) - 1, 0)

    degY1 = max(dA1 + deg(b2r) - 1, 0)
    degY2 = max(dA2 + deg(b1r) - 1, 0)

    # Adjust if C polynomials exceed the default degree estimate
    if degX + dA1 < dC1:
        degY1 = max(dC1 - dB1, degY1)
    if degX + dA2 < dC2:
        degY2 = max(dC2 - dB2, degY2)

    m = degX + 1
    n1 = degY1 + 1
    n2 = degY2 + 1
    deg1 = max(degX + max(dA1, dB1), dC1)
    deg2 = max(degX + max(dA2, dB2), dC2)
    r1, r2 = deg1 + 1, deg2 + 1

    # Block Sylvester matrix:
    # [Toep(A1,r1,m) | Toep(B1,r1,n1) |     0          ] [x  ]   [C1]
    # [Toep(A2,r2,m) |     0          | Toep(B2,r2,n2) ] [y1 ] = [C2]
    #                                                      [y2 ]
    Am1X = _toep(ca1, r1, m)
    Am1Y1 = _toep(cb1, r1, n1)
    Am2X = _toep(ca2, r2, m)
    Am2Y2 = _toep(cb2, r2, n2)

    Am = np.block([
        [Am1X, Am1Y1, np.zeros((r1, n2))],
        [Am2X, np.zeros((r2, n1)), Am2Y2],
    ])
    Cm = np.concatenate([
        _toep(cc1, r1, 1).ravel(),
        _toep(cc2, r2, 1).ravel(),
    ])

    condA = float(np.linalg.cond(Am))
    sol = _lstsq(Am, Cm)

    xc = _strip_lz(sol[:m][::-1])
    y1c = _strip_lz(sol[m:m+n1][::-1])
    y2c = _strip_lz(sol[m+n1:m+n1+n2][::-1])

    x = _from_coef(xc, var)
    y1 = _from_coef(y1c, var)
    y2 = _from_coef(y2c, var)

    err = float((a1*x + b1*y1 - c1).norm() + (a2*x + b2*y2 - c2).norm())
    return x, y1, y2, err, condA


# ---------------------------------------------------------------------------
# System of two spectral-type equations: X·Ai + X̃·Bi + Yi·Ci = 0, i=1,2
# ---------------------------------------------------------------------------

def diophsys2(a1, b1, c1, a2, b2, c2, degX: int = -1) -> tuple:
    """
    Solve the system:
        X·A1 + X̃·B1 + Y1·C1 = 0
        X·A2 + X̃·B2 + Y2·C2 = 0

    X̃ is the reciprocal of X. The leading coefficient of X is fixed to 1.

    Returns
    -------
    x, y1, y2 : Poln
    err : float
        ||X·A1+X̃·B1+Y1·C1|| + ||X·A2+X̃·B2+Y2·C2||
    """
    a1, b1, c1, a2, b2, c2 = compat(a1, b1, c1, a2, b2, c2)
    var = a1.var

    ca1, cb1, cc1 = _to_coef(a1), _to_coef(b1), _to_coef(c1)
    ca2, cb2, cc2 = _to_coef(a2), _to_coef(b2), _to_coef(c2)

    dA1, dB1, dC1 = deg(a1), deg(b1), deg(c1)
    dA2, dB2, dC2 = deg(a2), deg(b2), deg(c2)

    if degX < 0:
        degX = max(dC1, dC2)

    deg1 = max(degX + max(dA1, dB1), dC1)
    deg2 = max(degX + max(dA2, dB2), dC2)
    degY1 = deg1 - dC1
    degY2 = deg2 - dC2

    r1, r2 = deg1 + 1, deg2 + 1
    mX = degX + 1
    nY1 = degY1 + 1
    nY2 = degY2 + 1

    # X coefficient matrix: toep(Ai) + fliplr(toep(Bi)) for i=1,2
    AmX1 = _toep(ca1, r1, mX) + np.fliplr(_toep(cb1, r1, mX))
    AmY1 = _toep(cc1, r1, nY1)
    AmX2 = _toep(ca2, r2, mX) + np.fliplr(_toep(cb2, r2, mX))
    AmY2 = _toep(cc2, r2, nY2)

    # Flip columns as in MATLAB: fliplr(AmX) puts highest-coef column first
    # Fix highest coef of X to 1 (remove first column = highest coef)
    Am = np.block([
        [np.fliplr(AmX1), np.fliplr(AmY1), np.zeros((r1, nY2))],
        [np.fliplr(AmX2), np.zeros((r2, nY1)), np.fliplr(AmY2)],
    ])
    Cm = -Am[:, 0]
    Am = Am[:, 1:]

    sol = _lstsq(Am, Cm)

    # Extract: first column of X is fixed to 1 (highest coef in descending)
    x_desc = np.concatenate([[1.0], sol[:degX]])   # descending, highest first
    y1_asc = sol[degX:degX+nY1]
    y2_asc = sol[degX+nY1:degX+nY1+nY2]

    xc = _strip_lz(x_desc)
    y1c = _strip_lz(y1_asc[::-1])
    y2c = _strip_lz(y2_asc[::-1])

    x = _from_coef(xc, var)
    y1 = _from_coef(y1c, var)
    y2 = _from_coef(y2c, var)

    err = float(
        (a1*x + b1*x.reciprocal() + c1*y1).norm()
        + (a2*x + b2*x.reciprocal() + c2*y2).norm()
    )
    return x, y1, y2, err
