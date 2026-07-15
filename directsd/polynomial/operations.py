"""
Polynomial operations for DirectSD — general polynomial algebra helpers
(degree, GCD/coprime factoring, triple cancellation, factorization, etc.).

Diophantine equation solvers (dioph, dioph2, diophsys, diophsys2) live in
directsd.polynomial.diophantine, so all Diophantine solvers are importable
from one place.
"""

from __future__ import annotations

import numpy as np

from directsd.polynomial.poln import (
    Poln, coprime as _coprime,
    _findzero, _real_if_close,
)

_EPS = np.finfo(float).eps
_SQRT_EPS = np.sqrt(_EPS)

# NumPy version shim (kept for backward compat with existing tests)
_trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# ---------------------------------------------------------------------------
# Utility helpers (keep public aliases for other modules)
# ---------------------------------------------------------------------------

def compat(*polys):
    """Coerce all arguments to share the same variable, widening raw scalars/
    arrays/scipy LTI objects/(num,den) tuples to `Poln` along the way.

    NOT a re-export of `poln.compat` -- deliberately a
    separate, WIDER function: `poln.compat` is an internal-use-only helper
    for `Poln`'s own dunder methods (always called with at least one `Poln`
    argument already in hand, e.g. `self`), so it can afford to pass
    non-Poln inputs through unchanged when no `Poln` is present at all. This
    version is the public-facing entry point for `coprime`/`gcd`/`triple`
    and `directsd.polynomial.diophantine`'s solvers, which routinely receive
    raw arrays, scipy `lti`/`dlti` objects, or `(num,den)` tuples and need
    them ALWAYS promoted to `Poln` (defaulting to `var='s'` if no `Poln`
    argument is present at all) before any polynomial algebra can proceed.
    """
    import scipy.signal as sig
    var = None
    for p in polys:
        if isinstance(p, Poln):
            var = p.var
            break
    if var is None:
        var = 's'
    out = []
    for p in polys:
        if isinstance(p, Poln):
            out.append(p if p.var == var else Poln(p.coef, var, p.shift))
        elif isinstance(p, (int, float, complex, np.number)):
            out.append(Poln(np.array([complex(p)]), var))
        elif isinstance(p, np.ndarray) and p.ndim == 0:
            out.append(Poln(np.array([float(p)]), var))
        elif isinstance(p, np.ndarray):
            out.append(Poln(p.ravel(), var))
        elif isinstance(p, (sig.lti, sig.dlti)):
            out.append(Poln(np.atleast_1d(p.den).ravel(), var))
        elif isinstance(p, tuple) and len(p) == 2:
            out.append(Poln(np.atleast_1d(p[1]).ravel(), var))
        else:
            raise TypeError(f"Cannot coerce {type(p)} to Poln")
    return out[0] if len(out) == 1 else tuple(out)


def deg(p) -> int:
    if isinstance(p, Poln):
        return p.degree
    return len(np.atleast_1d(np.asarray(p))) - 1


def striplz(coef, tol=_SQRT_EPS):
    arr = np.atleast_1d(np.array(coef, dtype=complex))
    while arr.size > 1 and abs(arr[0]) <= tol:
        arr = arr[1:]
    return np.real_if_close(arr, tol=1e6)


def coprime(A, B, tol=None):
    """Cancel common roots of A and B; returns (A/gcd, B/gcd, gcd)."""
    if tol is None:
        tol = _SQRT_EPS
    A, B = compat(A, B)
    return _coprime(A, B, tol)


def gcd(A, B, *rest):
    """Monic GCD of two (or more) polynomials."""
    _, _, G = _coprime(*compat(A, B))
    for C in rest:
        _, _, G = _coprime(*compat(G, C))
    return G.monic()


def triple(A, B, C, tol=None):
    """
    Extract common GCD of three polynomials simultaneously.

    Returns (A/g, B/g, C/g, g) where g is the monic common factor.
    Equivalent to MATLAB triple().
    """
    if tol is None:
        tol = _SQRT_EPS
    A, B, C = compat(A, B, C)
    var = A.var

    g = Poln(np.array([1.0]), var)
    if A.degree < 1 or A.norm() < 1e-10:
        if A.norm() < 1e-10:
            B, C, g = _coprime(B, C, tol)
        return A, B, C, g
    if B.degree < 1 or B.norm() < 1e-10:
        if B.norm() < 1e-10:
            A, C, g = _coprime(A, C, tol)
        return A, B, C, g
    if C.degree < 1 or C.norm() < 1e-10:
        if C.norm() < 1e-10:
            A, B, g = _coprime(A, B, tol)
        return A, B, C, g

    rA = list(A.roots)
    rB = list(B.roots)
    rC = list(C.roots)
    kA, kB, kC = A.k, B.k, C.k
    rG: list = []

    i = 0
    while i < len(rA):
        if not rB or not rC:
            break
        R = rA[i]
        tolR = max(tol, tol * abs(R))
        dB = [abs(R - b) for b in rB]
        dC = [abs(R - c) for c in rC]
        jB, jC = int(np.argmin(dB)), int(np.argmin(dC))
        if dB[jB] < tolR and dC[jC] < tolR:
            rG.append(R)
            rA.pop(i)
            rB.pop(jB)
            rC.pop(jC)
        else:
            i += 1

    if not rG:
        return A, B, C, g

    cA = _real_if_close(kA * np.poly(rA)) if rA else np.array([kA])
    cB = _real_if_close(kB * np.poly(rB)) if rB else np.array([kB])
    cC = _real_if_close(kC * np.poly(rC)) if rC else np.array([kC])
    cG = _real_if_close(np.poly(rG))
    return (
        Poln(cA, var, A.shift),
        Poln(cB, var, B.shift),
        Poln(cC, var, C.shift),
        Poln(cG, var),
    )


def factor(f, ftype=None):
    """Split f into (stable, unstable, neutral) polynomial factors."""
    if not isinstance(f, Poln):
        f = Poln(np.atleast_1d(f))
    if ftype is None:
        ftype = 'd' if f.is_dt else 's'
    if ftype not in ('s', 'z', 'd'):
        raise ValueError(f"Unknown type '{ftype}'")
    var = f.var
    rts = f.roots
    k = f.k
    tol = _SQRT_EPS
    if ftype == 's':
        ms = np.real(rts) < -tol
        mu = np.real(rts) > tol
    elif ftype == 'z':
        ms = np.abs(rts) < 1 - tol
        mu = np.abs(rts) > 1 + tol
    else:
        ms = np.abs(rts) > 1 + tol
        mu = np.abs(rts) < 1 - tol
    m0 = ~ms & ~mu

    def _b(r, gain):
        c = gain * np.poly(r) if len(r) > 0 else np.array([gain])
        return Poln(np.real_if_close(c, tol=1e6), var)

    return _b(rts[ms], k), _b(rts[mu], 1.0), _b(rts[m0], 1.0)


def recip(p):
    if isinstance(p, Poln):
        return p.reciprocal()
    return _findzero(np.atleast_1d(np.array(p))[::-1])


def vec(p):
    if isinstance(p, Poln):
        return p.coef.copy()
    return np.atleast_1d(np.array(p)).ravel()


def delzero(A, tol=None):
    """
    Remove zeros at the origin from a polynomial.

    Port of MATLAB ``@poln/delzero`` and ``private/delzero``.

    Parameters
    ----------
    A : Poln or array-like
        Input polynomial or coefficient array.
    tol : float, optional
        Tolerance for identifying zero roots.  Default: machine epsilon.

    Returns
    -------
    B : Poln or np.ndarray
        Reduced polynomial / array with zero roots removed.
    nz : int
        Number of zeros removed.
    """
    if tol is None:
        tol = _EPS

    if isinstance(A, Poln):
        z = A.roots
        mask = np.abs(z) < tol
        nz = int(np.sum(mask))
        if nz == 0:
            return A, 0
        z_new = z[~mask]
        k = A.k
        c = _real_if_close(k * np.poly(z_new)) if z_new.size > 0 else np.array([k])
        return Poln(c, A.var, A.shift), nz

    # Plain array: strip trailing near-zero coefficients
    arr = np.atleast_1d(np.array(A, dtype=float)).ravel()
    nz = 0
    while arr.size > 1 and abs(arr[-1]) <= tol:
        arr = arr[:-1]
        nz += 1
    return arr, nz
