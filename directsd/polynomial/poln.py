"""
Polynomial class for DirectSD — idiomatic Python port of MATLAB @poln.

Variables:
  's', 'p'       — continuous-time (Laplace / alternative)
  'z', 'q', 'd'  — discrete-time (Z-transform / delta / delta-alt)

A quasipolynomial in 'z' with shift m represents:
  coef[0]*x^(n-m) + coef[1]*x^(n-1-m) + ... + coef[n]*x^(-m)
where n = len(coef) - 1.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union

_VALID_VARS = frozenset("spzqd")
_DT_VARS = frozenset("zqd")
_CT_VARS = frozenset("sp")
_TOL = np.sqrt(np.finfo(float).eps)


# ---------------------------------------------------------------------------
# Coefficient-array utilities
# ---------------------------------------------------------------------------

def _strip_lz(c: np.ndarray, tol: float = np.finfo(float).eps) -> np.ndarray:
    """Strip leading near-zero coefficients. Returns at least [0]."""
    c = np.asarray(c, dtype=complex).ravel()
    nrm = np.linalg.norm(c)
    if nrm < max(tol * tol, 1e-300):
        return np.zeros(1)
    while c.size > 1 and abs(c[0]) <= tol * np.linalg.norm(c[1:] if c.size > 1 else c):
        c = c[1:]
    return c


def _real_if_close(c: np.ndarray) -> np.ndarray:
    if np.max(np.abs(np.imag(c))) < _TOL * (np.max(np.abs(np.real(c))) + 1e-300):
        return np.real(c)
    return c


# ---------------------------------------------------------------------------
# Root-level helpers (used by coprime / add)
# ---------------------------------------------------------------------------

def _remove_common_roots(
    rA: np.ndarray, rB: np.ndarray, tol: float = _TOL
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find roots common to rA and rB (greedy nearest-match).
    Returns (reduced_rA, reduced_rB, common_roots).
    """
    rA, rB = list(rA), list(rB)
    common: list = []
    i = 0
    while i < len(rA):
        if not rB:
            break
        r = rA[i]
        tolr = max(tol, tol * abs(r))
        dists = [abs(r - b) for b in rB]
        j = int(np.argmin(dists))
        if dists[j] < tolr:
            common.append(r)
            rA.pop(i)
            rB.pop(j)
        else:
            i += 1
    return np.array(rA, dtype=complex), np.array(rB, dtype=complex), np.array(common, dtype=complex)


def _others(a: np.ndarray, b: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """Elements of a that are not in b (greedy nearest-match removal)."""
    a, b = list(a), list(b)
    result: list = []
    for ai in a:
        dists = [abs(ai - bj) for bj in b]
        if not dists or min(dists) >= max(tol, tol * abs(ai)):
            result.append(ai)
        else:
            b.pop(int(np.argmin(dists)))
    return np.array(result, dtype=complex)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Poln:
    """
    Polynomial / quasipolynomial for sampled-data control design.

    Primary storage: ``coef`` (descending-power numpy array).
    A quasipolynomial is stored as a plain polynomial times x^{-shift}.

    Parameters
    ----------
    a : array-like or scalar
        Descending-order coefficients, or roots if ``var`` starts with ``'r'``.
    var : str
        Variable: ``'s'``, ``'p'`` (CT) or ``'z'``, ``'q'``, ``'d'`` (DT).
        Prefix ``'r'`` selects root-input mode, e.g. ``var='rz'``.
    shift : int
        Non-negative delay exponent for DT quasipolynomials.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, a=0, var: str = "s", shift: int = 0) -> None:
        root_mode = isinstance(var, str) and var.startswith("r")
        if root_mode:
            var = var[1:] or "s"

        if var not in _VALID_VARS:
            raise ValueError(f"Unknown variable '{var}'. Use one of: s p z q d")
        if var in _CT_VARS and shift > 0:
            raise ValueError("'shift' is only for discrete-time polynomials")
        if shift < 0:
            raise ValueError("'shift' must be non-negative")

        self.var = var
        self.shift = int(shift)

        if isinstance(a, Poln):
            self.coef = a.coef.copy()
            self.var = var
            self.shift = a.shift
            return

        a = np.atleast_1d(np.asarray(a, dtype=complex)).ravel()

        if root_mode:
            self.coef = _real_if_close(np.poly(a)) if a.size > 0 else np.array([1.0])
        else:
            if a.size == 0:
                a = np.zeros(1, dtype=complex)
            self.coef = _real_if_close(_strip_lz(a.astype(complex)))

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def k(self) -> float:
        """Leading coefficient."""
        return float(np.real(self.coef[0])) if self.coef.size > 0 else 0.0

    @property
    def roots(self) -> np.ndarray:
        """Roots of the polynomial (computed on demand)."""
        if np.linalg.norm(self.coef) < np.finfo(float).eps or len(self.coef) <= 1:
            return np.array([], dtype=complex)
        return np.roots(self.coef)

    @property
    def degree(self) -> int:
        """Highest positive-power degree. For quasipolynomials: len(coef)-1-shift."""
        if np.linalg.norm(self.coef) < np.finfo(float).eps:
            return 0
        return len(self.coef) - 1 - self.shift

    @property
    def deg_neg(self) -> int:
        """Highest negative-power degree (= shift for quasipolynomials)."""
        if np.linalg.norm(self.coef) < np.finfo(float).eps:
            return 0
        return self.shift

    @property
    def is_ct(self) -> bool:
        return self.var in _CT_VARS

    @property
    def is_dt(self) -> bool:
        return self.var in _DT_VARS

    def norm(self) -> float:
        return float(np.linalg.norm(self.coef))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __neg__(self) -> Poln:
        return Poln(-self.coef, self.var, self.shift)

    def __mul__(self, other) -> Poln:
        if isinstance(other, (int, float, complex, np.number)):
            return Poln(float(np.real(other)) * self.coef, self.var, self.shift)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            return self.__mul__(float(other))
        a, b = compat(self, other)
        return Poln(np.polymul(a.coef, b.coef), a.var, a.shift + b.shift)

    def __rmul__(self, other) -> Poln:
        return self.__mul__(other)

    def __add__(self, other) -> Poln:
        if isinstance(other, (int, float, complex)):
            other = Poln(np.array([other]), self.var)
        a, b = compat(self, other)
        return _poly_add(a, b)

    def __radd__(self, other) -> Poln:
        return self.__add__(other)

    def __sub__(self, other) -> Poln:
        if isinstance(other, (int, float, complex)):
            other = Poln(np.array([other]), self.var)
        return self.__add__(-other)

    def __rsub__(self, other) -> Poln:
        return (-self).__add__(other)

    def __pow__(self, n: int) -> Poln:
        n = int(n)
        # Special case: (scalar * x)^n for a pure linear monomial
        r = self.roots
        if self.shift == 0 and self.degree == 1 and r.size == 1 and abs(r[0]) < np.finfo(float).eps:
            if n < 0:
                if self.is_ct:
                    raise ValueError("Negative power not admissible for CT polynomials")
                return Poln(np.array([self.k ** n]), self.var, -n)
            if n == 0:
                return Poln(np.array([1.0]), self.var)
            new_z = np.zeros(n, dtype=complex)
            c = float(self.k ** n) * np.poly(new_z)
            return Poln(_real_if_close(c), self.var)
        if n < 0:
            raise ValueError("Negative power not supported for general polynomials")
        if n == 0:
            return Poln(np.array([1.0]), self.var)
        result = self
        for _ in range(n - 1):
            result = result * self
        return result

    def __truediv__(self, other) -> Poln:
        """Exact polynomial division. Use quorem() for division with remainder."""
        if isinstance(other, (int, float, complex)):
            return Poln(self.coef / float(np.real(other)), self.var, self.shift)
        q, r = quorem(self, other)
        if r.norm() > _TOL * max(self.norm(), 1e-300):
            raise ValueError("Division is not exact; use quorem() for quotient + remainder")
        return q

    def __call__(self, x):
        """Evaluate the (quasi)polynomial at x."""
        if self.shift > 0 and x == 0:
            return float("inf")
        val = np.polyval(self.coef, x)
        if self.shift > 0:
            val = val / (x ** self.shift)
        return float(np.real(val)) if abs(np.imag(complex(x))) < np.finfo(float).eps else val

    # ------------------------------------------------------------------
    # Control-specific methods
    # ------------------------------------------------------------------

    def c2d(self, T: float = 1.0, var: str = "z") -> Poln:
        """Discretize CT polynomial: roots r → exp(-r*T)."""
        if self.is_dt:
            raise ValueError("Already discrete-time")
        if var not in _DT_VARS:
            raise ValueError(f"Unknown DT variable '{var}'")
        new_z = np.exp(-self.roots * T)
        c = _real_if_close(self.k * np.poly(new_z)) if new_z.size > 0 else np.array([self.k])
        return Poln(c, var, self.shift)

    def c2z(self, T: float = 1.0, var: str = "z") -> Poln:
        """Discretize CT polynomial: roots r → exp(r*T)."""
        if self.is_dt:
            raise ValueError("Already discrete-time")
        if var not in _DT_VARS:
            raise ValueError(f"Unknown DT variable '{var}'")
        new_z = np.exp(self.roots * T)
        c = _real_if_close(self.k * np.poly(new_z)) if new_z.size > 0 else np.array([self.k])
        return Poln(c, var, self.shift)

    def conj_reciprocal(self) -> Poln:
        """
        Conjugate-reciprocal polynomial (MATLAB ctranspose ``'``).

        CT (s/p): P'(s) = P(-s) with sign fix for odd degree.
        DT (z/q/d): P'(z) = z^{-(n+m)} * P(1/z), roots r → 1/r.
        """
        if self.is_ct:
            # Transform: coef → fliplr, negate odd positions, fliplr back.
            # The root map r → -r and this coefficient op are equivalent.
            # The odd-degree sign lives inside the transformed coef; no extra -1 needed.
            c = np.flipud(self.coef.copy())
            c[1::2] = -c[1::2]
            c = np.flipud(c)
            return Poln(c, self.var, self.shift)
        else:
            z = self.roots
            ind_zero = np.where(np.abs(z) < np.finfo(float).eps)[0]
            z_nz = np.delete(z, ind_zero)
            k_new = float(np.real(self.k * np.prod(-z_nz))) if z_nz.size > 0 else self.k
            z_new = 1.0 / z_nz if z_nz.size > 0 else z_nz
            shift_new = int(len(z_nz) + len(ind_zero) - self.shift)
            if shift_new < 0:
                z_new = np.concatenate([z_new, np.zeros(-shift_new)])
                shift_new = 0
            c = _real_if_close(k_new * np.poly(z_new)) if z_new.size > 0 else np.array([k_new])
            if shift_new + 1 > len(c):
                c = np.concatenate([np.zeros(shift_new + 1 - len(c)), c])
            return Poln(c, self.var, shift_new)

    def reciprocal(self) -> Poln:
        """Reciprocal polynomial (reverse coefficient order, strip leading zeros)."""
        if self.shift > 0:
            raise ValueError("reciprocal() is not applicable to quasipolynomials")
        return Poln(_real_if_close(_strip_lz(self.coef[::-1])), self.var)

    def shift_by(self, n: int) -> Poln:
        """Return self * x^n  (n may be negative for DT quasipolynomials)."""
        n = int(n)
        if n == 0:
            return Poln(self.coef.copy(), self.var, self.shift)
        new_shift = self.shift - n
        if n > 0 and new_shift < 0:
            extra = np.zeros(-new_shift, dtype=self.coef.dtype)
            new_coef = np.concatenate([self.coef, extra])
            new_shift = 0
        elif not self.is_dt and n < 0:
            z = self.roots
            ind_zero = np.where(np.abs(z) < np.finfo(float).eps)[0]
            need = -new_shift
            if len(ind_zero) < need:
                raise ValueError("shift_by(x^-N) requires enough zero roots for CT polynomials")
            z_new = np.delete(z, ind_zero[:need])
            new_coef = (_real_if_close(self.k * np.poly(z_new)) if z_new.size > 0
                        else np.array([self.k]))
            new_shift = 0
        else:
            new_coef = self.coef.copy()
        return Poln(new_coef, self.var, max(0, new_shift))

    def common_den(self, other: Poln) -> Poln:
        """Polynomial with the union of roots from self and other (LCM denominator)."""
        a, b = compat(self, other)
        extra = _others(b.roots, a.roots)
        all_poles = np.concatenate([a.roots, extra])
        if all_poles.size == 0:
            return Poln(np.array([1.0]), a.var)
        return Poln(all_poles, "r" + a.var)

    def derivative(self, n: int = 1) -> Poln:
        """n-th derivative polynomial."""
        p = self
        for _ in range(n):
            d = p.degree
            if d == 0:
                return Poln(np.array([0.0]), self.var)
            new_coef = np.array([(d - i) * c for i, c in enumerate(p.coef[:-1])], dtype=complex)
            p = Poln(_real_if_close(new_coef), self.var)
        return p

    def monic(self) -> Poln:
        """Return monic version (leading coefficient = 1)."""
        if abs(self.k) < 1e-14:
            return Poln(np.array([0.0]), self.var, self.shift)
        return Poln(self.coef / self.k, self.var, self.shift)

    def quorem(self, other: Union[Poln, float]) -> Tuple[Poln, Poln]:
        """Quotient and remainder: self = q*other + r, deg(r) < deg(other)."""
        return quorem(self, other)

    def sfactor(self, ftype: str | None = None) -> Poln:
        """Stable spectral factor of a Hermitian polynomial."""
        return sfactor(self, ftype)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Poln({list(np.round(self.coef, 10))}, var='{self.var}', shift={self.shift})"

    def __str__(self) -> str:
        return _poly_to_str(self)

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, Poln):
            return False
        return (self.var == other.var and self.shift == other.shift
                and np.allclose(self.coef, other.coef))


# ---------------------------------------------------------------------------
# Module-level functions (mirror MATLAB calling convention)
# ---------------------------------------------------------------------------

def compat(*polys) -> tuple:
    """
    Coerce all arguments to share the same variable.

    The first Poln argument's variable wins; scalars are promoted.
    """
    var = None
    for p in polys:
        if isinstance(p, Poln):
            var = p.var
            break
    if var is None:
        return tuple(polys)
    result = []
    for p in polys:
        if isinstance(p, Poln):
            result.append(p if p.var == var else Poln(p.coef, var, p.shift))
        else:
            result.append(Poln(np.atleast_1d(np.asarray(p, dtype=float)), var))
    return tuple(result)


def coprime(
    a: Poln, b: Poln, tol: float = _TOL
) -> Tuple[Poln, Poln, Poln]:
    """
    Cancel common roots of two polynomials.

    Returns (a/gcd, b/gcd, gcd) where gcd is monic.
    """
    a, b = compat(a, b)
    var = a.var
    g = Poln(np.array([1.0]), var)
    if a.degree < 1 or a.norm() < 1e-10:
        return a, b, g
    if b.degree < 1 or b.norm() < 1e-10:
        return a, b, g
    rA, rB, rG = _remove_common_roots(a.roots, b.roots, tol)
    if rG.size == 0:
        return a, b, g
    kA, kB = a.k, b.k
    cA = _real_if_close(kA * np.poly(rA)) if rA.size > 0 else np.array([kA])
    cB = _real_if_close(kB * np.poly(rB)) if rB.size > 0 else np.array([kB])
    cG = _real_if_close(np.poly(rG)) if rG.size > 0 else np.array([1.0])
    return (
        Poln(cA, var, a.shift),
        Poln(cB, var, b.shift),
        Poln(cG, var),
    )


def gcd(a: Poln, b: Poln) -> Poln:
    """Monic greatest common divisor of two polynomials."""
    _, _, g = coprime(a, b)
    return g


def quorem(a: Poln, b: Union[Poln, float]) -> Tuple[Poln, Poln]:
    """
    Polynomial division with remainder: a = q*b + r, deg(r) < deg(b).

    Returns (quotient, remainder).
    """
    if isinstance(b, (int, float, complex)):
        return Poln(a.coef / float(np.real(b)), a.var, a.shift), Poln(np.array([0.0]), a.var)

    a, b = compat(a, b)
    shift_a, shift_b = a.shift, b.shift

    # Work on plain (shift=0) polynomials
    pa = Poln(a.coef.copy(), a.var)
    pb = Poln(b.coef.copy(), b.var)
    pa, pb, _ = coprime(pa, pb)

    if pb.degree > 0:
        q_coef, r_coef = np.polydiv(pa.coef, pb.coef)
        q = Poln(_real_if_close(_strip_lz(q_coef)), a.var)
        r = Poln(_real_if_close(_strip_lz(r_coef)), a.var)
    else:
        q = Poln(_real_if_close(pa.coef / pb.k), a.var)
        r = Poln(np.array([0.0]), a.var)

    q = q.shift_by(shift_b - shift_a)
    r = r.shift_by(-shift_a)
    return q, r


def sfactor(p: Poln, ftype: str | None = None) -> Poln:
    """
    Stable spectral factor of a Hermitian polynomial.

    Finds ``fs`` such that ``p ≈ fs * fs.conj_reciprocal()``.
    ``ftype``: ``'s'`` for CT (default), ``'z'`` or ``'d'`` for DT.
    """
    if ftype is None:
        ftype = "d" if p.is_dt else "s"
    if ftype not in ("s", "z", "d"):
        raise ValueError(f"Unknown factorization type '{ftype}'")
    if abs(p.k) < np.finfo(float).eps:
        return Poln(np.array([0.0]), p.var)

    zs, z_rem, z0 = _extrpair(p.roots, ftype)
    if z_rem.size > 0 or z0 > 0:
        raise ValueError("Exact Hermitian factorization is impossible (unpaired roots remain)")

    if ftype == "s":
        K = float(np.real(p.k))
        if len(zs) % 2 == 1:
            K = -K
    else:
        K = float(np.real(p.k / np.prod(-zs))) if zs.size > 0 else float(np.real(p.k))

    if K < 0:
        raise ValueError("Exact Hermitian factorization is impossible (negative gain)")

    c = _real_if_close(np.sqrt(K) * np.poly(zs)) if zs.size > 0 else np.array([np.sqrt(K)])
    return Poln(c, p.var)


def _extrpair(
    z: np.ndarray, ftype: str, tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract symmetric root pairs.

    Port of MATLAB `extrpair.m` (K. Polyakov) — full 3-output form, including
    the "zeros at the origin" count needed by `minreals`-style symmetric
    minimal realizations. Previously only the 2-output (`zs`, unpaired
    non-origin roots) form was ported here, which silently treated any root
    exactly at the origin as an unresolvable "unpaired" failure (`sfactor`'s
    only caller raises whenever anything comes back in that slot) instead of
    counting it separately, as MATLAB does.

    's': pairs z_i + z_j ≈ 0; stable root has Re < 0.
    'z': pairs z_i*z_j ≈ 1; stable root has |z| < 1.
    'd': pairs z_i*z_j ≈ 1; stable root has |z| > 1.

    Returns
    -------
    zs : ndarray
        One "stable" representative per extracted symmetric pair.
    z_rem : ndarray
        Unpaired, nonzero leftover roots (empty for a genuinely symmetric input).
    z0 : int
        Count of remaining roots at the origin (removed from `z_rem`).
    """
    z = list(np.asarray(z, dtype=complex).ravel())
    z.sort(key=lambda zi: -abs(zi))
    zs: list = []
    while len(z) >= 2:
        min_dist = np.inf
        i_min = j_min = -1
        for i in range(len(z)):
            for j in range(len(z)):
                if i == j:
                    continue
                dij = abs(z[i] + z[j]) if ftype == "s" else abs(z[i] * z[j] - 1)
                # MATLAB: exclude i as a candidate "iMin" only when z[i] is
                # complex and exactly on the unit circle, UNLESS it's close
                # to the point z=1 specifically (line 49-51 of extrpair.m):
                # `abs(z(i)) == 1` -- a LITERAL bitwise equality against
                # 1.0, not a tolerance check. Porting this
                # as `abs(abs(z[i])-1) < eps` looks equivalent but ISN'T --
                # MATLAB's `==1` is true only for a value that IS exactly
                # 1.0 to the last bit (never true for a genuine unit-circle
                # eigenvalue computed via `eig`/`expm`, which always carries
                # ~1e-16-level floating-point noise), so in real MATLAB
                # usage this exclusion never actually fires. The `<eps`
                # tolerance version DOES fire for such noise-level
                # deviations (e.g. abs(z)=0.9999999999999999, off from 1.0
                # by 1.11e-16 < eps=2.22e-16) -- exactly the case for a
                # genuine on-unit-circle conjugate pair with no other root
                # available as an alternative pairing anchor, where BOTH
                # members get excluded (each is complex and "on-circle") and
                # neither can ever become `i_min`, so the pair is wrongly
                # rejected as unpaired (`_sdh2coef`'s "could not symmetrize
                # A1" warning, `test_sdahinf_returns_controller` et al.).
                # Matching MATLAB's literal `==1` here (never true for a
                # computed value) is not a bug retained on purpose -- MATLAB
                # never intended this branch to be reachable in practice.
                near_one = abs(z[i] - 1) < tol
                on_unit_circle_complex = (
                    abs(z[i]) == 1.0
                    and abs(np.imag(z[i])) > np.finfo(float).eps
                )
                excluded = on_unit_circle_complex and not near_one
                if not excluded and dij < min_dist:
                    i_min, j_min, min_dist = i, j, dij
        if i_min < 0 or (min_dist > tol and min_dist > tol * abs(z[i_min])):
            break
        zi, zj = z[i_min], z[j_min]
        if ftype == "s":
            chosen = zi if np.real(zi) < 0 else zj
        elif ftype == "z":
            chosen = zi if abs(zi) < 1 else zj
        else:
            chosen = zi if abs(zi) > 1 else zj
        zs.append(chosen)
        for idx in sorted([i_min, j_min], reverse=True):
            z.pop(idx)

    # Post-pass: any complex entry in `zs` whose nearest match (by distance
    # to its own conjugate) is itself has no genuine conjugate partner in
    # `zs` — force it real, matching MATLAB's "unpaired complex" cleanup
    # (extrpair.m lines 75-86).
    zs_arr = np.array(zs, dtype=complex)
    for i in range(len(zs_arr)):
        if np.imag(zs_arr[i]) == 0:
            continue
        dists = np.abs(zs_arr - np.conj(zs_arr[i]))
        if int(np.argmin(dists)) == i:
            zs_arr[i] = np.real(zs_arr[i])

    # Zeros at the origin: counted and removed separately, not left in the
    # "unpaired" leftover list (extrpair.m lines 87-95).
    z_arr = np.array(z, dtype=complex)
    origin_mask = np.abs(z_arr) < tol
    z0 = int(np.sum(origin_mask))
    z_rem = z_arr[~origin_mask]

    return zs_arr, z_rem, z0


# ---------------------------------------------------------------------------
# Polynomial addition (private — handles quasipolynomial shifts and GCD)
# ---------------------------------------------------------------------------

def _poly_add(a: Poln, b: Poln) -> Poln:
    """Add two Poln objects with the same variable. Handles quasipolynomial shifts."""
    eps = np.finfo(float).eps
    an, bn = a.norm(), b.norm()
    if an < eps * (bn + eps) and bn < eps * (an + eps):
        return Poln(np.array([0.0]), a.var)
    if an < eps * (bn + eps):
        return b
    if bn < eps * (an + eps):
        return a

    # Extract GCD to improve numerical stability
    ac, bc, x = coprime(a, b)

    # Align quasipolynomial degrees (MATLAB plus.m logic)
    pA, mA = ac.degree, ac.deg_neg
    pB, mB = bc.degree, bc.deg_neg
    p = max(pA, pB)
    m = max(mA, mB)

    cA = np.concatenate([np.zeros(p - pA), ac.coef, np.zeros(m - mA)])
    cB = np.concatenate([np.zeros(p - pB), bc.coef, np.zeros(m - mB)])
    c_sum = cA + cB

    nrm = np.linalg.norm(c_sum)
    if nrm > 0:
        c_sum[np.abs(c_sum) < _TOL * nrm] = 0.0
    c_sum = _strip_lz(c_sum)

    result_coef = np.polymul(_real_if_close(c_sum), x.coef)
    result = Poln(_real_if_close(result_coef), a.var, m)
    return _zeroing(result)


def _zeroing(p: Poln) -> Poln:
    """Cancel zero roots against shift (reduces quasipolynomial trailing terms)."""
    eps = np.finfo(float).eps
    if p.norm() < eps or p.shift == 0:
        return p
    z = p.roots
    ind_zero = np.where(np.abs(z) < eps)[0]
    n_cancel = min(p.shift, len(ind_zero))
    if n_cancel == 0:
        return p
    new_coef = p.coef[:-n_cancel]
    new_shift = p.shift - n_cancel
    return Poln(_real_if_close(new_coef), p.var, new_shift)


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------

def _poly_to_str(p: Poln) -> str:
    coef = p.coef
    n = len(coef)
    deg_top = n - 1 - p.shift
    parts = []
    for i, ci in enumerate(coef):
        cur_deg = deg_top - i
        ci_r = float(np.real(ci))
        if abs(ci_r) < 1e-12:
            continue
        sign = "- " if ci_r < 0 else ("+ " if parts else "")
        val = abs(ci_r)
        show_num = abs(val - 1.0) > 1e-12 or cur_deg == 0
        term = sign
        if show_num:
            term += f"{val:g}"
        if cur_deg != 0:
            term += (" " if show_num else "") + p.var
            if cur_deg != 1:
                term += f"^{cur_deg}"
        parts.append(term.rstrip())
    return " ".join(parts) if parts else "0"


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (used by operations.py and other modules)
# ---------------------------------------------------------------------------
_DISCRETE_VARS = _DT_VARS          # old name
_CONTINUOUS_VARS = _CT_VARS        # old name
_VALID_VARS = frozenset("spzqd")   # re-export (was set, now frozenset — same values)

def _findzero(coef, tol=np.sqrt(np.finfo(float).eps)):
    """Alias for _strip_lz kept for backward compatibility."""
    return _strip_lz(np.asarray(coef, dtype=complex), tol=tol)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def s_var() -> Poln:
    """Elementary polynomial: s."""
    return Poln([1, 0], "s")

def z_var() -> Poln:
    """Elementary polynomial: z."""
    return Poln([1, 0], "z")

def q_var() -> Poln:
    """Elementary polynomial: q."""
    return Poln([1, 0], "q")

def p_var() -> Poln:
    """Elementary polynomial: p."""
    return Poln([1, 0], "p")

def d_var() -> Poln:
    """Elementary polynomial: d."""
    return Poln([1, 0], "d")
