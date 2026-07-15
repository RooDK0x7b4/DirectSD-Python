"""
Root-list (zero-pole-gain) representation with MATLAB CST/DirectSD semantics.

Carries (zeros, poles, gain) through rational arithmetic the way MATLAB's
zpk objects do — cancellations are EXACT root matching and roots are never
re-derived from high-degree coefficient arrays (only sums re-root their
small reduced numerators, mirroring sumzpk.m). This representation is what
makes MATLAB's polynomial-design pipeline numerically survive: running the
original ynyd.m/zterm.m/polhinf.m algorithms with a coefficient-array
pipeline instead fails to cancel Z = E - B*B~/A exactly (yielding spurious
degree 10/12 results instead of the true degree 3/4), because the shared
roots of E and B*B~/A only agree to floating-point tolerance rather than
exactly. Exact root-list cancellation sidesteps that failure mode entirely.

Ports: @lti/ctranspose.m (DT), @zpk/minreal.m (via Minreal._reducezp),
@zpk/minreals.m, @zpk/symmetr.m, sumzpk.m (SISO), root-level sfactor
(via spectral._sfactor_lti_scipy).

This module holds the Zpk class as a single file rather than mirroring
MATLAB's @zpk/ folder (one file per method): that per-method split is an
artifact of old-style MATLAB class folders, not evidence that Python
should split the class the same way — a single class file is the
idiomatic Python equivalent. `_zpk_snap` lives here alongside it since
it's a Zpk-construction helper, not a design algorithm.
"""

from __future__ import annotations

import numpy as np

_SQRT_EPS = float(np.sqrt(np.finfo(float).eps))

__all__ = ['Zpk', '_zpk_snap']


def _col(x):
    return np.atleast_1d(np.asarray(x, complex)).ravel()


def _realify(r, tol=_SQRT_EPS):
    r = _col(r).copy()
    m = np.abs(np.imag(r)) < tol
    r[m] = np.real(r[m])
    return r


def _conj_symmetric(r, tol=1e-7):
    """True iff every non-real entry has a conjugate partner (the invariant
    of a real-coefficient rational's root list)."""
    r = _col(r)
    cplx = r[np.abs(np.imag(r)) > tol * (1.0 + np.abs(r))]
    for x in cplx:
        if not np.any(np.abs(cplx - np.conj(x)) < tol * (1.0 + abs(x))):
            return False
    return True


def _others2(a, b, tol=_SQRT_EPS):
    """others.m 2-output semantics: elements of `a` not matched in `b`
    (greedy nearest matching within tol*(1+|.|)), plus the matched ones."""
    a = list(_col(a)); b = list(_col(b))
    rest, common = [], []
    for x in a:
        hit = -1
        best = tol * (1.0 + abs(x))
        for j, y in enumerate(b):
            d = abs(x - y)
            if d < best:
                best = d; hit = j
        if hit >= 0:
            b.pop(hit)
            common.append(x)
        else:
            rest.append(x)
    return np.array(rest, complex), np.array(common, complex)


class Zpk:
    """SISO zero-pole-gain rational: k * prod(x - z_i) / prod(x - p_j)."""

    __slots__ = ('z', 'p', 'k')

    def __init__(self, z, p, k):
        self.z = _col(z)
        self.p = _col(p)
        self.k = float(np.real(k))

    # ── construction / conversion ───────────────────────────────────────
    @staticmethod
    def from_tf(num, den):
        num = np.atleast_1d(np.asarray(num, float)).ravel()
        den = np.atleast_1d(np.asarray(den, float)).ravel()
        # striplz semantics: leading coefficients that are negligible
        # RELATIVE to the polynomial's scale are padding, not degree — a
        # 1e-16-relative leading term would otherwise root into a spurious
        # ~1e13 zero carrying a compensating ~1e-16 gain (as seen in A/E out
        # of _sdh2coef for demo_h2hinf's double-integrator plant).
        _nmax = np.max(np.abs(num)) if num.size else 0.0
        _dmax = np.max(np.abs(den)) if den.size else 0.0
        while len(num) > 1 and abs(num[0]) <= 1e-12 * _nmax:
            num = num[1:]
        while len(den) > 1 and abs(den[0]) <= 1e-12 * _dmax:
            den = den[1:]
        # Trailing noise coefficients are different: the low-order terms of a
        # quasipolynomial can be STRUCTURALLY zero (an exact origin root, e.g.
        # z·(a z² + b z + a) self-adjoint against a palindromic quartic), so
        # a noise-level constant term must be snapped to exact 0 — not
        # dropped — letting np.roots produce an exact origin root that
        # conj_dt's origin bookkeeping handles. Left as ~1e-16 it roots into
        # a ~1e-14 zero that conj_dt inverts into a ~1e13 monster (how the
        # artifact reached A/E in ζ-domain).
        num = num.copy(); den = den.copy()
        num[np.abs(num) <= 1e-12 * _nmax] = 0.0
        den[np.abs(den) <= 1e-12 * _dmax] = 0.0
        num = np.trim_zeros(num, 'f')
        den = np.trim_zeros(den, 'f')
        if len(num) == 0:
            return Zpk([], [], 0.0)
        z = np.roots(num) if len(num) > 1 else np.zeros(0)
        p = np.roots(den) if len(den) > 1 else np.zeros(0)
        return Zpk(z, p, num[0] / den[0])

    def to_tf(self):
        num = np.real(self.k * np.poly(self.z)) if len(self.z) else np.array([self.k])
        den = np.real(np.poly(self.p)) if len(self.p) else np.array([1.0])
        return np.atleast_1d(num).astype(float), np.atleast_1d(den).astype(float)

    def copy(self):
        return Zpk(self.z.copy(), self.p.copy(), self.k)

    def __repr__(self):
        return (f"Zpk(k={self.k:.10g}, z={np.round(self.z, 6)}, "
                f"p={np.round(self.p, 6)})")

    # ── arithmetic ──────────────────────────────────────────────────────
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Zpk(self.z, self.p, self.k * other)
        return Zpk(np.concatenate([self.z, other.z]),
                   np.concatenate([self.p, other.p]),
                   self.k * other.k)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Zpk(self.z, self.p, self.k / other)
        if other.k == 0.0:
            raise ZeroDivisionError("Zpk: division by zero system")
        return Zpk(np.concatenate([self.z, other.p]),
                   np.concatenate([self.p, other.z]),
                   self.k / other.k)

    def __neg__(self):
        return Zpk(self.z, self.p, -self.k)

    # ── conjugate H(1/z) — port of @lti/ctranspose.m, discrete branch ───
    def conj_dt(self):
        zj = np.conj(self.z); pj = np.conj(self.p)
        origin_z = np.abs(zj) < 1e-14
        origin_p = np.abs(pj) < 1e-14
        n_oz = int(np.sum(origin_z)); n_op = int(np.sum(origin_p))
        zj = zj[~origin_z]; pj = pj[~origin_p]
        k = self.k
        if len(zj):
            k = k * float(np.real(np.prod(-zj)))
        if len(pj):
            k = k / float(np.real(np.prod(-pj)))
        zj = 1.0 / zj if len(zj) else zj
        pj = 1.0 / pj if len(pj) else pj
        zpow = n_op + len(pj) - (n_oz + len(zj))
        zn = np.concatenate([zj, np.zeros(max(zpow, 0), complex)])
        pn = np.concatenate([pj, np.zeros(max(-zpow, 0), complex)])
        return Zpk(zn, pn, k)

    # ── minreal — @zpk/minreal.m (REDUCEZP core shared with Minreal) ────
    def minreal(self, tol=_SQRT_EPS):
        from directsd.linalg.minreal import Minreal
        zr, pr = Minreal._reducezp(self.z, self.p, tol)
        return Zpk(zr, pr, self.k)

    # ── minreals — @zpk/minreals.m (reciprocal-pair-preserving) ─────────
    def minreals(self, tol=_SQRT_EPS, ftype='d'):
        from directsd.polynomial.poln import _extrpair
        from directsd.linalg.minreal import Minreal
        zs, z_rem, z0 = _extrpair(_realify(self.z), ftype, tol)
        ps, p_rem, p0 = _extrpair(_realify(self.p), ftype, tol)
        if len(np.atleast_1d(z_rem)) or len(np.atleast_1d(p_rem)):
            raise ValueError("Zpk.minreals: the function is not symmetric")
        zs = _col(zs); ps = _col(ps)
        c0 = min(z0, p0)
        z0 -= c0; p0 -= c0
        if len(zs) + z0 != len(ps) + p0:
            raise ValueError("Zpk.minreals: incorrect zeros or poles at the origin")
        # cancel common representative roots (each removal also drops the
        # reciprocal partner because zs/ps store one root per pair)
        modified = False
        i = 0
        ps_l = list(ps); zs_l = list(zs)
        while i < len(ps_l):
            if not zs_l:
                break
            A = ps_l[i]
            tolA = max(tol * abs(A), tol)
            d = np.abs(np.array(zs_l) - A)
            if d.min() < tolA:
                modified = True
                targets = [A, np.conj(A)] if abs(np.imag(A)) > 1e-14 else [A]
                zs_l = list(Minreal._remove(np.array(zs_l), targets, 100 * tolA))
                ps_l = list(Minreal._remove(np.array(ps_l), targets, 100 * tolA))
                i = 0
            else:
                i += 1
        if not modified:
            return self.copy()
        zs = np.array(zs_l, complex); ps = np.array(ps_l, complex)
        zz = np.concatenate([zs, 1.0 / zs if len(zs) else zs,
                             np.zeros(z0, complex)])
        pp = np.concatenate([ps, 1.0 / ps if len(ps) else ps,
                             np.zeros(p0, complex)])
        # Invariant: a real-coefficient rational's root lists are
        # conjugate-symmetric. The reciprocal rebuild can break this when
        # extrpair mistakes a near-unit-circle CONJUGATE pair {c, c~} for
        # reciprocal partners (|c*c~ - 1| small because |c| ~ 1) and keeps
        # only one member — MATLAB's tighter default tolerance rejects that
        # pairing and minreals errors instead: the broken rebuild produces
        # exactly-on-circle complex singles that even MATLAB's own extrpair
        # cannot process.
        if not (_conj_symmetric(zz) and _conj_symmetric(pp)):
            raise ValueError("Zpk.minreals: reduction would break conjugate "
                             "symmetry (near-circle conjugate pair mistaken "
                             "for a reciprocal pair)")
        return Zpk(zz, pp, self.k)

    # ── symmetr — @zpk/symmetr.m ('z' default for DT) ───────────────────
    def symmetr(self, ftype='z', tol=1e-2):
        from directsd.polynomial.poln import _extrpair
        zs, z_rem, z0 = _extrpair(_realify(self.z), ftype, tol)
        ps, p_rem, p0 = _extrpair(_realify(self.p), ftype, tol)
        if len(np.atleast_1d(z_rem)) or len(np.atleast_1d(p_rem)):
            raise ValueError("Zpk.symmetr: cannot symmetrize the fraction")
        zs = _col(zs); ps = _col(ps)
        g = min(z0, p0)
        z0 -= g; p0 -= g
        pe = len(ps) + p0 - len(zs) - z0
        if pe != 0:
            raise ValueError("Zpk.symmetr: cannot symmetrize the fraction")
        zz = np.concatenate([zs, 1.0 / zs if len(zs) else zs,
                             np.zeros(z0, complex)])
        pp = np.concatenate([ps, 1.0 / ps if len(ps) else ps,
                             np.zeros(p0, complex)])
        if not (_conj_symmetric(zz) and _conj_symmetric(pp)):
            raise ValueError("Zpk.symmetr: symmetrization would break "
                             "conjugate symmetry")
        return Zpk(zz, pp, self.k)

    # ── setpoles — @zpk/setpoles.m (SISO) ───────────────────────────────
    def setpoles(self, p, tol=1e-4):
        """Force the pole list to `p`: existing poles matching an entry of
        `p` are snapped to it; EXTRA poles (not in `p`) cancel against the
        nearest zero (analytic cancellation declared by the caller — this
        is how PCancel poles leave L in ynyd.m); missing entries of `p` are
        inserted as pole+zero pairs (function unchanged). DT origin poles
        are never cancelled."""
        F = self.minreal()
        p_rem = list(_col(p))
        pF = list(F.p); zF = list(F.z)
        pFCorr = []
        for i in range(len(pF) - 1, -1, -1):
            if not p_rem:
                break
            d = [abs(q - pF[i]) for q in p_rem]
            j = int(np.argmin(d))
            if d[j] < tol:
                pFCorr.append(p_rem[j])
                pF.pop(i)
                p_rem.pop(j)
        # remaining pF are extra poles: keep origin poles, cancel the rest
        # against the nearest zeros
        lack = []
        for q in pF:
            if abs(q) < np.finfo(float).eps:
                pFCorr.append(q)
            else:
                lack.append(q)
        for q in lack:
            if not zF:
                raise ValueError(f"Zpk.setpoles: initial model contains an "
                                 f"extra pole {q}")
            d = [abs(zz - q) for zz in zF]
            zF.pop(int(np.argmin(d)))
        # remaining desired poles are new: insert as pole+zero pairs
        zF.extend(p_rem)
        pFCorr.extend(p_rem)
        return Zpk(np.array(zF, complex), np.array(pFCorr, complex), F.k)

    # ── pole-aware sum — sumzpk.m (SISO, two terms) ─────────────────────
    def zsum(self, other, tol=_SQRT_EPS):
        a = self; b = other
        if a.k == 0.0:
            return b.minreal()
        if b.k == 0.0:
            return a.minreal()
        az = _realify(a.z); ap = _realify(a.p)
        bz = _realify(b.z); bp = _realify(b.p)
        # separate common zeros / poles (kept aside, reattached at the end)
        bz_r, zcommon = _others2(bz, az, tol)
        az_r, _ = _others2(az, zcommon, tol)
        zcommon = np.roots(np.real(np.poly(zcommon))) if len(zcommon) else zcommon
        bp_r, pcommon = _others2(bp, ap, tol)
        ap_r, _ = _others2(ap, pcommon, tol)
        # reduced-part polynomial sum (small degrees; only the numerator of
        # the sum is re-rooted — poles are carried over exactly)
        na = a.k * (np.poly(az_r) if len(az_r) else np.array([1.0]))
        da = np.poly(ap_r) if len(ap_r) else np.array([1.0])
        nb = b.k * (np.poly(bz_r) if len(bz_r) else np.array([1.0]))
        db = np.poly(bp_r) if len(bp_r) else np.array([1.0])
        n = np.polyadd(np.polymul(na, db), np.polymul(nb, da))
        n = np.real(n)
        n = np.trim_zeros(n, 'f')
        if len(n) == 0 or np.max(np.abs(n)) < 1e-300:
            return Zpk([], [], 0.0)
        z_new = np.roots(n) if len(n) > 1 else np.zeros(0)
        p_new = np.concatenate([_col(ap_r), _col(bp_r)])
        s = Zpk(np.concatenate([z_new, _col(zcommon)]),
                np.concatenate([p_new, _col(pcommon)]),
                n[0])
        return s.minreal()

    # ── spectral factor (root level, joint gain rule) ───────────────────
    def sfactor(self, ftype='d'):
        import scipy.signal as sig
        from directsd.polynomial.spectral import _sfactor_lti_scipy
        _, fs0 = _sfactor_lti_scipy(
            sig.ZerosPolesGain(self.z, self.p, self.k), ftype)
        return Zpk(np.atleast_1d(fs0.zeros), np.atleast_1d(fs0.poles),
                   float(np.real(fs0.gain)))


# ---------------------------------------------------------------------------
# _zpk_snap -- Zpk.from_tf + a setpoles-style pole snap onto a known set
# (a Zpk-construction helper, not a design algorithm, so it belongs here
# alongside Zpk itself)
# ---------------------------------------------------------------------------

def _zpk_snap(num, den, known_poles, tol=1e-3):
    """
    Zpk.from_tf + a setpoles-style snap of the pole list onto the
    exactly-known set (MATLAB's setpoles discipline: discretized quantities
    have poles at exp(±λT) for known continuous eigenvalues λ; snapping the
    np.roots-derived copies back onto those exact values is what lets every
    downstream root-list cancellation match exactly).

    When the whole pole list can be matched one-to-one against the target
    MULTISET (greedy nearest pairing, each target consumed once) within a
    loose 5e-2 tolerance, the list is replaced wholesale — the equivalent of
    setpoles.m's correct→cancel→insert net effect. This is what handles
    high-multiplicity roots: np.roots scatters a (z-1)⁴ factor by
    ±|ε|^(1/4) ≈ 3e-3, far beyond any per-root snap tolerance that would
    still be safe globally, but analytically every pole of these
    discretized quantities IS in the target set, so a full-cover match is
    trustworthy at a loose tolerance (as with double-integrator plants).
    Otherwise only the individually-close poles are snapped (tight tol).
    """
    Z = Zpk.from_tf(num, den)
    kp = np.atleast_1d(np.asarray(known_poles, complex)).ravel()
    if len(kp) and len(Z.p):
        # Greedy nearest pairing with multiset consumption.
        avail = list(kp)
        pairs = []          # (dist, pole_index, target)
        for i, rt in enumerate(Z.p):
            if not avail:
                pairs = None
                break
            d = [abs(rt - t) for t in avail]
            j = int(np.argmin(d))
            pairs.append((d[j], i, avail.pop(j)))
        # Accept the wholesale replacement only when every match is both
        # close (5e-2) and UNAMBIGUOUS — the pole must be much nearer its
        # matched target than any *different* target value. Without the
        # dominance check, small-T problems (where distinct exp(λT) crowd
        # around 1) get legitimately-distinct poles merged (measured:
        # sdahinf(1/(s+1)², T=0.1) → hinfbisec failure → nan).
        def _unambiguous(d, i, t):
            others = np.abs(kp[np.abs(kp - t) > 1e-9] - Z.p[i])
            return len(others) == 0 or d < 0.2 * float(np.min(others))
        p = Z.p.copy()
        if pairs is not None and all(
                d < 5e-2 * (1.0 + abs(t)) and _unambiguous(d, i, t)
                for d, i, t in pairs):
            for _, i, t in pairs:
                p[i] = t
        else:
            for i, rt in enumerate(p):
                d = np.abs(kp - rt)
                j = int(np.argmin(d))
                if d[j] < tol * (1.0 + abs(rt)):
                    p[i] = kp[j]
        Z = Zpk(Z.z, p, Z.k)
    return Z
