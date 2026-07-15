"""
Spectral factorization for polynomials and transfer functions.

Ports of: sfactor.m (root-level LTI), sfactfft.m
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as opt
from typing import Tuple, Union

from directsd.polynomial.poln import Poln, _extrpair, _real_if_close, _strip_lz


def sfactor(
    s: Union[Poln, np.ndarray, float, int],
    ftype: str | None = None,
):
    """
    Spectral factorization of a polynomial or LTI transfer function.

    Finds ``fs`` such that ``fs * fs~ ≈ s`` (Hermitian / spectral factorization).
    Neutral zeros/poles are assigned to the stable factor.

    Parameters
    ----------
    s : Poln, array-like, or scipy ZerosPolesGain / lti
    ftype : {'s', 'z', 'd'}, optional
        Factorization type.  Inferred from *s* if omitted.

    Returns
    -------
    fs : spectral factor (minimal realization)
    fs0 : spectral factor before pole-zero cancellation
    """
    # Promote scalars / arrays to Poln
    if isinstance(s, (int, float, complex)):
        s = Poln(np.array([float(s)]))
    if isinstance(s, np.ndarray):
        s = Poln(s.ravel())

    if isinstance(s, Poln):
        if ftype is None:
            ftype = "d" if s.is_dt else "s"
        fs = _sfactor_poln(s, ftype)
        return fs, fs  # both the same (polynomial has no pole-zero cancellation)

    # scipy LTI / ZPK systems
    try:
        import scipy.signal as sig
        if isinstance(s, (sig.lti, sig.dlti, sig.ZerosPolesGain)):
            return _sfactor_lti_scipy(s, ftype)
    except ImportError:
        pass

    raise TypeError(f"sfactor: unsupported input type {type(s)}")


# ---------------------------------------------------------------------------
# Polynomial spectral factorization — delegates to the correct implementation
# ---------------------------------------------------------------------------

def _sfactor_poln(p: Poln, ftype: str) -> Poln:
    """
    Call the canonical polynomial spectral factorization from poln.py.

    That implementation uses ``_extrpair`` and the correct MATLAB gain formula.
    """
    from directsd.polynomial.poln import sfactor as _poly_sfactor
    return _poly_sfactor(p, ftype)


# ---------------------------------------------------------------------------
# LTI (ZPK) spectral factorization — port of root-level sfactor.m
# ---------------------------------------------------------------------------

def _sfactor_lti_scipy(sys, ftype: str | None):
    """
    Spectral factorization of a scipy LTI / ZerosPolesGain system.

    Port of the root-level sfactor.m (K. Polyakov).
    """
    import scipy.signal as sig

    # Convert to ZPK so we can access zeros, poles, gain
    if isinstance(sys, sig.ZerosPolesGain):
        zpk = sys
    elif hasattr(sys, 'to_zpk'):
        zpk = sys.to_zpk()
    else:
        # wrap in ZerosPolesGain
        zpk = sig.ZerosPolesGain(*sig.tf2zpk(sys.num, sys.den))

    z_all = np.asarray(zpk.zeros, dtype=complex).ravel()
    p_all = np.asarray(zpk.poles, dtype=complex).ravel()
    K = float(np.real(zpk.gain))

    # Infer ftype from sample time if not given
    if ftype is None:
        Ts = getattr(zpk, 'dt', 0) or 0
        ftype = 'd' if Ts else 's'

    if K == 0.0:
        fs0 = sig.ZerosPolesGain([], [], 0.0)
        return fs0, fs0

    errmsg = "Exact Hermitian factorization is impossible"

    # ------------------------------------------------------------------
    # Symmetrize zeros (1 % tolerance, matching MATLAB sfactor.m)
    # ------------------------------------------------------------------
    z_symm = _symmetrize_zeros(z_all, ftype, tol=1e-2)

    # ------------------------------------------------------------------
    # Extract symmetric root pairs
    # ------------------------------------------------------------------
    zs, z_rem, n_z0 = _extrpair(z_symm, ftype)
    ps, p_rem, n_p0 = _extrpair(p_all, ftype)

    # Unpaired zeros must equal unpaired poles (they cancel in ratio)
    if z_rem.size != p_rem.size:
        raise ValueError(errmsg)
    if z_rem.size > 0:
        from directsd.polynomial.poln import _others
        if _others(z_rem, p_rem, 1e-2).size > 0:
            raise ValueError(errmsg)

    # ------------------------------------------------------------------
    # Check conjugation of complex zeros (sfactor.m lines 128-134): a
    # complex zero selected into the factor WITHOUT its conjugate partner
    # is realified. This happens e.g. for a conjugate pair ON the unit
    # circle ({c, conj(c)} with |c| = 1): the two members are each other's
    # reciprocal partners, so extrpair keeps only one — leaving a factor
    # with complex coefficients unless it is snapped to its real part.
    # (Without this fixup, Λ comes out complex for spectral densities
    # touching zero on the circle, corrupting every downstream root-list
    # operation.)
    # ------------------------------------------------------------------
    zs = np.asarray(zs, dtype=complex).copy()
    _ctol = 1e-9
    for _i in range(len(zs)):
        _zc = zs[_i]
        if abs(np.imag(_zc)) > _ctol:
            _has_conj = np.any(np.abs(zs - np.conj(_zc)) < _ctol * (1.0 + abs(_zc)))
            if not _has_conj:
                zs[_i] = np.real(_zc)

    # ------------------------------------------------------------------
    # Gain calculation
    # ------------------------------------------------------------------
    n_zs, n_ps = len(zs), len(ps)
    if ftype == 's':
        K_f = float(np.real(K))
        if (n_zs + n_ps) % 2 == 1:
            K_f = -K_f
    else:
        # DT: K * prod(-ps) / prod(-zs)
        if n_zs + n_z0 != n_ps + n_p0:
            raise ValueError(errmsg)
        prod_ps = float(np.real(np.prod(-ps))) if n_ps > 0 else 1.0
        prod_zs = float(np.real(np.prod(-zs))) if n_zs > 0 else 1.0
        K_f = float(np.real(K * prod_ps / prod_zs))

    if K_f < 0:
        raise ValueError("Function is negative definite; spectral factorization impossible")

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    Ts = getattr(zpk, 'dt', None)
    if Ts:
        fs0 = sig.ZerosPolesGain(zs, ps, float(np.sqrt(K_f)), dt=Ts)
    else:
        fs0 = sig.ZerosPolesGain(zs, ps, float(np.sqrt(K_f)))

    # Minimal realisation: cancel common zeros/poles
    try:
        fs_tf = sig.ZerosPolesGain(*sig.tf2zpk(*sig.zpk2tf(zs, ps, np.sqrt(K_f))))
        fs = fs_tf
    except Exception:
        fs = fs0

    return fs, fs0


def _symmetrize_zeros(
    z: np.ndarray, ftype: str, tol: float = 1e-2
) -> np.ndarray:
    """
    Average each zero with its Hermitian mirror to enforce exact symmetry.

    Port of the symmetrization block in sfactor.m (K. Polyakov).
    """
    eps = np.sqrt(np.finfo(float).eps)
    zz = list(z.copy())
    out: list = []

    while zz:
        z0 = zz.pop(0)
        # sfactor.m lines 65-72: the Hermitian mirror is -z0 ('s') / 1/z0
        # (DT) — NOT the conjugate-reciprocal. Using 1/conj(z0) here paired
        # each root with the WRONG member of its conjugate quadruple, and
        # the (1/z0 + z1)/2 average then cancelled the imaginary parts —
        # silently realifying every complex quadruple that passed through
        # sfactor (Λ then comes out with a double real root where MATLAB
        # keeps the complex pair).
        if ftype == 's':
            z0H = -z0
        elif abs(z0) > eps:
            z0H = 1.0 / z0
        else:
            out.append(z0)
            continue

        if not zz:
            out.append(z0)
            break

        diffs = np.abs(np.array(zz) - z0H)
        idx = int(np.argmin(diffs))
        close_enough = (diffs[idx] < tol * (abs(z0) + abs(z0H)) / 2
                        or diffs[idx] < eps)

        if close_enough:
            z1 = zz.pop(idx)
            # Handle real-vs-complex mismatch
            if (np.imag(z0) == 0) != (np.imag(z1) == 0):
                z0 = np.real(z0)
                z1 = np.real(z1)
            if ftype == 's':
                za = (z0 - z1) / 2
                out.extend([za, -za])
            else:
                za = (1.0 / z0 + z1) / 2
                out.extend([za, 1.0 / za])
        else:
            out.append(z0)

    out.extend(zz)
    return np.array(out, dtype=complex)




# ---------------------------------------------------------------------------
# FFT-based spectral factorization — port of sfactfft.m (K. Polyakov)
# ---------------------------------------------------------------------------

def sfactfft(
    p: Union[Poln, np.ndarray],
    ftype: str = 'd',
    N: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Polynomial spectral factorization using FFT / cepstrum (Hromcik's method).

    Parameters
    ----------
    p : Poln or array-like
        Symmetric quasipolynomial (DT only: var in {'z','d','q'}).
    ftype : {'d', 'z'}, optional
        Factorization type.  Default is 'd'.
    N : int, optional
        Zero-padding multiplier; FFT length = ``N * len(coef)``.  Default 10.

    Returns
    -------
    fp : np.ndarray
        Stable spectral factor coefficients.
    fm : np.ndarray
        Unstable (conjugate) spectral factor, ``fm = fp[::-1]``.

    References
    ----------
    Hromcik, Jezek, Sebek (ECC'2001).
    """
    if ftype not in ('z', 'd'):
        raise ValueError(f"sfactfft: unknown type '{ftype}'; must be 'z' or 'd'")

    poly_mode = isinstance(p, Poln)
    if poly_mode:
        var = p.var
        if var not in ('z', 'd', 'q'):
            raise ValueError("sfactfft: only applicable to DT polynomials (z, d, q)")
        p0 = np.asarray(p.coef, dtype=float)
    else:
        p0 = np.asarray(p, dtype=float).ravel()

    n_coef = len(p0)

    # --- Half degree and zero-padding ---
    dg = (n_coef - 1) // 2
    R = N * n_coef
    n_zeros = 2 * R + 1 - n_coef

    # Build the circular correlation sequence:
    # [c_0, c_1, ..., c_dg, 0, ..., 0, c_{-dg}, ..., c_{-1}]
    # In descending p0: c_k = p0[dg - k]; negative lags use p0[:dg]
    p_circ = np.concatenate([
        p0[:dg + 1][::-1],   # [c_0, c_1, ..., c_dg]  (=  p0[dg], ..., p0[0])
        np.zeros(n_zeros),
        p0[:dg],              # [c_{-dg}, ..., c_{-1}]  (=  p0[0], ..., p0[dg-1])
    ])

    # --- FFT I ---
    P = np.fft.fft(p_circ)

    # --- Log ---
    N_log = np.log(P)          # complex log

    # --- IFFT I (cepstrum) ---
    n_cep = np.fft.ifft(N_log)
    xp = n_cep[:R + 1].copy()
    xp[0] = n_cep[0] / 2      # half the DC term

    # --- FFT II → exp → IFFT II ---
    Xp = np.fft.fft(xp)
    Pvp = np.exp(Xp)
    pvp = np.fft.ifft(Pvp)

    # --- Truncate to degree dg ---
    fp = np.real(pvp[:dg + 1])

    # --- Type-specific flip ---
    if ftype[0] == 'd':
        fp = fp[::-1]

    # --- Optional refinement via scipy.optimize ---
    # (Mirrors MATLAB's fminunc call when length(type) > 1)
    if len(ftype) > 1:
        def _obj(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=float)
            p1 = np.convolve(x, x[::-1])
            # Align with p0 (subtract and compute norm)
            e = _sumpol2(p1, -p0)
            return float(np.linalg.norm(e))

        result = opt.minimize(_obj, fp, method='L-BFGS-B',
                              options={'disp': False})
        if result.success:
            fp = result.x

    fm = fp[::-1].copy()

    # --- Restore Poln form ---
    if poly_mode:
        fp_out = Poln(fp, var)
        fm_out = Poln(fm, var)
        return fp_out, fm_out

    return fp, fm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sumpol2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two polynomials of possibly different length (zero-pads shorter)."""
    la, lb = len(a), len(b)
    if la >= lb:
        out = a.copy().astype(float)
        out[la - lb:] += b
    else:
        out = b.copy().astype(float)
        out[lb - la:] += a
    return out
