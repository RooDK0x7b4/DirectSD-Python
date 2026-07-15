"""
Advanced global optimization routines for DirectSD.

Ports of dsdglopt (K. Polyakov, 2006):
  Utilities   : updateopt, uniproj, u2range, randbeta, randgamma, sa_testfun
  Encoding    : val2bin, bin2val, coord2hilb, hilb2coord
  Sector maps : r2range, r1range, admproj, par2cp, cp2par, guesspoles
  Controllers : k2ksi, go_par2k, f_sdh2p, f_sdl2p, go_sdh2p, go_sdl2p
  Optimizers  : sasimplex, arandsearch, infglob, infglobc, optglob, optglobc
"""

from __future__ import annotations

import warnings
import numpy as np


# ---------------------------------------------------------------------------
# Simple utilities
# ---------------------------------------------------------------------------

def updateopt(options: dict, opt: dict) -> dict:
    """
    Update an options dict with fields from *opt*, ignoring unknown keys.

    Port of MATLAB ``updateopt``.
    """
    if not isinstance(opt, dict):
        return options
    result = dict(options)
    for k, v in opt.items():
        if k in result:
            result[k] = v
    return result


def uniproj(x):
    """
    Project values onto [0, 1].

    Port of MATLAB ``uniproj``.
    """
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def u2range(u, lo, hi):
    """
    Linear mapping from [0, 1] onto [lo, hi].

    Port of MATLAB ``u2range``.
    """
    return float(lo) + (float(hi) - float(lo)) * float(u)


def randgamma(a, rows=1, cols=None):
    """
    Gamma-distributed random numbers using Marsaglia-Tsang (2000).

    Port of MATLAB ``randgamma``.
    """
    if cols is None:
        cols = rows
    out = np.zeros((rows, cols))
    if a == 0:
        return out
    if a < 1:
        return randgamma(a + 1, rows, cols) * np.random.rand(rows, cols) ** (1.0 / a)
    d = a - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    for idx in np.ndindex(rows, cols):
        while True:
            while True:
                x = np.random.randn()
                v = 1.0 + c * x
                if v > 0:
                    break
            v = v ** 3
            u = np.random.rand()
            if u < 1 - 0.0331 * x ** 4:
                out[idx] = d * v
                break
            if np.log(u) < 0.5 * x ** 2 + d * (1 - v + np.log(v)):
                out[idx] = d * v
                break
    return out


def randbeta(a, b, rows=1, cols=None):
    """
    Beta-distributed random numbers via ratio of Gamma samples.

    Port of MATLAB ``randbeta``.
    """
    if cols is None:
        cols = rows
    g1 = randgamma(a, rows, cols)
    g2 = randgamma(b, rows, cols)
    return g1 / (g1 + g2)


def sa_testfun(x):
    """
    Complex test function with many local minima; global minimum f=0 at (0,0).

      f(x,y) = x²  + 2y² + 0.3(1−cos(3πx)) + 0.4(1−cos(4πy))

    Port of MATLAB ``sa_testfun``.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    a, b = 1.0, 2.0
    c, d_ = 0.3, 0.4
    alpha, gamma = 3 * np.pi, 4 * np.pi
    return (a * x[0] ** 2 + b * x[1] ** 2
            + c * (1 - np.cos(alpha * x[0]))
            + d_ * (1 - np.cos(gamma * x[1])))


# ---------------------------------------------------------------------------
# Binary / Hilbert-curve encoding
# ---------------------------------------------------------------------------

def val2bin(x, nfrac: int = 0):
    """
    Convert a non-negative real number to binary strings.

    Returns
    -------
    bint  : str  — integer part in binary
    bfrac : str  — fractional part in binary (length = nfrac)

    Port of MATLAB ``val2bin``.
    """
    x = abs(float(x))
    ix = int(x)
    fx = x - ix
    bint = bin(ix)[2:] or '0'
    bfrac = ''
    for _ in range(int(nfrac)):
        fx *= 2
        bit = int(fx)
        fx -= bit
        bfrac += str(bit)
    return bint, bfrac


def bin2val(bint: str, bfrac: str) -> float:
    """
    Convert binary strings to a real number.

    Port of MATLAB ``bin2val``.
    """
    x = int(bint, 2) if bint else 0
    if bfrac:
        x += int(bfrac, 2) / (2 ** len(bfrac))
    return float(x)


def coord2hilb(coord, precision: int) -> float:
    """
    Map an N-D vector in [0,1]^N to a 1-D value in [0,1] using the Hilbert
    space-filling curve.

    Parameters
    ----------
    coord     : array-like, shape (N,)  — coordinates in [0, 1]
    precision : int — bits per coordinate

    Returns
    -------
    x : float in [0, 1]

    Port of MATLAB ``coord2hilb`` (Lawder 2000).
    """
    coord = np.asarray(coord, dtype=float).ravel()
    dimensions = len(coord)
    p = int(precision)

    # Build alpha matrix (p × d) — binary digits of each coordinate
    alpha = np.zeros((p, dimensions), dtype=int)
    for i in range(dimensions):
        _, s = val2bin(coord[i], p)
        for j in range(p):
            alpha[j, i] = int(s[j]) if j < len(s) else 0

    # Allocate arrays
    omega      = np.zeros((p, dimensions), dtype=int)
    rho        = np.zeros((p, dimensions), dtype=int)
    sigma      = np.zeros((p, dimensions), dtype=int)
    tilde_sigma = np.zeros((p, dimensions), dtype=int)
    tau        = np.zeros((p, dimensions), dtype=int)
    tilde_tau  = np.zeros((p, dimensions), dtype=int)
    J          = np.zeros(p, dtype=int)
    shift      = np.zeros(p + 1, dtype=int)

    for i in range(p):
        if i > 0:
            omega[i, :] = omega[i - 1, :] ^ tilde_tau[i - 1, :]

        tilde_sigma[i, :] = alpha[i, :] ^ omega[i, :]

        # Left-cyclic shift of tilde_sigma by shift[i]
        k = shift[i]
        sig = np.concatenate([tilde_sigma[i, :], tilde_sigma[i, :]])
        sigma[i, :] = sig[k: k + dimensions]

        # Gray-code-like: rho[0] = sigma[0]; rho[j] = sigma[j] ^ rho[j-1]
        rho[i, 0] = sigma[i, 0]
        for j in range(1, dimensions):
            rho[i, j] = sigma[i, j] ^ rho[i, j - 1]

        # Principal position J[i]: max index where rho != rho[-1]
        rhoi = rho[i, :]
        ind = [j for j in range(dimensions) if rhoi[j] != rhoi[-1]]
        J[i] = max(ind) + 1 if ind else dimensions  # 1-based

        shift[i + 1] = (shift[i] + J[i] - 1) % dimensions

        # tau: invert last bit; if odd parity, also invert J-th bit (1-based)
        tau[i, :] = sigma[i, :]
        tau[i, -1] ^= 1
        if np.sum(tau[i, :]) % 2 == 1:
            tau[i, J[i] - 1] ^= 1

        # tilde_tau: right-cyclic shift of tau by shift[i]
        k = dimensions - shift[i]
        titau = np.concatenate([tau[i, :], tau[i, :]])
        tilde_tau[i, :] = titau[k: k + dimensions]

    # Flatten rho column-major (row by row in transposed sense) → binary string
    r = rho.T.ravel()
    r_str = ''.join(str(b) for b in r)
    if len(r_str) > 52:
        r_str = r_str[:52]

    x = bin2val('0', r_str)
    return x


def hilb2coord(x: float, dimensions: int, precision: int) -> np.ndarray:
    """
    Map a 1-D value in [0, 1] to an N-D vector using the Hilbert curve.

    Parameters
    ----------
    x          : float in [0, 1]
    dimensions : int — output dimensionality
    precision  : int — bits per output coordinate

    Returns
    -------
    coord : np.ndarray, shape (dimensions,)

    Port of MATLAB ``hilb2coord`` (Butz 1971, Lawder 2000).
    """
    d = int(dimensions)
    p = int(precision)

    _, r_str = val2bin(float(x), d * p)
    # Pad to length d*p
    r_str = r_str.ljust(d * p, '0')[:d * p]
    r = np.array([int(c) for c in r_str], dtype=int)

    # rho: shape (p, d) — each row is one precision level
    rho = r.reshape(p, d)

    # J[i]: principal position in each row
    J = np.zeros(p, dtype=int)
    for i in range(p):
        rhoi = rho[i, :]
        ind = [j for j in range(d) if rhoi[j] != rhoi[-1]]
        J[i] = max(ind) + 1 if ind else d  # 1-based

    # sigma: Gray-code of rho  →  sigma[i,j] = rho[i,j] ^ rho[i,j+1] (with rho[i,-1]=0)
    sigma = np.zeros((p, d), dtype=int)
    for i in range(p):
        sig_row = np.concatenate([[rho[i, 0]], np.zeros(d, dtype=int)])
        # sigma = rho XOR (rho shifted by 1)
        sig_row = rho[i, :] ^ np.concatenate([[0], rho[i, :-1]])
        sigma[i, :] = sig_row

    # tau: complement last bit; if odd parity, complement J-th bit
    tau = sigma.copy()
    tau[:, -1] ^= 1
    for i in range(p):
        if np.sum(tau[i, :]) % 2 == 1:
            tau[i, J[i] - 1] ^= 1

    # shifts: shift[i] = sum_{k=0}^{i-1} (J[k]-1) mod d
    shift_arr = np.zeros(p, dtype=int)
    for i in range(1, p):
        shift_arr[i] = (shift_arr[i - 1] + J[i - 1] - 1) % d

    # tilde_sigma: right-cyclic shift of sigma[i] by shift[i]
    tilde_sigma = np.zeros((p, d), dtype=int)
    tilde_sigma[0, :] = sigma[0, :]
    for i in range(1, p):
        k = d - shift_arr[i]
        ts = np.concatenate([sigma[i, :], sigma[i, :]])
        tilde_sigma[i, :] = ts[k: k + d]

    # tilde_tau: right-cyclic shift of tau[i] by shift[i]
    tilde_tau = np.zeros((p, d), dtype=int)
    tilde_tau[0, :] = tau[0, :]
    for i in range(1, p):
        k = d - shift_arr[i]
        tt = np.concatenate([tau[i, :], tau[i, :]])
        tilde_tau[i, :] = tt[k: k + d]

    # omega[i] = omega[i-1] ^ tilde_tau[i-1]
    omega = np.zeros((p, d), dtype=int)
    for i in range(1, p):
        omega[i, :] = omega[i - 1, :] ^ tilde_tau[i - 1, :]

    # alpha[i] = omega[i] ^ tilde_sigma[i]
    alpha = np.zeros((p, d), dtype=int)
    for i in range(p):
        alpha[i, :] = omega[i, :] ^ tilde_sigma[i, :]

    # Convert alpha columns to real coordinates
    coord = np.zeros(d)
    for i in range(d):
        s = ''.join(str(b) for b in alpha[:, i])
        coord[i] = bin2val('0', s)
    return coord


# ---------------------------------------------------------------------------
# Stability sector parameter mapping
# ---------------------------------------------------------------------------

def r2range(Ea: float, Eb: float, shifted: bool):
    """
    Compute bounds for the free term r2 in a second-order factor (z² + r1·z + r2).

    Parameters
    ----------
    Ea      : exp(−α·T)
    Eb      : exp(−π/β)
    shifted : True for shifted sector, False for truncated

    Returns
    -------
    r2min, r2max, E0

    Port of MATLAB ``r2range``.
    """
    r2max = Ea ** 2
    if not shifted:
        E0 = min(Ea, Eb)
        r2min = -Ea * E0
    else:
        r2min = -(Ea ** 2) * Eb
        E0 = Ea * Eb
    return float(r2min), float(r2max), float(E0)


def r1range(r2: float, Ea: float, beta: float, shifted: bool):
    """
    Compute bounds for r1 given r2 and sector parameters.

    Port of MATLAB ``r1range``.
    """
    Eb = np.exp(-np.pi / beta) if not np.isinf(beta) and beta > 0 else 0.0
    E0 = Ea * Eb if shifted else min(Ea, Eb)
    r2crit = E0 ** 2
    r1min = -r2 / max(Ea, 1e-12) - Ea
    if r2 <= r2crit:
        r1max = r2 / max(E0, 1e-12) + E0
    else:
        sqr2 = np.sqrt(max(r2, 0.0))
        if shifted:
            sqr2 /= max(Ea, 1e-12)
        if np.isinf(beta) or sqr2 < 1e-12:
            r1max = 2.0 * np.sqrt(max(r2, 0.0))
        else:
            r1max = -2.0 * sqr2 * np.cos(-beta * np.log(sqr2))
    return float(r1min), float(r1max), float(E0)


def admproj(p, alpha: float = 0.0, beta: float = np.inf,
            dom: str = 's', T: float = 1.0):
    """
    Project poles onto an admissible stability sector.

    Parameters
    ----------
    p     : array-like — poles to project
    alpha : degree of stability (≥ 0)
    beta  : oscillation limit (>0 truncated sector, <0 shifted sector)
    dom   : 's' (CT), 'z', or 'd'
    T     : sampling period (for 'z' or 'd')

    Returns
    -------
    p_proj : np.ndarray — projected poles

    Port of MATLAB ``admproj``.
    """
    p = np.atleast_1d(np.array(p, dtype=complex)).ravel().copy()
    shifted = bool(beta < 0)
    beta = abs(float(beta))

    p_zero = np.array([], dtype=complex)
    if dom == 'z':
        ind_zero = np.where(np.abs(p) < 1e-12)[0]
        p_zero = np.zeros(len(ind_zero), dtype=complex)
        p = np.delete(p, ind_zero)

    if dom == 'z':
        p = np.log(p) / T
    elif dom == 'd':
        p = -np.log(p) / T

    p = p + (shifted * alpha)
    alpha0 = (1 - int(shifted)) * alpha

    for i in range(len(p)):
        re = float(np.real(p[i]))
        im = float(np.imag(p[i]))
        re = min(re, -alpha0)
        betaX = abs(im / re) if abs(re) > 1e-12 else np.inf
        if betaX > beta and not np.isinf(beta):
            if dom != 's' and abs(im - np.pi) < 1e-10:
                re = -np.pi / beta
            else:
                im = im * beta / betaX
        p[i] = re + 1j * im

    p = p - (shifted * alpha)

    if dom != 's':
        ind_pi = np.where(np.abs(np.imag(p) - np.pi) < 1e-10)[0]
        if dom == 'z':
            p = np.exp(p * T)
        elif dom == 'd':
            p = np.exp(-p * T)
        p[ind_pi] = np.real(p[ind_pi])
        if dom == 'z':
            p = np.concatenate([p, p_zero])

    return p


def par2cp(rho, alpha: float = 0.0, beta: float = np.inf, n_pairs: int = None):
    """
    Map a vector of parameters rho ∈ [0,1]^n to the characteristic polynomial.

    Parameters
    ----------
    rho     : array-like — parameters in [0, 1]
    alpha   : stability degree
    beta    : oscillation limit (>0 truncated, <0 shifted)
    n_pairs : number of complex-conjugate pairs (default: len(rho)//2)

    Returns
    -------
    Delta   : np.ndarray — reciprocal of DeltaZ (for dioph input)
    DeltaZ  : np.ndarray — characteristic polynomial in z
    poles   : np.ndarray — closed-loop poles

    Port of MATLAB ``par2cp``.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    if n_pairs is None:
        n_pairs = len(rho) // 2

    shifted = bool(beta < 0)
    beta_abs = abs(float(beta))
    Ea = np.exp(-float(alpha))
    Eb = np.exp(-np.pi / beta_abs) if not np.isinf(beta_abs) and beta_abs > 0 else 0.0

    if not shifted and Eb > Ea and not np.isinf(beta_abs):
        beta_abs = np.inf
        Eb = 1.0

    r2min, r2max, E0 = r2range(Ea, Eb, shifted)

    poles = np.zeros(len(rho), dtype=complex)
    DeltaZ = np.array([1.0])
    k = 0

    for _ in range(n_pairs):
        r2 = r2min + np.clip(rho[k], 0.0, 1.0) * (r2max - r2min)
        r1min, r1max, _ = r1range(r2, Ea, beta_abs, shifted)
        r1 = r1min + np.clip(rho[k + 1], 0.0, 1.0) * (r1max - r1min)
        d = np.array([1.0, r1, r2])
        poles[k: k + 2] = np.roots(d)
        DeltaZ = np.polymul(DeltaZ, d)
        k += 2

    r0min = -Ea
    r0max = float(E0)
    while k < len(rho):
        r0 = r0min + np.clip(rho[k], 0.0, 1.0) * (r0max - r0min)
        if abs(r0) < 1e-3:
            r0 = 0.0
        poles[k] = -r0
        DeltaZ = np.polymul(DeltaZ, np.array([1.0, r0]))
        k += 1

    # Delta = reciprocal polynomial of DeltaZ (fliplr + strip leading zeros)
    Delta = np.real(DeltaZ[::-1])
    while len(Delta) > 1 and abs(Delta[0]) < 1e-14:
        Delta = Delta[1:]

    return Delta, DeltaZ, poles


def cp2par(DeltaZ, alpha: float = 0.0, beta: float = np.inf, n_pairs: int = None):
    """
    Inverse of par2cp: map a characteristic polynomial to [0,1]^n parameters.

    Port of MATLAB ``cp2par``.
    """
    DeltaZ = np.asarray(DeltaZ, dtype=float).ravel()
    if n_pairs is None:
        n_pairs = (len(DeltaZ) - 1) // 2

    poles = np.roots(DeltaZ)
    poles = admproj(poles, alpha, beta, 'z')

    # Sort by |imag| descending (complex pairs first)
    idx = np.argsort(np.abs(np.imag(poles)))[::-1]
    poles = poles[idx]

    shifted = bool(beta < 0)
    beta_abs = abs(float(beta))
    Ea = np.exp(-float(alpha))
    Eb = np.exp(-np.pi / beta_abs) if not np.isinf(beta_abs) and beta_abs > 0 else 0.0

    if not shifted and Eb > Ea and not np.isinf(beta_abs):
        beta_abs = np.inf
        Eb = 1.0

    r2min, r2max, E0 = r2range(Ea, Eb, shifted)

    rho = np.zeros(len(poles))
    k = 0

    for _ in range(n_pairs):
        d = np.real(np.poly(poles[k: k + 2]))
        r1, r2 = float(d[1]), float(d[2])
        rho[k] = (r2 - r2min) / max(r2max - r2min, 1e-14)
        r1min, r1max, _ = r1range(r2, Ea, beta_abs, shifted)
        rho[k + 1] = (r1 - r1min) / max(r1max - r1min, 1e-14)
        k += 2

    r0min = -Ea
    r0max = float(E0)
    while k < len(poles):
        r0 = float(-np.real(poles[k]))
        rho[k] = (r0 - r0min) / max(r0max - r0min, 1e-14)
        k += 1

    return np.clip(rho, 0.0, 1.0)


def guesspoles(poles, n_poles: int) -> np.ndarray:
    """
    Select N poles as an initial guess, preferring those with largest modulus.

    Port of MATLAB ``guesspoles``.
    """
    poles = np.atleast_1d(np.array(poles, dtype=complex))
    idx = np.argsort(np.abs(poles))[::-1]
    poles = list(poles[idx])

    p = np.zeros(n_poles, dtype=complex)
    i = 0
    while i < n_poles and poles:
        if abs(np.imag(poles[0])) > 1e-10:
            if i < n_poles - 1:
                p[i]     = poles[0]
                p[i + 1] = poles[1]
                i += 2
            poles = poles[2:]
        else:
            p[i] = poles[0]
            poles.pop(0)
            i += 1

    return p


# ---------------------------------------------------------------------------
# k2ksi – recover the ksi polynomial from a given controller
# ---------------------------------------------------------------------------

def k2ksi(plant, K, dK0=None, T=None):
    """
    Find the polynomial parameter ksi for a controller K.

    The controller is parameterised as::

        K(z) = (aDelta + d·ksi) / (bDelta − n·ksi)

    where ``n/d = recip(Pz)`` (conjugate reciprocal of the discretised plant)
    and ``aDelta, bDelta`` solve the Diophantine  ``n·aDelta + dd·bDelta = Delta``.

    Parameters
    ----------
    plant : (num, den) tuple or scipy.signal.lti
        Continuous-time SISO plant (P22 channel).
    K     : (num, den) tuple or scipy.signal.dlti
        Discrete-time controller.
    dK0   : array-like, optional
        Mandatory factor in the controller denominator (default: [1.0]).
    T     : float, optional
        Sampling period.  Required when K is a tuple.

    Returns
    -------
    ksi    : np.ndarray  — polynomial parameter (should be near-constant for
                          a controller obtained from par2cp/go_par2k)
    aDelta : np.ndarray  — Diophantine numerator solution
    bDelta : np.ndarray  — Diophantine denominator solution

    Port of MATLAB ``k2ksi``.
    """
    import scipy.signal as sig
    from directsd.polynomial.transforms import dtfm
    from directsd.polynomial.operations import striplz, recip
    from directsd.polynomial.diophantine import dioph

    # Extract T, K_num, K_den
    if isinstance(K, sig.dlti):
        T = K.dt
        tf_k = K.to_tf()
        K_num = np.atleast_1d(np.array(tf_k.num, float)).ravel()
        K_den = np.atleast_1d(np.array(tf_k.den, float)).ravel()
    elif isinstance(K, tuple):
        if T is None:
            raise ValueError("T must be provided when K is a (num, den) tuple")
        K_num = np.atleast_1d(np.array(K[0], float)).ravel()
        K_den = np.atleast_1d(np.array(K[1], float)).ravel()
    else:
        raise TypeError(f"Unsupported controller type: {type(K)}")

    if dK0 is None:
        dK0 = np.array([1.0])
    else:
        dK0 = np.atleast_1d(np.array(dK0, float)).ravel()

    # Discretise the plant
    if isinstance(plant, sig.lti):
        P_num = np.atleast_1d(np.array(plant.num, float)).ravel()
        P_den = np.atleast_1d(np.array(plant.den, float)).ravel()
    elif isinstance(plant, tuple):
        P_num = np.atleast_1d(np.array(plant[0], float)).ravel()
        P_den = np.atleast_1d(np.array(plant[1], float)).ravel()
    else:
        raise TypeError(f"Unsupported plant type: {type(plant)}")

    D22_num, D22_den = dtfm((P_num, P_den), T)
    D22_num = np.real(np.asarray(D22_num, float)).ravel()
    D22_den = np.real(np.asarray(D22_den, float)).ravel()

    # Characteristic polynomial DeltaZ = D22_den*K_den + D22_num*K_num
    DeltaZ = np.polyadd(np.polymul(D22_den, K_den), np.polymul(D22_num, K_num))
    DeltaZ = np.real(DeltaZ / DeltaZ[0])          # monic

    # Delta = recip(DeltaZ)
    Delta = np.real(DeltaZ[::-1])
    Delta = striplz(Delta) if len(Delta) > 1 else Delta

    # Pz' = conjugate reciprocal of Pz  →  for real coefficients: recip(Pz)
    n = striplz(D22_num[::-1])  # recip of D22_num
    d = striplz(D22_den[::-1])  # recip of D22_den

    # dd = d * dK0
    dd = np.polymul(d, dK0)

    # Diophantine: n*aDelta + dd*bDelta = Delta
    if len(n) - 1 <= len(d) - 1:
        X, Y, _, _ = dioph(n, dd, Delta)
        aDelta = np.real(np.asarray(
            X.coef if hasattr(X, 'coef') else X, float)).ravel()
        bDelta = np.real(np.asarray(
            Y.coef if hasattr(Y, 'coef') else Y, float)).ravel()
    else:
        X, Y, _, _ = dioph(dd, n, Delta)
        bDelta = np.real(np.asarray(
            X.coef if hasattr(X, 'coef') else X, float)).ravel()
        aDelta = np.real(np.asarray(
            Y.coef if hasattr(Y, 'coef') else Y, float)).ravel()

    aDelta = striplz(aDelta) if len(aDelta) > 1 else aDelta
    bDelta = striplz(bDelta) if len(bDelta) > 1 else bDelta

    # K' = recip(K) for real DT polynomial
    Kr_num = K_num[::-1]   # recip of K numerator
    Kr_den = K_den[::-1]   # recip of K denominator

    # ksiN = K'*bDelta - aDelta  (as TF numerator / denominator)
    # K'*bDelta  →  (polymul(Kr_num, bDelta), Kr_den)
    # -aDelta    →  (-aDelta, [1.0])
    # sum        →  (polymul(Kr_num,bDelta)*1 + (-aDelta)*Kr_den,  Kr_den)
    ksiN_num = np.polyadd(np.polymul(Kr_num, bDelta),
                          np.polymul(-aDelta, Kr_den))
    ksiN_den = Kr_den

    # ksiD = K'*n + d  (as TF)
    # K'*n  →  (polymul(Kr_num, n), Kr_den)
    # d     →  (d, [1.0])
    # sum   →  (polymul(Kr_num,n) + d*Kr_den, Kr_den)
    ksiD_num = np.polyadd(np.polymul(Kr_num, n),
                          np.polymul(d, Kr_den))
    ksiD_den = Kr_den

    # ksi = ksiN / ksiD  →  (ksiN_num * ksiD_den) / (ksiN_den * ksiD_num)
    ksi_num = np.polymul(ksiN_num, ksiD_den)
    ksi_den = np.polymul(ksiN_den, ksiD_num)

    # Simplify by GCD
    from directsd.polynomial.operations import gcd
    g = gcd(ksi_num, ksi_den)
    g_coef = np.asarray(g.coef if hasattr(g, 'coef') else g, float).ravel()
    if len(g_coef) > 1:
        ksi_num = np.real(np.polydiv(ksi_num, g_coef)[0])
        ksi_den = np.real(np.polydiv(ksi_den, g_coef)[0])

    ksi_num = striplz(np.real(ksi_num))
    ksi_den = striplz(np.real(ksi_den))

    # Polynomial long division: ksi should be a polynomial (near-zero remainder)
    ksi, err = np.polydiv(ksi_num, ksi_den)
    ksi = np.real(ksi)

    err_norm = np.linalg.norm(err)
    den_norm = np.linalg.norm(ksi_den)
    if err_norm > 1e-3 * den_norm:
        import warnings
        warnings.warn(
            f"k2ksi: cancellation error {err_norm:.4g} (den norm {den_norm:.4g}). "
            "ksi may not be a pure polynomial.",
            RuntimeWarning, stacklevel=2,
        )

    return ksi, aDelta, bDelta


# ---------------------------------------------------------------------------
# Controller-parameter target functions (closures, no global state)
# ---------------------------------------------------------------------------

def go_par2k(coef, ctx: dict):
    """
    Map a parameter vector coef ∈ [0,1]^n to a discrete controller.

    Parameters
    ----------
    coef : array-like — parameters in [0, 1]
    ctx  : dict with keys:
             alpha, beta, n_pairs, D22_num, D22_den
             (and optionally dK0 for forced denominator factor)

    Returns
    -------
    K_num, K_den : np.ndarray — controller numerator/denominator

    Simplified port of MATLAB ``go_par2k`` (ksi=0 path).
    """
    from directsd.polynomial.operations import striplz
    from directsd.polynomial.diophantine import dioph

    coef = np.asarray(coef, dtype=float).ravel()
    alpha   = float(ctx.get('alpha', 0.0))
    beta    = float(ctx.get('beta', np.inf))
    n_pairs = ctx.get('n_pairs', len(coef) // 2)

    Delta, DeltaZ, _ = par2cp(coef, alpha, beta, n_pairs)

    D22_num = np.asarray(ctx['D22_num'], dtype=float).ravel()
    D22_den = np.asarray(ctx['D22_den'], dtype=float).ravel()
    dK0     = np.asarray(ctx.get('dK0', [1.0]), dtype=float).ravel()

    dd = np.polymul(D22_den, dK0)
    n, d = D22_num, dd

    if len(n) <= len(d):
        X, Y, _, _ = dioph(n, d, Delta)
        aDelta = np.asarray(X.coef if hasattr(X, 'coef') else X, float).ravel()
        bDelta = np.asarray(Y.coef if hasattr(Y, 'coef') else Y, float).ravel()
    else:
        X, Y, _, _ = dioph(d, n, Delta)
        bDelta = np.asarray(X.coef if hasattr(X, 'coef') else X, float).ravel()
        aDelta = np.asarray(Y.coef if hasattr(Y, 'coef') else Y, float).ravel()

    ksi = np.array([0.0])
    K_num = np.polyadd(aDelta, np.polymul(np.polymul(D22_den, ksi), dK0))
    K_den = np.polymul(np.polyadd(bDelta, -np.polymul(D22_num, ksi)), dK0)

    K_num = np.real(striplz(K_num))
    K_den = np.real(striplz(K_den))
    return K_num, K_den


def f_sdh2p(coef, ctx: dict):
    """
    H2-norm target function for modal optimization.

    Parameters
    ----------
    coef : array-like — parameters in [0, 1]; or (K_num, K_den) tuple
    ctx  : dict with keys: plant, T, alpha, beta, n_pairs, D22_num, D22_den

    Returns
    -------
    f : float — H2-norm value
    K : (K_num, K_den) tuple

    Port of MATLAB ``f_sdh2p``.
    """
    from directsd.analysis.norms import sdh2norm

    if isinstance(coef, tuple) and len(coef) == 2:
        K = coef
    else:
        K = go_par2k(coef, ctx)

    plant = ctx['plant']
    T     = float(ctx['T'])
    try:
        f = float(sdh2norm(plant, K, T))
        if not np.isfinite(f):
            f = float('inf')
    except Exception:
        f = float('inf')
    return f, K


def f_sdl2p(coef, ctx: dict):
    """
    L2-error target function for modal optimization.

    Parameters
    ----------
    coef : array-like — parameters in [0, 1]; or (K_num, K_den) tuple
    ctx  : dict with keys: plant, T, alpha, beta, n_pairs, D22_num, D22_den

    Returns
    -------
    f : float — L2-error value
    K : (K_num, K_den) tuple

    Port of MATLAB ``f_sdl2p``.
    """
    from directsd.analysis.errors import sdl2err

    if isinstance(coef, tuple) and len(coef) == 2:
        K = coef
    else:
        K = go_par2k(coef, ctx)

    plant = ctx['plant']
    T     = float(ctx['T'])
    try:
        f = float(sdl2err(plant, K, T))
        if not np.isfinite(f):
            f = float('inf')
    except Exception:
        f = float('inf')
    return f, K


def go_sdh2p(x: float, ctx: dict):
    """
    H2 target for modal Hilbert-curve optimization.

    Parameters
    ----------
    x   : float in [0, 1] — Hilbert curve parameter
    ctx : dict with keys as for f_sdh2p, plus 'dim' and 'bits'

    Returns
    -------
    f, coef, K

    Port of MATLAB ``go_sdh2p``.
    """
    dim  = int(ctx['dim'])
    bits = int(ctx.get('bits', 8))
    coef = hilb2coord(float(x), dim, bits)
    f, K = f_sdh2p(coef, ctx)
    return f, coef, K


def go_sdl2p(x: float, ctx: dict):
    """
    L2 target for modal Hilbert-curve optimization.

    Port of MATLAB ``go_sdl2p``.
    """
    dim  = int(ctx['dim'])
    bits = int(ctx.get('bits', 8))
    coef = hilb2coord(float(x), dim, bits)
    f, K = f_sdl2p(coef, ctx)
    return f, coef, K


# ---------------------------------------------------------------------------
# Sasimplex – simulated annealing with Nelder-Mead
# ---------------------------------------------------------------------------

def sasimplex(func, x0, options=None):
    """
    Simulated annealing using the Nelder-Mead simplex method.

    Parameters
    ----------
    func    : callable — f(x) -> float
    x0      : array-like — initial guess
    options : dict, optional
        display, tol, maxFunEvals, dispIter, multiStep,
        startTemp, tempDecRate

    Returns
    -------
    x_best : np.ndarray
    y_best : float
    n_evals : int

    Port of MATLAB ``sasimplex``.
    """
    defaults = dict(
        display='off',
        tol=1e-5,
        maxFunEvals=10000,
        dispIter=100,
        multiStep=20,
        startTemp=100.0,
        tempDecRate=None,
    )
    if options:
        defaults = updateopt(defaults, options)
    opt = defaults

    display      = opt['display']
    tol          = float(opt['tol'])
    max_feval    = int(opt['maxFunEvals'])
    disp_iter    = int(opt['dispIter'])
    M            = int(opt['multiStep'])
    T0           = -float(opt['startTemp'])
    alpha_decay  = opt['tempDecRate']
    if alpha_decay is None:
        alpha_decay = -np.log(0.1) / 200.0

    verbosity = {'off': 0, 'final': 1, 'on': 2}.get(display, 0)

    x0 = np.asarray(x0, dtype=float).ravel()
    n = len(x0)
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5

    # Build initial simplex
    X = np.zeros((n, n + 1))
    y = np.zeros(n + 1)
    X[:, 0] = x0
    y[0] = func(x0)
    n_evals = 1

    for j in range(n):
        z = x0.copy()
        z[j] = z[j] * 1.05 if z[j] != 0 else 0.00025
        X[:, j + 1] = z
        y[j + 1] = func(z)   # fixed MATLAB bug: was feval(func, y)
        n_evals += 1

    x_best = x0.copy()
    y_best = y[0]
    T = T0
    iteration = 0

    while np.max(np.abs(X[:, 1:] - X[:, :1])) > tol:
        # Stochastic fluctuation for all but best
        order = np.argsort(y)
        y = y[order]; X = X[:, order]
        y_flu = np.concatenate([[y[0]],
                                y[1:] + T * np.log(np.random.rand(n) + 1e-300)])

        order2 = np.argsort(y_flu)
        y = y[order2]; X = X[:, order2]; y_flu = y_flu[order2]

        xbar = X[:, :n].mean(axis=1)

        # Reflection
        xr = (1 + rho) * xbar - rho * X[:, n]
        fr = func(xr)
        fr_flu = fr - T * np.log(np.random.rand() + 1e-300)
        n_evals += 1

        if fr_flu < y_flu[0]:
            # Try expansion
            xe = (1 + rho * chi) * xbar - rho * chi * X[:, n]
            fe = func(xe)
            fe_flu = fe - T * np.log(np.random.rand() + 1e-300)
            n_evals += 1
            if fe_flu < fr_flu:
                X[:, n] = xe; y[n] = fe
            else:
                X[:, n] = xr; y[n] = fr
        elif fr_flu < y_flu[n - 1]:
            X[:, n] = xr; y[n] = fr
        else:
            # Contraction
            if fr_flu < y_flu[n]:
                xc = (1 + psi * rho) * xbar - psi * rho * X[:, n]
            else:
                xc = (1 - psi) * xbar + psi * X[:, n]
            fc = func(xc)
            fc_flu = fc - T * np.log(np.random.rand() + 1e-300)
            n_evals += 1
            if fc_flu <= min(fr_flu, y_flu[n]):
                X[:, n] = xc; y[n] = fc
            else:
                for j in range(1, n + 1):
                    X[:, j] = X[:, 0] + sigma * (X[:, j] - X[:, 0])
                    y[j] = func(X[:, j])
                n_evals += n

        yi_min = np.min(y)
        if yi_min < y_best:
            y_best = yi_min
            x_best = X[:, np.argmin(y)].copy()

        iteration += 1
        if n_evals >= max_feval:
            break

        T = T0 * np.exp(-alpha_decay * iteration)

        if verbosity > 1 and iteration % disp_iter == 0:
            print(f"  sasimplex iter {iteration}: y_best={y_best:.5g}, T={T:.5g}")

    if verbosity > 0:
        print(f"sasimplex done: y_best={y_best:.5g}, evals={n_evals}")

    return x_best, y_best, n_evals


# ---------------------------------------------------------------------------
# Arandsearch – accelerated random search
# ---------------------------------------------------------------------------

def arandsearch(func, x0, options=None, constraint_func=None, proj_func=None):
    """
    Accelerated random search optimization (Appel et al. 2004).

    Parameters
    ----------
    func            : callable — f(x) -> float
    x0              : array-like — initial guess
    options         : dict, optional
    constraint_func : callable, optional — g(x) -> array; <=0 is feasible
    proj_func       : callable, optional — proj(x) -> x_projected

    Returns
    -------
    x_best : np.ndarray
    val_best : float
    n_evals : int

    Port of MATLAB ``arandsearch``.
    """
    defaults = dict(
        display='off',
        tol=1e-4,
        maxFunEvals=10000,
        dispIter=100,
        iniStep=0.1,
        multiStep=1,
        maxFail=10,
        decStepBy=0.5,
        maxSuccess=2,
        incStepBy=2.0,
        adaptRate=0.1,
    )
    if options:
        defaults = updateopt(defaults, options)
    opt = defaults

    verbosity = {'off': 0, 'final': 1, 'on': 2}.get(opt['display'], 0)

    x = np.asarray(x0, dtype=float).ravel().copy()
    n = len(x)

    if proj_func:
        x = proj_func(x)

    val_best = func(x)
    x_best = x.copy()
    n_evals = 1

    step         = float(opt['iniStep'])
    h            = step
    multi_step   = int(opt['multiStep'])
    max_fail     = int(opt['maxFail'])
    dec_step     = float(opt['decStepBy'])
    max_success  = int(opt['maxSuccess'])
    inc_step     = float(opt['incStepBy'])
    adapt_rate   = float(opt['adaptRate'])
    max_feval    = int(opt['maxFunEvals'])
    disp_iter    = int(opt['dispIter'])

    fail_count    = 0
    success_count = 0
    adapt_dir     = np.zeros(n)
    iteration     = 0

    while step > opt['tol']:
        iteration += 1
        if n_evals >= max_feval:
            break

        val_min = np.inf
        dx_best = np.zeros(n)
        for _ in range(multi_step):
            direction = adapt_dir + np.random.rand(n) - 0.5
            norm_d = np.linalg.norm(direction)
            if norm_d > 1e-14:
                direction /= norm_d
            dx = direction * h
            x_new = x + dx
            if proj_func:
                x_new = proj_func(x_new)
            if constraint_func and np.any(np.asarray(constraint_func(x_new)) > 0):
                continue
            val_new = func(x_new)
            n_evals += 1
            if val_new < val_min:
                val_min = val_new
                dx_best = x_new - x

        if not np.isinf(val_min):
            if val_min > -np.inf:
                adapt_dir = ((1 - adapt_rate) * adapt_dir
                             + adapt_rate * dx_best * np.sign(val_best - val_min))
                norm_ad = np.linalg.norm(adapt_dir)
                if norm_ad > 1e-14:
                    adapt_dir = adapt_rate * adapt_dir / norm_ad

        if val_min < val_best:
            val_best = val_min
            x = x + dx_best
            x_best = x.copy()
            success_count += 1
            fail_count = 0
            h = step
            if success_count >= max_success:
                step *= inc_step
                success_count = 0
                if verbosity > 1:
                    print(f"  arandsearch iter {iteration}: "
                          f"val={val_best:.5g} step={step:.4g} ^^")
        else:
            success_count = 0
            h *= dec_step
            if h < opt['tol']:
                fail_count += 1
                if fail_count >= max_fail:
                    step *= dec_step
                    fail_count = 0
                    if verbosity > 1:
                        print(f"  arandsearch iter {iteration}: "
                              f"val={val_best:.5g} step={step:.4g} vv")
                h = step

        if verbosity > 1 and iteration % disp_iter == 0:
            print(f"  arandsearch iter {iteration}: val={val_best:.5g} step={step:.4g}")

    if verbosity > 0:
        if n_evals >= max_feval:
            print(f"arandsearch: not converged in {max_feval} evals.")
        else:
            print("arandsearch: converged.")

    return x_best, val_best, n_evals


# ---------------------------------------------------------------------------
# infglob – information algorithm (1-D Lipschitz optimizer)
# ---------------------------------------------------------------------------

def infglob(func, options=None):
    """
    Global optimization of a scalar function on [0, 1] using Strongin's
    information algorithm.

    Parameters
    ----------
    func    : callable — f(x) -> float, x ∈ [0, 1]
    options : dict, optional
        display, tol, maxIter, dispIter, dim, r, ksi, guess

    Returns
    -------
    x_best : float
    z_best : float
    n_iter : int
    x_trace : list of float

    Port of MATLAB ``infglob``.
    """
    defaults = dict(
        display='off',
        tol=1e-4,
        maxIter=500,
        dispIter=10,
        dim=1,
        r=2.0,
        ksi=1.0,
        guess=[],
    )
    if options:
        defaults = updateopt(defaults, options)
    opt = defaults

    verbosity = {'off': 0, 'final': 1, 'on': 2}.get(opt['display'], 0)
    tol       = float(opt['tol'])
    max_iter  = int(opt['maxIter'])
    disp_iter = int(opt['dispIter'])
    n         = int(opt['dim'])
    r         = float(opt['r'])
    ksi       = float(opt['ksi'])
    guess     = list(opt.get('guess', []))

    x = sorted(set(list(np.arange(0, 1.1, 0.1)) + [float(g) for g in guess]))
    x = np.array(x, dtype=float)
    x = np.clip(x, 0.0, 1.0)
    z = np.array([func(xi) for xi in x], dtype=float)
    x_trace = list(x)
    k = len(x)

    best_i = int(np.argmin(z))
    z_best = float(z[best_i])
    x_best = float(x[best_i])

    lam = np.zeros(k + max_iter)

    iteration = 0
    while True:
        iteration += 1
        if iteration > max_iter:
            break

        # Sort
        order = np.argsort(x)
        x = x[order]; z = z[order]

        dxn  = (x[1:] - x[:-1]) ** (1.0 / n)
        dz   = z[1:] - z[:-1]
        adz  = np.abs(dz)
        with np.errstate(divide='ignore', invalid='ignore'):
            zDx = np.where(dxn > 1e-300, adz / dxn, 0.0)

        X   = np.max(dxn)
        mu  = np.max(zDx) if len(zDx) > 0 else 1.0
        gamma = mu * dxn / max(X, 1e-300)

        zDx_ext = np.concatenate([[zDx[0]], zDx, [zDx[-1]]])
        for i in range(k - 1):
            lam[i] = np.max(zDx_ext[i: i + 3])

        hat_L = np.maximum(np.maximum(gamma, lam[:k - 1]), ksi)
        rLx   = r * hat_L * dxn

        with np.errstate(divide='ignore', invalid='ignore'):
            R = (rLx
                 - 2 * (z[1:] + z[:-1])
                 + np.where(rLx > 1e-300, dz ** 2 / rLx, 0.0))

        t = int(np.argmax(R))

        with np.errstate(divide='ignore', invalid='ignore'):
            x_new = (x[t + 1] + x[t]) / 2.0 - (
                np.sign(dz[t]) * (adz[t] / max(hat_L[t], 1e-300)) ** n / (2.0 * r)
            )
        x_new = float(np.clip(x_new, 0.0, 1.0))
        z_new = float(func(x_new))

        if z_new < z_best:
            z_best = z_new
            x_best = x_new

        if verbosity > 1 and iteration % disp_iter == 0:
            print(f"  infglob iter {iteration}: x={x_best:.5f} f={z_best:.5g}")

        if abs(x[t + 1] - x[t]) < tol:
            break

        x = np.append(x, x_new)
        x_trace.append(x_new)
        z = np.append(z, z_new)
        k += 1

    if verbosity > 0:
        status = "not converged" if iteration > max_iter else "converged"
        print(f"infglob {status}: x={x_best:.5f}, f={z_best:.5g}, iter={iteration}")

    return x_best, z_best, iteration, x_trace


# ---------------------------------------------------------------------------
# infglobc – constrained information algorithm
# ---------------------------------------------------------------------------

def infglobc(func, options=None):
    """
    Constrained global optimization using Strongin's information algorithm.

    The function must return a list/array:
      - length < nConstr+1: some constraint is violated (returns partial values)
      - length == nConstr+1: all constraints satisfied; last element is objective

    Parameters
    ----------
    func    : callable — f(x) -> list of length 1..nConstr+1
    options : dict, optional
        dim, nConstr, tol, r, maxIter, dispIter, display

    Returns
    -------
    x_best, z_best, n_iter, x_trace

    Port of MATLAB ``infglobc``.
    """
    defaults = dict(
        display='off',
        tol=1e-4,
        maxIter=500,
        dispIter=50,
        dim=1,
        r=2.0,
        nConstr=1,
    )
    if options:
        defaults = updateopt(defaults, options)
    opt = defaults

    verbosity  = {'off': 0, 'final': 1, 'on': 2}.get(opt['display'], 0)
    tol        = float(opt['tol'])
    max_iter   = int(opt['maxIter'])
    disp_iter  = int(opt['dispIter'])
    n          = int(opt['dim'])
    r          = float(opt['r'])
    n_constr   = int(opt['nConstr'])

    eps_r = 0.1 * np.ones(n_constr)

    def _eval(xi):
        g = func(float(xi))
        if isinstance(g, (int, float)):
            g = [g]
        return list(g)

    x  = [0.0, 1.0]
    g0 = _eval(0.0)
    g1 = _eval(1.0)

    nu    = [len(g0), len(g1)]
    g_arr = [g0, g1]
    z     = [g0[-1], g1[-1]]
    x_trace = list(x)

    ind_feas = [i for i, ni in enumerate(nu) if ni == n_constr + 1]
    if ind_feas:
        z_best = min(z[i] for i in ind_feas)
        x_best = x[ind_feas[np.argmin([z[i] for i in ind_feas])]]
    else:
        z_best = np.inf
        x_best = None

    mu    = np.ones(n_constr + 1)
    k     = 2

    for iteration in range(1, max_iter + 1):
        # Sort by x
        order = np.argsort(x)
        x     = [x[i] for i in order]
        z     = [z[i] for i in order]
        nu    = [nu[i] for i in order]
        g_arr = [g_arr[i] for i in order]

        x_arr = np.array(x)
        z_arr = np.array(z)
        nu_arr = np.array(nu)

        # Compute mu for each constraint level
        z_ast = np.zeros(n_constr + 1)
        Iv = list(range(k))
        for v in range(1, n_constr + 2):
            if len(Iv) < 2:
                mu[v - 1] = 1.0
            else:
                mu_v = 0.0
                for ii in range(len(Iv) - 1):
                    for jj in range(ii + 1, len(Iv)):
                        xi_ii, xi_jj = x_arr[Iv[ii]], x_arr[Iv[jj]]
                        gi_ii = g_arr[Iv[ii]][v - 1] if len(g_arr[Iv[ii]]) >= v else np.inf
                        gi_jj = g_arr[Iv[jj]][v - 1] if len(g_arr[Iv[jj]]) >= v else np.inf
                        dx_ = abs(xi_jj - xi_ii)
                        if dx_ > 1e-300:
                            mu_v = max(mu_v,
                                       abs(gi_jj - gi_ii) / dx_ ** (1.0 / n))
                mu[v - 1] = max(mu_v, 1.0)

            Iv_next = [i for i in Iv if nu_arr[i] >= v + 1]
            if not Iv_next:
                z_ast[v - 1] = min(z_arr[i] for i in Iv)
                break
            else:
                z_ast[v - 1] = -eps_r[v - 1] if v - 1 < len(eps_r) else 0.0
            Iv = Iv_next

        # Compute R for each interval
        Delta_arr = (x_arr[1:] - x_arr[:-1]) ** (1.0 / n)
        R = np.zeros(k - 1)
        for i in range(k - 1):
            v = max(nu_arr[i], nu_arr[i + 1])
            rMu = r * mu[v - 1]
            rMuDelta = rMu * Delta_arr[i]
            dz_i = z_arr[i + 1] - z_arr[i]
            if nu_arr[i] == nu_arr[i + 1]:
                R[i] = (rMuDelta
                        + dz_i ** 2 / max(rMuDelta, 1e-300)
                        - 2 * (z_arr[i + 1] + z_arr[i] - 2 * z_ast[v - 1]))
            elif nu_arr[i] < nu_arr[i + 1]:
                R[i] = (2 * Delta_arr[i]
                        - 4 * (z_arr[i + 1] - z_ast[v - 1]) / max(rMu, 1e-300))
            else:
                R[i] = (2 * Delta_arr[i]
                        - 4 * (z_arr[i] - z_ast[v - 1]) / max(rMu, 1e-300))

        t = int(np.argmax(R))
        v_t = max(nu_arr[t], nu_arr[t + 1])
        dz_t = z_arr[t + 1] - z_arr[t]
        x_new = (x_arr[t + 1] + x_arr[t]) / 2.0
        if nu_arr[t] == nu_arr[t + 1]:
            x_new -= (np.sign(dz_t) * abs(dz_t) / max(mu[v_t - 1], 1e-300)
                      ) ** n / (2.0 * r)
        x_new = float(np.clip(x_new, 0.0, 1.0))

        g_new = _eval(x_new)
        nu_new = len(g_new)

        if nu_new == n_constr + 1:
            z_new = g_new[-1]
            if z_new < z_best:
                z_best = z_new
                x_best = x_new

        if verbosity > 1 and iteration % disp_iter == 0:
            if x_best is not None:
                print(f"  infglobc iter {iteration}: x={x_best:.6f} f={z_best:.5g}")

        if abs(x_arr[t + 1] - x_arr[t]) < tol:
            break

        x.append(x_new); x_trace.append(x_new)
        z.append(g_new[-1]); nu.append(nu_new)
        g_arr.append(g_new)
        k += 1

    if verbosity > 0:
        print(f"infglobc done: f={z_best:.5g}, iter={iteration}")

    return x_best, z_best, iteration, x_trace


# ---------------------------------------------------------------------------
# optglob – multi-run infglob with zooming
# ---------------------------------------------------------------------------

def optglob(func, options=None, limits=None):
    """
    Global optimization via repeated infglob with limit zooming.

    Parameters
    ----------
    func    : callable — f(x) -> float, x ∈ [0, 1]
    options : dict, optional
        Fields: tol, maxIter, maxLoop, display, bounds, decLim, incR, r
    limits  : np.ndarray (Ncoef × 2), optional — current search bounds
              (passed via context; if None, uses [[0,1]])

    Returns
    -------
    x_best, z_best, coef, n_iter, x_trace

    Port of MATLAB ``optglob``.
    """
    defaults = dict(
        display='off',
        tol=1e-4,
        tolLim=0.01,
        maxIter=100,
        maxLoop=20,
        bounds=False,
        decLim=2.5,
        incR=1.2,
        r=2.0,
    )
    if options:
        defaults = updateopt(defaults, options)
    opt = defaults

    verbosity = {'off': 0, 'final': 1, 'on': 2}.get(opt['display'], 0)
    max_loop  = int(opt['maxLoop'])
    dec_lim   = float(opt['decLim'])
    inc_r     = float(opt['incR'])
    check_bounds = bool(opt['bounds'])

    if limits is None:
        Lim = np.array([[0.0, 1.0]])
    else:
        Lim = np.asarray(limits, dtype=float).reshape(-1, 2)
    Lim0    = Lim.copy()
    N_coef  = Lim.shape[0]
    w       = np.zeros(N_coef)

    coef_best = None
    z_best    = np.inf
    x_best    = None
    x_trace   = []

    ig_opt = dict(opt)
    ig_opt['maxIter'] = int(opt['maxIter'])

    for loop in range(max_loop):
        if verbosity > 1:
            print(f"  optglob run {loop + 1}")

        xb, zb, _, xt = infglob(func, ig_opt)
        x_trace += xt

        if xb is not None and zb < z_best:
            z_best    = zb
            x_best    = xb
            coef_best = hilb2coord(xb, N_coef, int(ig_opt.get('bits', 8)))

        if loop + 1 >= max_loop:
            break

        # Zoom limits
        if coef_best is not None:
            if np.max(np.abs(Lim[:, 1] - Lim[:, 0])) < opt['tolLim']:
                break
            for i in range(N_coef):
                w[i] = (Lim[i, 1] - Lim[i, 0]) / dec_lim
                Lim[i, 0] = coef_best[i] - w[i]
                Lim[i, 1] = coef_best[i] + w[i]
        else:
            for i in range(N_coef):
                ci = (Lim[i, 1] + Lim[i, 0]) / 2.0
                w[i] = (Lim[i, 1] - Lim[i, 0]) / dec_lim
                Lim[i, 0] = ci - w[i]
                Lim[i, 1] = ci + w[i]

        if check_bounds:
            for i in range(N_coef):
                if Lim[i, 0] < Lim0[i, 0]:
                    Lim[i, 0] = Lim0[i, 0]; Lim[i, 1] = Lim0[i, 0] + 2 * w[i]
                if Lim[i, 1] > Lim0[i, 1]:
                    Lim[i, 1] = Lim0[i, 1]; Lim[i, 0] = Lim0[i, 1] - 2 * w[i]

        ig_opt['r'] = min(10.0, inc_r * ig_opt.get('r', 2.0))

    return x_best, z_best, coef_best, loop + 1, x_trace


# ---------------------------------------------------------------------------
# optglobc – constrained multi-run infglobc with zooming
# ---------------------------------------------------------------------------

def optglobc(func, options=None, limits=None):
    """
    Constrained global optimization via repeated infglobc with zooming.

    Parameters and returns as optglob, but func follows the infglobc
    convention (returns partial vector when constraints are violated).

    Port of MATLAB ``optglobc``.
    """
    defaults = dict(
        display='off',
        tol=1e-4,
        tolLim=0.01,
        maxIter=100,
        maxLoop=20,
        nConstr=1,
        bounds=False,
        decLim=2.0,
        incR=1.2,
        r=2.0,
    )
    if options:
        defaults = updateopt(defaults, options)
    opt = defaults

    verbosity = {'off': 0, 'final': 1, 'on': 2}.get(opt['display'], 0)
    max_loop  = int(opt['maxLoop'])
    dec_lim   = float(opt['decLim'])
    inc_r     = float(opt['incR'])
    check_bounds = bool(opt['bounds'])

    if limits is None:
        Lim = np.array([[0.0, 1.0]])
    else:
        Lim = np.asarray(limits, dtype=float).reshape(-1, 2)
    Lim0   = Lim.copy()
    N_coef = Lim.shape[0]
    w      = np.zeros(N_coef)

    coef_best = None
    z_best    = np.inf
    x_best    = None
    x_trace   = []

    ig_opt = dict(opt)

    for loop in range(max_loop):
        if verbosity > 1:
            print(f"  optglobc run {loop + 1}")

        xb, zb, _, xt = infglobc(func, ig_opt)
        x_trace += xt

        if xb is not None and zb < z_best:
            z_best    = zb
            x_best    = xb
            coef_best = hilb2coord(xb, N_coef, int(ig_opt.get('bits', 8)))

        if loop + 1 >= max_loop:
            break

        if coef_best is not None:
            if np.max(np.abs(Lim[:, 1] - Lim[:, 0])) < opt['tolLim']:
                break
            for i in range(N_coef):
                w[i] = (Lim[i, 1] - Lim[i, 0]) / dec_lim
                Lim[i, 0] = coef_best[i] - w[i]
                Lim[i, 1] = coef_best[i] + w[i]
        else:
            for i in range(N_coef):
                ci = (Lim[i, 1] + Lim[i, 0]) / 2.0
                w[i] = (Lim[i, 1] - Lim[i, 0]) / dec_lim
                Lim[i, 0] = ci - w[i]
                Lim[i, 1] = ci + w[i]

        if check_bounds:
            for i in range(N_coef):
                if Lim[i, 0] < Lim0[i, 0]:
                    Lim[i, 0] = Lim0[i, 0]; Lim[i, 1] = Lim0[i, 0] + 2 * w[i]
                if Lim[i, 1] > Lim0[i, 1]:
                    Lim[i, 1] = Lim0[i, 1]; Lim[i, 0] = Lim0[i, 1] - 2 * w[i]

        ig_opt['r'] = min(10.0, inc_r * ig_opt.get('r', 2.0))

    return x_best, z_best, coef_best, loop + 1, x_trace
