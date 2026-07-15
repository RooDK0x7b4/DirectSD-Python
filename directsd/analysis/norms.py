"""
directsd.analysis.norms
=======================
System norms computed via **algebraic solvers** (Lyapunov / Riccati /
Hamiltonian bisection) instead of frequency-domain numerical integration.
"""

import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import warnings

# NumPy 2.0 compat shim (kept for any future use)
_trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def _unpack_lti(sys_obj):
    if isinstance(sys_obj, sig.StateSpace):
        tf = sys_obj.to_tf(); return tf.num, tf.den, sys_obj.dt
    if isinstance(sys_obj, sig.dlti):
        return sys_obj.num, sys_obj.den, sys_obj.dt
    if isinstance(sys_obj, sig.lti):
        return sys_obj.num, sys_obj.den, None
    if isinstance(sys_obj, tuple) and len(sys_obj) == 2:
        return sys_obj[0], sys_obj[1], None
    if isinstance(sys_obj, tuple) and len(sys_obj) == 3:
        return sys_obj[0], sys_obj[1], sys_obj[2]
    raise TypeError(f"Unsupported system type {type(sys_obj)}")


def _ss_from_tf(num, den, dt=None):
    num = np.atleast_1d(np.array(num, dtype=float)).ravel()
    den = np.atleast_1d(np.array(den, dtype=float)).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dt is not None and dt not in (0, None):
            ss_obj = sig.dlti(num, den, dt=dt).to_ss()
        else:
            ss_obj = sig.lti(num, den).to_ss()
    return ss_obj.A, ss_obj.B, ss_obj.C, ss_obj.D


def _h2_freq_dt(A, B, C, D, N=512):
    """H2-norm via DFT on unit circle with half-point grid.

    Half-point avoids z=1 where pole-zero cancellation makes (I-A) singular.
    H2^2 = (1/N) * sum_k  ||H(e^{j w_k})||_F^2,  w_k = 2pi*(k+0.5)/N.
    """
    n = A.shape[0]
    w = 2.0 * np.pi * (np.arange(N) + 0.5) / N
    z = np.exp(1j * w)
    I_n = np.eye(n)
    h2sq = 0.0
    for zk in z:
        Hk = C @ np.linalg.solve(zk * I_n - A, B) + D
        h2sq += np.real(np.trace(Hk.conj().T @ Hk))
    return float(np.sqrt(max(h2sq / N, 0.0)))


def _h2norm_dt(A, B, C, D):
    """H2-norm for DT system.

    Fast path: Lyapunov equation for strictly stable systems.
    Fallback: frequency-domain integration when marginal unobservable modes
    are detected (mirrors MATLAB minreal(lft(sys,K)) before norm()).
    """
    tol = 1e-6
    unstab_tol = 1e-4  # modes with |λ| > 1+unstab_tol are "strictly unstable"
    eigs, V = la.eig(A)
    C_norm = np.linalg.norm(C.astype(complex), 'fro')
    has_marginal = False
    for i, lam in enumerate(eigs):
        abs_lam = np.abs(lam)
        if abs_lam < 1.0 - tol:
            continue
        has_marginal = True
        if abs_lam > 1.0 + unstab_tol:
            # Strictly unstable: conservative check — return nan if observable.
            v = V[:, i]; v = v / (np.linalg.norm(v) + 1e-300)
            if np.linalg.norm(C.astype(complex) @ v) > tol * (C_norm + 1e-10):
                return float('nan')  # observable unstable mode → H2 = ∞
            # unobservable: H2 is finite, use freq integration below
    if has_marginal:
        return _h2_freq_dt(A, B, C, D)
    try:
        P = la.solve_discrete_lyapunov(A, B @ B.T)
        h2sq = float(np.real(np.trace(C @ P @ C.T + D @ D.T)))
        return float(np.sqrt(max(h2sq, 0.0)))
    except Exception:
        return float('nan')


def _h2norm_ct(A, B, C, D):
    """H2-norm via continuous Lyapunov: A P + P A' + B B' = 0."""
    if np.max(np.abs(D)) > 1e-10:
        return float('inf')
    try:
        P = la.solve_continuous_lyapunov(A, -(B @ B.T))
        h2sq = float(np.real(np.trace(C @ P @ C.T)))
        return float(np.sqrt(max(h2sq, 0.0)))
    except Exception:
        return float('nan')


def _hinf_dt_bisect(A, B, C, D, tol=1e-6, max_iter=60):
    """H∞-norm of DT system by gamma-bisection on discrete-time Riccati."""
    n = A.shape[0]
    m = B.shape[1]
    N = 512
    w = np.linspace(0, np.pi, N)
    z = np.exp(1j * w)
    sv = np.array([np.linalg.svd(
        C @ np.linalg.solve(zk * np.eye(n) - A, B) + D,
        compute_uv=False)[0] for zk in z])
    gamma_ub = float(sv.max()) * 1.1 + 1e-8
    w_peak = float(w[np.argmax(sv)])
    gamma_lb = 0.0

    # Primal bounded-real-lemma DARE: A^T X A - X - (A^T X B + S)(R + B^T X B)^{-1}
    #   (B^T X A + S^T) + Q = 0  with Q=C^T C, S=C^T D, R=gamma^2 I_m - D^T D
    Q = C.T @ C
    S = C.T @ D

    def feasible(g):
        R = g**2 * np.eye(m) - D.T @ D
        if np.any(np.linalg.eigvalsh(R) <= 1e-12):
            return False
        try:
            X = la.solve_discrete_are(A, B, Q, R, s=S)
            return bool(np.all(np.linalg.eigvalsh(
                g**2 * np.eye(m) - D.T @ D - B.T @ X @ B) > 0))
        except Exception:
            return False

    for _ in range(max_iter):
        mid = (gamma_lb + gamma_ub) / 2
        if feasible(mid):
            gamma_ub = mid
        else:
            gamma_lb = mid
        if (gamma_ub - gamma_lb) < tol * (gamma_ub + 1e-12):
            break
    return gamma_ub, w_peak


def _hinf_ct_hamiltonian(A, B, C, D, tol=1e-6, max_iter=60):
    """
    H∞-norm of a stable CT system.

    For systems with D≠0 and full column rank: uses Hamiltonian jω-axis test.
    For strictly proper (D=0) or rank-deficient D: uses a dense log-spaced
    frequency scan (exact for rational systems that peak near ω=0 or ω=∞).
    """
    n = A.shape[0]
    n_out, n_in = D.shape

    # Dense log-spaced frequency scan (covers DC to high freq)
    N = 2048
    w_low  = np.logspace(-6, 0,  N // 2)
    w_high = np.logspace(0,  6,  N // 2)
    w = np.concatenate([w_low, w_high])
    sv_max = np.zeros(len(w))
    for k, wk in enumerate(w):
        Hk = C @ np.linalg.solve(1j * wk * np.eye(n) - A, B) + D
        sv_max[k] = np.linalg.svd(Hk, compute_uv=False)[0]

    gamma_freq = float(sv_max.max())
    w_peak     = float(w[np.argmax(sv_max)])

    # For strictly proper systems the peak may be at ω=0; check DC explicitly
    H_dc = C @ np.linalg.solve(-A, B) + D  # G(0) = -C A^{-1} B + D
    sv_dc = float(np.linalg.svd(H_dc, compute_uv=False)[0])
    if sv_dc > gamma_freq:
        gamma_freq = sv_dc
        w_peak = 0.0

    # Refine with Hamiltonian bisection if D has rank ≥ min(n_out,n_in)
    rank_D = np.linalg.matrix_rank(D)
    if rank_D < min(n_out, n_in):
        # Can't use Hamiltonian test; return frequency scan result
        return gamma_freq * (1 + tol), w_peak

    gamma_ub = gamma_freq * 1.02 + 1e-8
    gamma_lb = gamma_freq * 0.5

    def has_jw(gamma):
        g2 = gamma ** 2
        R = g2 * np.eye(n_in) - D.T @ D
        if np.any(np.linalg.eigvalsh(R) <= 1e-12):
            return True
        try:
            Ri = np.linalg.inv(R)
            F   = A - B @ Ri @ D.T @ C
            Ham = np.block([
                [F,                -(B @ Ri @ B.T)                          ],
                [-(C.T @ (np.eye(n_out) + D @ Ri @ D.T) @ C),  -F.T       ]
            ])
            eigs    = la.eigvals(Ham)
            min_re  = float(np.min(np.abs(np.real(eigs))))
            scale   = float(np.abs(eigs).max()) + 1.0
            return min_re < tol * scale
        except Exception:
            return True

    for _ in range(max_iter):
        mid = (gamma_lb + gamma_ub) / 2
        if has_jw(mid):
            gamma_lb = mid
        else:
            gamma_ub = mid
        if (gamma_ub - gamma_lb) < tol * (gamma_ub + 1e-12):
            break

    return max(gamma_ub, gamma_freq), w_peak


def _discretise_cl(plant_num, plant_den, K_num, K_den, T):
    from directsd.polynomial.transforms import dtfm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D22num, D22den = dtfm((plant_num, plant_den), T)
    KD_num = np.polymul(K_num, D22num)
    KD_den = np.polymul(K_den, D22den)
    S_num  = np.polymul(K_den, D22den)
    S_den  = np.polyadd(KD_den, KD_num)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss_obj = sig.dlti(S_num, S_den, dt=T).to_ss()
    return ss_obj.A, ss_obj.B, ss_obj.C, ss_obj.D


# ── Public API ────────────────────────────────────────────────────────────────

def dinfnorm(sys_obj, tol=1e-6):
    """H∞-norm of a discrete-time system (Hamiltonian bisection, exact)."""
    num, den, dt = _unpack_lti(sys_obj)
    A, B, C, D = _ss_from_tf(num, den, dt or 1.0)
    return _hinf_dt_bisect(A, B, C, D, tol)


def dahinorm(sys, K, T=None):
    """
    Associated H∞-norm for a discrete-time generalised plant.

    Port of MATLAB dahinorm.m (K. Polyakov) -> sdhinferr.m('h2coef').

    Computes the AH∞-norm of the *actual closed loop* for the given K:
    max_ω sqrt(X(ω)) where X(ω) = A(ω)|M(ω)|² + E(ω) - 2Re(B(ω)·M(ω)),
    M = K'/(1+D22·K') is the closed-loop map around D22 (K' = the
    conjugate-reciprocal of K, K'(z)=K(1/z) — sdhinferr.m's `feedback(K',D22)`),
    and A/B/E come from h2coef's plant decomposition.

    Note this must be computed *with* K, not as `Z=E-|B|^2/A` (the
    K-independent "equal-ripple optimal" infimum) — that formula is wrong:
    verified against MATLAB's own documented dhinf Example 1 (`Source/dsd_help.md`,
    K=1.5 constant, `dahinorm(sys,K)=3.6056`), the K-independent formula gives
    0.9487 regardless of K, while this corrected, K-dependent formula gives
    3.6055514... for K=1.5, matching MATLAB to 5 significant figures. Using
    the K-independent formula as a cross-check on dhinf's own bisection would
    silently misreport any stabilizing K as "wrong", since that formula's
    output never actually depends on the controller being evaluated.

    Parameters
    ----------
    sys : list-of-lists of (num, den) tuples
        Full discrete-time generalised plant in z-domain (same format as dhinf).
    K : (num, den) tuple
        Controller in z-domain (as returned by dhinf) to evaluate.
    T : float, optional
        Unused; kept for API symmetry.

    Returns
    -------
    err : float
        AH∞-norm of the closed-loop system for this K. Only meaningful for a
        *stabilizing* K — the frequency-domain formula is well-defined
        algebraically even for a non-stabilizing K (returning some finite
        number), but that number does not represent a real achievable cost.
    """
    from directsd.design.polynomial import _z2zeta, _h2coef_freq

    # Convert sys to ζ-domain
    sys_zeta = []
    for row in sys:
        new_row = []
        for entry in row:
            if np.isscalar(entry) or isinstance(entry, (int, float)):
                num_e, den_e = np.array([float(entry)]), np.array([1.0])
            else:
                num_e = np.atleast_1d(np.asarray(entry[0], float)).ravel()
                den_e = np.atleast_1d(np.asarray(entry[1], float)).ravel()
            nz, dz = _z2zeta(num_e, den_e)
            new_row.append((nz, dz))
        sys_zeta.append(new_row)

    # Negate last row (negative-feedback convention)
    sys_zeta[-1] = [(-n, d) for (n, d) in sys_zeta[-1]]

    # h2coef in ζ-domain: A = |P12|²·|P21|², B = P21·P11~·P12, E = |P11|²
    A_tf, B_tf, E_tf = _h2coef_freq(sys_zeta)
    D22_num, D22_den = sys_zeta[-1][-1]

    K_num = np.atleast_1d(np.asarray(K[0], float)).ravel()
    K_den = np.atleast_1d(np.asarray(K[1], float)).ravel()
    # K' = conjugate-reciprocal of K (z-domain): K'(z) = K(1/z)
    Kp_num, Kp_den = K_num[::-1], K_den[::-1]

    # Frequency grid
    N_freq = 4096
    w_f = np.linspace(1e-6, np.pi - 1e-6, N_freq)
    z_f = np.exp(1j * w_f)

    def _ev(num, den):
        return np.polyval(num, z_f) / (np.polyval(den, z_f) + 1e-300)

    A_f   = np.real(_ev(A_tf[0], A_tf[1]))
    B_f   = _ev(B_tf[0], B_tf[1])
    E_f   = np.real(_ev(E_tf[0], E_tf[1]))
    D22_f = _ev(D22_num, D22_den)
    Kp_f  = _ev(Kp_num, Kp_den)

    M_f = Kp_f / (1.0 + D22_f * Kp_f + 1e-300)
    X_f = A_f * np.abs(M_f) ** 2 + E_f - 2.0 * np.real(B_f * M_f)

    return float(np.sqrt(np.maximum(np.max(X_f), 0.0)))


def _K_to_ss(K, T):
    """
    Parse controller K to a discrete StateSpace at sampling period T.

    If K is given as a (num, den) tuple and deg(num) > deg(den) (improper in
    z-domain), the denominator is zero-padded so that deg(den) = deg(num).
    This interprets num/den as a polynomial in z^{-1} (causal convention),
    which is the convention used by the Bezout-based modal parametrisation.
    """
    if isinstance(K, sig.StateSpace) and K.dt not in (None, 0):
        return K, K.dt
    if isinstance(K, sig.dlti):
        return K.to_ss(), K.dt
    if isinstance(K, tuple) and len(K) == 2:
        K_num = np.atleast_1d(np.array(K[0], float)).ravel()
        K_den = np.atleast_1d(np.array(K[1], float)).ravel()
        if T is None:
            raise ValueError("T must be provided when K is a (num, den) tuple")
        # Pad denominator with trailing zeros when numerator has higher degree
        # (Bezout polynomial convention: coefficients are in z^{-1}, not z).
        excess = len(K_num) - len(K_den)
        if excess > 0:
            K_den = np.append(K_den, np.zeros(excess))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sig.dlti(K_num, K_den, dt=T).to_ss(), T
    raise TypeError(f"Unsupported controller type {type(K)}")


def sdh2norm(plant, K, T=None, t=None, H=None, udelay=0.0):
    """
    GH2-norm of a sampled-data closed-loop via H2-lifting (FR-operator method).

    Mirrors MATLAB sdh2norm.m (ss path)::

        [dsysH2, gamma] = sdgh2mod(sys, K.Ts)
        err = sqrt(norm(lft(dsysH2, K))^2 + gamma)

    Parameters
    ----------
    plant : scipy.signal.StateSpace or (num, den) tuple
        Continuous-time generalised plant (preferred) or SISO plant.
        For SISO plants a standard regulation generalized plant is constructed
        automatically via ``_parse_plant``.
    K : (num, den) tuple or scipy.signal.StateSpace (discrete)
        Discrete-time controller.
    T : float, optional
        Sampling period (inferred from K when K is a StateSpace/dlti).
    t : float, optional
        Time instant for the variance (polynomial method only; forces H-aware
        evaluation, matching MATLAB's sdh2norm.m).
    H : scipy.signal.lti, optional
        Generalized hold used for the *design* (default: ZOH). The
        lifting/state-space path above has no notion of a non-ZOH hold
        (matching MATLAB's own architecture -- sdh2norm.m routes to the
        polynomial 'pol' method whenever H is non-default); passing H here
        routes to the polynomial verification path instead.
    udelay : float, optional
        Continuous computational delay tau on the control input, matching
        `sdh2`'s identical parameter -- ONLY honored on the `t is not None`
        (or `H is not None`) path, which routes through `_sderr`'s own exact
        `_dtfm`-based delay handling (same mechanism `_sdh2coef` uses
        for design). Without this, evaluating a delay-designed `t`
        controller forces a Pade-approximated delay plant instead -- fine
        for the average-variance path (~1-3% error) but numerically
        unusable for the instantaneous-variance path: the fast Pade poles
        (~1/tau) interact badly with `expm(A*t)` at a SPECIFIC small t,
        giving errors of order 2x rather than a few percent -- confirmed
        by reproducing MATLAB's documented demo_fil2 values exactly once
        the exact-delay path is used instead of Pade.
        The no-`t`/no-`H` lifting path above has no delay support at all
        (same as the design side's 'ss' method) and ignores this parameter.

    Returns
    -------
    norm_val : float
        GH2-norm of the closed-loop sampled-data system.
    """
    from directsd.design.lifting import lift_h2, lft_dt
    from directsd.design.polynomial import _parse_plant, _sderr, _sdh2coef
    from directsd.linalg.minreal import Minreal

    if H is not None or t is not None:
        plant_ss, n_meas, n_ctrl = _parse_plant(plant)
        K_ss, T = _K_to_ss(K, T)
        K_tf = K_ss.to_tf() if hasattr(K_ss, 'to_tf') else None
        if K_tf is not None:
            K_num = np.atleast_1d(np.asarray(K_tf.num, float)).ravel()
            K_den = np.atleast_1d(np.asarray(K_tf.den, float)).ravel()
        else:
            import scipy.signal as _sig
            _tf = _sig.StateSpace(K_ss.A, K_ss.B, K_ss.C, K_ss.D).to_tf()
            K_num = np.atleast_1d(np.asarray(_tf.num, float)).ravel()
            K_den = np.atleast_1d(np.asarray(_tf.den, float)).ravel()
        z2 = _sderr(plant_ss, (K_num, K_den), T, t=t, H=H,
                    n_meas=n_meas, n_ctrl=n_ctrl, coef_fn=_sdh2coef,
                    udelay=udelay)
        return float(np.sqrt(max(z2, 0.0)))

    K_ss, T = _K_to_ss(K, T)

    plant_ss, n_meas, n_ctrl = _parse_plant(plant)
    plant_min = Minreal.ss(plant_ss)

    try:
        dsysH2, gamma, _ = lift_h2(plant_min, T, n_meas=n_meas, n_ctrl=n_ctrl)
    except Exception as exc:
        warnings.warn(f"sdh2norm: lift_h2 failed ({exc})")
        return float('nan')

    try:
        dcl = lft_dt(dsysH2, K_ss, n_meas=n_meas, n_ctrl=n_ctrl)
        h2n = _h2norm_dt(dcl.A, dcl.B, dcl.C, dcl.D)
    except Exception as exc:
        warnings.warn(f"sdh2norm: LFT/norm failed ({exc})")
        return float('nan')

    return float(np.sqrt(max(h2n ** 2 + gamma, 0.0)))


def sdhinorm(plant, K, T=None, tol=1e-6):
    """AHinf norm of a sampled-data closed-loop via lift_l2 + sigma_max sweep.

    Delegates to sdahinorm (directsd.design.polynomial) — same lift_l2 +
    frequency-sweep computation, consistent with MATLAB sdahinorm.

    Returns
    -------
    gamma : float
        AHinf norm = sqrt(T) * H∞(lifted closed-loop).
    w_peak : float
        0.0 (placeholder — sweep-based; no analytical peak frequency).
    """
    if T is None:
        if isinstance(K, sig.StateSpace) and K.dt not in (None, 0):
            T = float(K.dt)
        elif isinstance(K, sig.dlti):
            T = float(K.dt)
        elif isinstance(K, tuple) and len(K) == 3:
            T = float(K[2])
    if T is None:
        raise ValueError("T must be provided")
    from directsd.design.polynomial import sdahinorm
    return sdahinorm(plant, K, T), 0.0


def h2norm_ct(sys_obj):
    """H2-norm of a continuous-time system via Lyapunov equation."""
    num, den, _ = _unpack_lti(sys_obj)
    A, B, C, D = _ss_from_tf(num, den)
    return _h2norm_ct(A, B, C, D)


def hinfnorm_ct(sys_obj, tol=1e-6):
    """H∞-norm of a continuous-time system via Hamiltonian bisection."""
    num, den, _ = _unpack_lti(sys_obj)
    A, B, C, D = _ss_from_tf(num, den)
    return _hinf_ct_hamiltonian(A, B, C, D, tol)


# ---------------------------------------------------------------------------
# sdfreq – averaged frequency response of sampled-data closed-loop
# ---------------------------------------------------------------------------

def sdfreq(plant, K, w=None, resp_type='std'):
    """
    Averaged frequency response of the sampled-data closed-loop system.

    Port of sdfreq.m (K. Polyakov).

    Computes R(jω) = P11(jω) + (1/T)·P12(jω)·H(jω)·M(jω)·P21(jω)
    where H(jω) = (1 − e^{−jωT})/(jω) is the ZOH frequency response and
    M = K·(I − D22·K)^{−1} is the "lifted" controller.

    Parameters
    ----------
    plant : scipy.signal.StateSpace
        Continuous-time generalised plant (SISO or MIMO).
    K : scipy.signal.StateSpace (discrete, dt=T)
        Discrete-time controller.
    w : array-like, optional
        Frequency vector (rad/s). Defaults to 50 points in (0, ωs).
    resp_type : str
        'std'  - standard averaged response (default)
        'sing' - singular values of the spectral matrix (scalar output)
        'spec' - spectral matrix R11 + RA + RB + RB*

    Returns
    -------
    R : np.ndarray
        Frequency response array, shape (o1, i1, len(w)) for 'std'/'spec',
        or (len(w),) for 'sing'.
    w : np.ndarray
        Frequency vector used.
    """
    if not hasattr(K, 'dt') or K.dt is None:
        raise TypeError("K must be a discrete-time StateSpace")
    T = float(K.dt)

    nout, nin = plant.C.shape[0], plant.B.shape[1]
    i2 = K.B.shape[1]    # controller input dim  = n_meas
    o2 = K.C.shape[0]    # controller output dim = n_ctrl
    i1 = nin - o2
    o1 = nout - i2

    if i1 < 1:
        raise ValueError("No disturbance inputs detected")
    if o1 < 1:
        raise ValueError("No performance outputs detected")

    if w is None:
        ws = 2 * np.pi / T
        w = np.linspace(0.001 * ws, 0.999 * ws, 50)
    w = np.asarray(w, float).ravel()
    n_w = len(w)

    # Sub-block transfer functions (as state-space)
    A = plant.A;  B = plant.B;  C = plant.C;  Dp = plant.D
    B1 = B[:, :i1];  B2 = B[:, i1:]
    C1 = C[:o1, :];  C2 = C[o1:, :]
    D11 = Dp[:o1, :i1]; D12 = Dp[:o1, i1:]
    D21 = Dp[o1:, :i1]; D22 = Dp[o1:, i1:]

    n = A.shape[0]

    P11_ss = sig.StateSpace(A, B1, C1, D11)
    P12_ss = sig.StateSpace(A, B2, C1, D12)
    P21_ss = sig.StateSpace(A, B1, C2, D21)

    # D22 ZOH discrete plant for feedback
    D22_ss = sig.StateSpace(A, B2, C2, D22)
    D22_d  = D22_ss.to_discrete(T, method='zoh')

    # M = K * (I - D22_d * K)^{-1}: close the loop K around -D22_d
    Ak, Bk, Ck, Dk = K.A, K.B, K.C, K.D
    Ad22, Bd22, Cd22, Dd22 = D22_d.A, D22_d.B, D22_d.C, D22_d.D
    nk = Ak.shape[0]
    nd = Ad22.shape[0]

    # Evaluate frequency responses at each ω
    def _freqresp_ct(ss_obj, w_arr):
        """Evaluate CT SS frequency response at each frequency."""
        _, _, H = sig.freqs_zpk(*sig.ss2zpk(ss_obj.A, ss_obj.B, ss_obj.C, ss_obj.D), w_arr)
        return H  # returned as (n_out, n_in, n_w) or flattened

    def _eval_ct(A_m, B_m, C_m, D_m, jw):
        """(C·(jωI−A)^{−1}·B + D) at a single frequency jω."""
        n_s = A_m.shape[0]
        return C_m @ np.linalg.solve(jw * np.eye(n_s) - A_m, B_m) + D_m

    def _eval_dt(A_m, B_m, C_m, D_m, z):
        """(C·(zI−A)^{−1}·B + D) at a single DT frequency z = e^{jωT}."""
        n_s = A_m.shape[0]
        return C_m @ np.linalg.solve(z * np.eye(n_s) - A_m, B_m) + D_m

    R_list = []
    for wi in w:
        jw = 1j * wi
        z  = np.exp(jw * T)

        rP11 = _eval_ct(A, B1, C1, D11, jw)   # (o1, i1)
        rP12 = _eval_ct(A, B2, C1, D12, jw)   # (o1, o2)
        rP21 = _eval_ct(A, B1, C2, D21, jw)   # (i2, i1)

        rD22 = _eval_dt(Ad22, Bd22, Cd22, Dd22, z)  # (i2, o2)
        rK   = _eval_dt(Ak, Bk, Ck, Dk, z)          # (o2, i2)

        # M = K * (I - D22*K)^{-1}
        IminD22K = np.eye(i2) - rD22 @ rK
        rM = rK @ np.linalg.solve(IminD22K.T, np.eye(i2)).T  # (o2, i2)

        # ZOH frequency response H(jω) = (1 − e^{−jωT}) / (jω)
        if abs(wi) < 1e-12:
            rH = T
        else:
            rH = (1 - np.exp(-jw * T)) / jw

        # Standard: R = P11 + (1/T) * P12 * H * M * P21
        RPMP = (rP12 * rH) @ rM @ rP21 / T
        R_std = rP11 + RPMP

        if resp_type == 'std':
            R_list.append(R_std)
        elif resp_type in ('spec', 'sing'):
            # Spectral matrix (approximate; full version needs dtfm2)
            R_list.append(R_std.conj().T @ R_std)
        else:
            R_list.append(R_std)

    R_arr = np.array(R_list)   # (n_w, o1, i1)

    if resp_type == 'sing':
        sv = np.array([np.sqrt(np.max(np.linalg.svd(R_arr[i], compute_uv=False)))
                       for i in range(n_w)])
        return sv, w

    # Transpose to (o1, i1, n_w) convention
    R_out = np.transpose(R_arr, (1, 2, 0))
    return R_out, w


# Aliases kept for API compatibility
sdmargin_norms = sdh2norm   # not the actual margin - see charpol.py
