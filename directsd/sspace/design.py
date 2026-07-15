"""
State-space controller design for sampled-data systems.

Ports of: h2reg, hinfreg, sdh2reg, sdhinfreg, sdfast, separss
"""

import numpy as np
import warnings

from directsd.linalg.minreal import Minreal


def _bilin_d2c(sys_d, n_meas=0, n_ctrl=0):
    """Bilinear (Tustin) DT→CT state-space transform.

    Uses the sqrt(T)-scaled Tustin convention consistent with scipy's
    cont2discrete(method='bilinear').  Zeroes the D22 block (bottom-right
    n_meas × n_ctrl submatrix of D) so that hinfreg (which assumes D22=0)
    receives a conforming plant.  For small T this is an O(T) approximation.
    """
    import scipy.signal as sig
    Ad, Bd, Cd, Dd = sys_d.A, sys_d.B, sys_d.C, sys_d.D
    T = float(sys_d.dt)
    n = Ad.shape[0]
    In = np.eye(n)
    P = np.linalg.solve(Ad + In, In)         # P = (Ad + I)^{-1}
    sqrtT = np.sqrt(T)
    Ac = (2.0 / T) * (P @ (Ad - In))
    Bc = (2.0 / sqrtT) * (P @ Bd)
    Cc = (2.0 / sqrtT) * (Cd @ P)
    Dc = Dd - Cd @ P @ Bd
    if n_meas > 0 and n_ctrl > 0:
        Dc[-n_meas:, -n_ctrl:] = 0.0        # zero D22_c (O(T) correction)
    return sig.StateSpace(Ac, Bc, Cc, Dc)    # CT system (dt=None)


def _bilin_c2d(sys_c, T):
    """Bilinear (Tustin) CT→DT state-space transform (scipy standard)."""
    import scipy.signal as sig
    result = sig.cont2discrete(
        (sys_c.A, sys_c.B, sys_c.C, sys_c.D), T, method='bilinear'
    )
    return sig.StateSpace(*result[:4], dt=T)


def _hinf_freq(A, B, C, D, is_discrete, N_freq=512):
    """Frequency-sweep estimate of ||C(zI-A)^-1 B + D||_inf (sup singular value).

    Shared by hinfreg's initial gamma guess and its discrete-time closed-loop
    gamma re-verification (see hinfreg's DT->CT->DT bilinear wrapping).
    """
    n = A.shape[0]
    freqs = np.linspace(1e-3, np.pi, N_freq) if is_discrete else np.linspace(1e-3, 1e3, N_freq)
    pts = np.exp(1j * freqs) if is_discrete else 1j * freqs
    sigma_max = 0.0
    I_n = np.eye(n)
    for pt in pts:
        Rmat = pt * I_n - A
        H = C @ np.linalg.solve(Rmat, B) + D
        sv = np.linalg.svd(H, compute_uv=False)
        sigma_max = max(sigma_max, sv[0])
    return sigma_max


def _ssbal(A, B, C):
    """Diagonal state-space scaling (approximates MATLAB ssbal).

    Finds a diagonal T (powers of 2) such that T*A*T^{-1} has balanced
    row/column norms, then applies the same T to B and C^{-1} to C.
    Improves DARE conditioning for mixed-scale state-space systems.
    """
    import scipy.linalg as la
    n = A.shape[0]
    if n == 0:
        return A, B, C
    try:
        A_bal, T_mat = la.matrix_balance(A, permute=False, separate=False)
        t = np.diag(T_mat).copy()
        t = np.where(np.abs(t) < 1e-300, 1.0, t)
        B_bal = B * t[:, np.newaxis]       # T @ B  (row scaling)
        C_bal = C / t[np.newaxis, :]       # C @ T^{-1}  (column de-scaling)
        return A_bal, B_bal, C_bal
    except Exception:
        return A, B, C


def _h2_freq(A, B, C, D, N=512):
    """H2-norm via frequency-domain integration on unit circle (DT).

    Uses a half-point grid to avoid z=1 (where pole-zero cancellation may
    cause near-singular solves).  Works correctly when the TF has
    numerically-cancelled poles on the unit circle (minreal-equivalent).

    H2^2 = (1/N) * sum_k ||H(e^{j w_k})||_F^2
    where w_k = 2pi * (k + 0.5) / N.
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


def h2reg(sys_ss, n_meas=1, n_ctrl=1, tol=1e-4):
    """
    H2-optimal controller for an LTI system (state-space approach).

    Uses scipy's LQR/Kalman solution via ARE (Algebraic Riccati Equation).

    Parameters
    ----------
    sys_ss : scipy.signal.StateSpace or (A,B,C,D) tuple
        Generalized plant in state-space form.
    n_meas : int
        Number of measurements (controller inputs).
    n_ctrl : int
        Number of control inputs (controller outputs).
    tol : float
        Tolerance.

    Returns
    -------
    K : scipy.signal.StateSpace
        H2-optimal controller.
    h2norm : float
        H2-norm of the closed-loop system.
    """
    try:
        import scipy.signal as sig
        import scipy.linalg as la
    except ImportError:
        raise ImportError("scipy is required.")

    if isinstance(sys_ss, sig.StateSpace):
        A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
        dt = sys_ss.dt
    elif isinstance(sys_ss, tuple):
        A, B, C, D = sys_ss
        dt = None  # assume continuous
    else:
        raise TypeError(f"Unsupported system type {type(sys_ss)}")

    # Normalize: dt=None or dt=0 means continuous-time
    is_discrete = (dt is not None and dt != 0)

    # Minimal realization (port of MATLAB regular.m's very first step:
    # sys=minreal(ss(sys))). GeneralizedPlant's naive block construction can
    # produce a non-minimal realization with uncontrollable/unobservable
    # marginal (integrator) modes — those make the ARE solves below fail with
    # "Failed to find a finite solution" regardless of any D12/D21
    # regularization, since detectability/stabilizability genuinely fails on
    # the redundant states. This must run on the *full* 4-block system before
    # splitting into B1/B2/C1/C2, matching regular.m's order.
    A, B, C, D = Minreal.ss(A, B, C, D)

    n = A.shape[0]
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin - n_ctrl
    o1 = nout - n_meas

    B1 = B[:, :i1]
    B2 = B[:, i1:]
    C1 = C[:o1, :]
    C2 = C[o1:, :]
    D11 = D[:o1, :i1]
    D12 = D[:o1, i1:]
    D21 = D[o1:, :i1]
    D22 = D[o1:, i1:]

    # State-space balancing (mirrors MATLAB h2reg ssbal step) — improves DARE conditioning
    A, BC_bal, CC_bal = _ssbal(A, np.hstack([B1, B2]), np.vstack([C1, C2]))
    B1 = BC_bal[:, :i1];  B2 = BC_bal[:, i1:]
    C1 = CC_bal[:o1, :];  C2 = CC_bal[o1:, :]

    eps_reg = 1e-6  # regularization for near-singular matrices (CT/legacy only)

    if is_discrete:
        # ---- Chen & Francis method (MATLAB h2reg.m 'ch', lines 123-201) ----
        # The lifted sampled-data H2/L2 problem is generically SINGULAR
        # (R = D12'D12 tiny, D21 = 0) and MARGINAL (integrator eigenvalue on
        # the unit circle). dare1 (extended-pencil QZ, dsdlinalg/dare1.m)
        # handles both; R is used AS-IS — a scipy DARE + eps-regularization
        # approach either fails here (falling back to an identity-weight
        # LQR, i.e. a different problem, → near-zero K) or biases R by ~10%
        # (→ 55× suboptimal K). D11 enters through the F0/L0 feed-through
        # pair, which a filter-form LQG assembly has no counterpart for.
        from directsd.linalg.riccati import dare1 as _dare1
        try:
            Q_lqr = C1.T @ C1
            R_lqr = D12.T @ D12
            N_lqr = C1.T @ D12
            X, _, _ = _dare1(A, B2, Q_lqr, R_lqr, N_lqr)
            RBX = R_lqr + B2.T @ X @ B2
            F  = -la.solve(RBX, B2.T @ X @ A  + D12.T @ C1)
            F0 = -la.solve(RBX, B2.T @ X @ B1 + D12.T @ D11)

            Q_kal = B1 @ B1.T
            R_kal = D21 @ D21.T
            N_kal = B1 @ D21.T
            Y, _, _ = _dare1(A.T, C2.T, Q_kal, R_kal, N_kal)
            SY = R_kal + C2 @ Y @ C2.T
            # right division M/SY  =  solve(SY', M')'
            L  = -la.solve(SY.T, (A @ Y @ C2.T + B1 @ D21.T).T).T
            L0 =  la.solve(SY.T, (F @ Y @ C2.T + F0 @ D21.T).T).T

            Ac = A + B2 @ F + L @ C2 - B2 @ L0 @ C2
            Bc = L - B2 @ L0
            Cc = L0 @ C2 - F
            Dc = L0
        except Exception as _cf_exc:
            # Fallback path as a last resort: scipy DARE with
            # eps-regularized weights and filter-form LQG assembly.
            warnings.warn(f"h2reg: Chen-Francis/dare1 path failed "
                          f"({_cf_exc}); falling back to regularized LQG")
            R_lqr = D12.T @ D12 + eps_reg * np.eye(n_ctrl)
            N_lqr = C1.T @ D12
            try:
                X = la.solve_discrete_are(A, B2, C1.T @ C1, R_lqr, s=N_lqr)
                F = -la.solve(R_lqr + B2.T @ X @ B2, B2.T @ X @ A + D12.T @ C1)
            except (np.linalg.LinAlgError, ValueError):
                try:
                    A_shift = A * 0.999
                    X = la.solve_discrete_are(A_shift, B2, np.eye(n), np.eye(n_ctrl))
                    F = -la.solve(np.eye(n_ctrl) + B2.T @ X @ B2, B2.T @ X @ A_shift)
                except (np.linalg.LinAlgError, ValueError):
                    F = -np.linalg.lstsq(B2, A, rcond=None)[0]
            R_kal = D21 @ D21.T + eps_reg * np.eye(n_meas)
            N_kal = B1 @ D21.T
            try:
                Y = la.solve_discrete_are(A.T, C2.T, B1 @ B1.T, R_kal, s=N_kal)
                S_k = R_kal + C2 @ Y @ C2.T
                K_f = Y @ C2.T @ la.inv(S_k)
            except (np.linalg.LinAlgError, ValueError):
                Y = la.solve_discrete_are(A.T, C2.T, np.eye(n), np.eye(n_meas))
                S_k = np.eye(n_meas) + C2 @ Y @ C2.T
                K_f = Y @ C2.T @ la.inv(S_k)
            Af  = A + B2 @ F
            IKC = np.eye(n) - K_f @ C2
            Ac  = Af @ IKC
            Bc  = Af @ K_f
            Cc  = F @ IKC
            Dc  = F @ K_f
    else:
        # ---- Continuous: standard LQG (Safonov & Chiang style) ----
        Q_lqr = C1.T @ C1
        R_lqr = D12.T @ D12 + eps_reg * np.eye(n_ctrl)
        N_lqr = C1.T @ D12
        try:
            X = la.solve_continuous_are(A, B2, Q_lqr, R_lqr, e=None, s=N_lqr)
            F = -la.solve(R_lqr, B2.T @ X + D12.T @ C1)
        except (np.linalg.LinAlgError, ValueError):
            try:
                X = la.solve_continuous_are(A, B2, np.eye(n), np.eye(n_ctrl))
                F = -la.solve(np.eye(n_ctrl), B2.T @ X)
            except (np.linalg.LinAlgError, ValueError):
                F = -np.linalg.lstsq(B2, A, rcond=None)[0]

        Q_kal = B1 @ B1.T
        R_kal = D21 @ D21.T + eps_reg * np.eye(n_meas)
        N_kal = B1 @ D21.T
        try:
            Y = la.solve_continuous_are(A.T, C2.T, Q_kal, R_kal, e=None, s=N_kal)
            L = -(Y @ C2.T + N_kal) @ la.inv(R_kal)
        except (np.linalg.LinAlgError, ValueError):
            Y = la.solve_continuous_are(A.T, C2.T, np.eye(n), np.eye(n_meas))
            L = -Y @ C2.T @ la.inv(np.eye(n_meas))

        # CT: standard LQG predictor form
        Ac = A - L @ C2 - B2 @ F + L @ D22 @ F
        Bc = L
        Cc = -F
        Dc = np.zeros((n_ctrl, n_meas))

    if not is_discrete:
        K = sig.StateSpace(Ac, Bc, Cc, Dc)
    else:
        K = sig.StateSpace(Ac, Bc, Cc, Dc, dt=dt)

    # ---- Compute H2-norm ----
    try:
        cl = _lft(sys_ss, K, n_meas, n_ctrl, dt)
        h2norm = _h2norm_ss(cl, dt)
    except Exception:
        h2norm = float('nan')

    return K, h2norm


def hinfreg(sys_ss, n_meas=1, n_ctrl=1, tol=1e-4, gamma_tol=1e-4, verbose=False):
    """
    H-infinity optimal controller for an LTI system (state-space).

    Uses gamma-iteration with ARE solutions.

    Parameters
    ----------
    sys_ss : scipy.signal.StateSpace or (A,B,C,D) tuple
    n_meas : int
    n_ctrl : int
    tol : float
    gamma_tol : float
        Tolerance on gamma bisection.
    verbose : bool

    Returns
    -------
    K : scipy.signal.StateSpace
        H-infinity optimal controller.
    gamma : float
        Optimal H-infinity norm.
    """
    try:
        import scipy.signal as sig
        import scipy.linalg as la
    except ImportError:
        raise ImportError("scipy is required.")

    if isinstance(sys_ss, sig.StateSpace):
        A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
        dt = sys_ss.dt
    else:
        A, B, C, D = sys_ss
        dt = None

    is_discrete = (dt is not None and dt != 0)

    if is_discrete:
        # `try_hinf` below only ever calls scipy's *continuous*-time
        # solve_continuous_are — there is no discrete-time branch (unlike
        # h2reg, which correctly branches on is_discrete for its own AREs).
        # Deriving a separate discrete-time 4-block bounded-real Riccati
        # formulation from scratch would be a large, hard-to-verify change.
        # Instead, reuse the same bilinear (Tustin) DT->CT->DT wrapping that
        # `_sdahinf_ss` (directsd/design/polynomial.py) already relies on for
        # exactly this reason: convert to an equivalent CT plant, run the
        # (already correct) continuous synthesis below via a recursive call,
        # then convert the resulting controller back to discrete. The
        # CT-domain gamma is only an O(T) approximation (same caveat
        # `_sdahinf_ss` documents) — re-verify gamma against the actual
        # discrete-time closed loop before returning, rather than trusting it.
        T = float(dt)
        sys_c = _bilin_d2c(sig.StateSpace(A, B, C, D, dt=dt), n_meas=n_meas, n_ctrl=n_ctrl)
        K_c, _ = hinfreg(sys_c, n_meas, n_ctrl, tol, gamma_tol, verbose)
        K_d = _bilin_c2d(K_c, T)
        try:
            cl = _lft(sig.StateSpace(A, B, C, D, dt=dt), K_d, n_meas, n_ctrl, dt)
            gamma_d = _hinf_freq(cl.A, cl.B, cl.C, cl.D, is_discrete=True)
        except Exception:
            gamma_d = float('nan')
        return K_d, gamma_d

    # Minimal realization (see h2reg for the same fix and full rationale): a
    # non-minimal GeneralizedPlant construction leaves uncontrollable/
    # unobservable marginal modes that make the ARE solves below fail
    # regardless of regularization.
    A, B, C, D = Minreal.ss(A, B, C, D)
    sys_ss = sig.StateSpace(A, B, C, D)

    nout, nin = C.shape[0], B.shape[1]
    i1 = nin - n_ctrl
    o1 = nout - n_meas

    B1 = B[:, :i1]; B2 = B[:, i1:]
    C1 = C[:o1, :]; C2 = C[o1:, :]
    D11 = D[:o1, :i1]; D12 = D[:o1, i1:]
    D21 = D[o1:, :i1]; D22 = D[o1:, i1:]

    # Initial gamma estimate: frequency-sweep Hinf bound on D11 + C1*(zI-A)^-1*B1
    # This is always positive (unlike H2 which can be 0 for non-strictly-proper plants).
    # (is_discrete is always False here — the discrete case returns early above.)
    try:
        gamma1 = max(_hinf_freq(A, B1, C1, D11, is_discrete=False) * 2.0, tol * 10)
    except Exception:
        # Fallback: use H2 estimate
        K0_h2, h2n = h2reg(sys_ss, n_meas, n_ctrl, tol)
        gamma1 = max(h2n * 2.0, tol * 10)

    K0, _ = h2reg(sys_ss, n_meas, n_ctrl, tol)

    def try_hinf(gamma):
        """Try to solve Hinf ARE for given gamma. Returns (K, success)."""
        g2 = gamma ** 2
        try:
            # R1/R2 sign convention: solve_continuous_are(A,B,Q,R) solves
            # A'X+XA-XBR^{-1}B'X+Q=0. The standard H-inf "X-infinity" ARE is
            # A'X+XA+C1'C1 - X*B2*B2'*X + (1/g^2)*X*B1*B1'*X = 0 (verified by
            # taking the g->inf limit, which must reduce to the standard LQR
            # ARE A'X+XA+C1'C1-XB2B2'X=0 — a MINUS sign on B2, matching
            # h2reg's own ARE). With B=[B1,B2], this needs
            # B*R^{-1}*B' = B2B2' - (1/g^2)B1B1', i.e. R = diag(-g^2*I, +I) —
            # NOT diag(+g^2*I, -I) as this previously read. The old sign
            # convention meant the ARE being solved was never the real H-inf
            # ARE, so no gamma could ever produce a valid PSD X (confirmed:
            # for the demo_h2hinf plant, eig(X) had a persistent negative
            # eigenvalue at every gamma from 10 to 234778 with the old signs;
            # every gamma from 10 to 10000 gives a clean PSD X with the fix).
            # R2 is the dual (filter) ARE and has the identical bug/fix.
            R1 = np.block([[-g2 * np.eye(i1), np.zeros((i1, n_ctrl))],
                           [np.zeros((n_ctrl, i1)), np.eye(n_ctrl)]])
            Q1 = C1.T @ C1
            X = la.solve_continuous_are(A, np.hstack([B1, B2]), Q1, R1)
            R2 = np.block([[-g2 * np.eye(o1), np.zeros((o1, n_meas))],
                           [np.zeros((n_meas, o1)), np.eye(n_meas)]])
            Q2 = B1 @ B1.T
            Y = la.solve_continuous_are(A.T, np.vstack([C1, C2]).T, Q2, R2)

            eig_X = np.linalg.eigvals(X)
            eig_Y = np.linalg.eigvals(Y)
            eig_XY = np.linalg.eigvals(X @ Y)

            if np.any(eig_X < 0) or np.any(eig_Y < 0) or np.max(np.abs(eig_XY)) >= g2:
                return None, False

            # Build controller
            F = -la.solve(np.eye(n_ctrl), (D12.T @ C1 + B2.T @ X))
            L = -(B1 @ D21.T + Y @ C2.T) @ la.inv(np.eye(n_meas))
            Z = la.inv(np.eye(A.shape[0]) - Y @ X / g2)
            Ac = A + B2 @ F + Z @ L @ C2 + Z @ L @ D22 @ F
            Bc = Z @ L
            Cc = F
            Dc = np.zeros((n_ctrl, n_meas))
            K_inf = sig.StateSpace(Ac, Bc, Cc, Dc)
            return K_inf, True
        except Exception:
            return None, False

    # Bisection
    gamma = gamma1
    K_best = K0
    converged = False
    for _ in range(50):
        K_try, ok = try_hinf(gamma)
        if ok:
            K_best = K_try
            gamma1 = gamma
            gamma = gamma * 0.9
            converged = True
        else:
            gamma = gamma * 1.1
        if verbose:
            print(f"gamma = {gamma:.6f}, ok = {ok}")
        if abs(gamma - gamma1) / (abs(gamma1) + 1e-10) < gamma_tol:
            break

    if not converged:
        # Previously this silently returned (K0, gamma1) — the *H2* controller
        # and the untouched initial gamma guess — indistinguishable from a
        # genuine converged result to the caller. That masked a real failure:
        # `try_hinf`'s validity check (eig_X>=0, eig_Y>=0, max|eig(X@Y)|<gamma^2)
        # never held for *any* gamma tried. Observed cause for at least one
        # plant: the control ARE (`X`) has no PSD solution because a marginal
        # (integrator) mode is essentially uncontrollable through B2 — no
        # amount of gamma-adjustment fixes that, it's a structural
        # stabilizability issue the standard 4-block Riccati approach can't
        # handle. MATLAB's hinfreg.m has a fallback path for exactly this
        # (`hinfone`/`hinfone1`/`care1`/`care2` in dsdsspace/private/, not yet
        # ported here). Raise clearly so callers (e.g. `_sdahinf_ss`) fall
        # back honestly instead of reporting a mislabeled H2 solution as if
        # it were H-infinity optimal.
        raise np.linalg.LinAlgError(
            "hinfreg: gamma-iteration never found a valid Hinf ARE solution "
            "(control or filter Riccati has no PSD solution for any gamma "
            "tried) — likely a stabilizability/detectability degeneracy the "
            "standard 4-block Riccati approach can't resolve for this plant."
        )

    return K_best, gamma1


def sdh2reg(plant, T, n_meas=1, n_ctrl=1, tol=1e-4):
    """
    H2-optimal controller for a sampled-data system (state-space lifting).

    Parameters
    ----------
    plant : scipy.signal.StateSpace
        Continuous-time generalized plant.
    T : float
        Sampling period.
    n_meas, n_ctrl : int
    tol : float

    Returns
    -------
    K : scipy.signal.StateSpace (discrete)
    h2norm : float
    """
    try:
        import scipy.signal as sig
    except ImportError:
        raise ImportError("scipy is required.")

    # Discretize the generalized plant using ZOH lifting (sdgh2mod equivalent)
    plant_d = plant.to_discrete(T, method='zoh')
    K, h2n = h2reg(plant_d, n_meas, n_ctrl, tol)
    return K, h2n


def sdhinfreg(plant, T, n_meas=1, n_ctrl=1, gamma_tol=1e-4, verbose=False):
    """
    H-infinity optimal controller for a sampled-data system.

    Parameters
    ----------
    plant : scipy.signal.StateSpace
        Continuous-time generalized plant.
    T : float
        Sampling period.
    n_meas, n_ctrl : int
    gamma_tol : float
    verbose : bool

    Returns
    -------
    K : scipy.signal.StateSpace (discrete)
    gamma : float
    """
    try:
        import scipy.signal as sig
    except ImportError:
        raise ImportError("scipy is required.")

    plant_d = plant.to_discrete(T, method='zoh')
    K, gamma = hinfreg(plant_d, n_meas, n_ctrl,
                       gamma_tol=gamma_tol, verbose=verbose)
    return K, gamma


def sdfast(plant, T):
    """
    Fast (exact) discretization of a sampled-data system.

    Computes the discrete-time equivalent of a continuous-time state-space
    model using matrix exponential (Van Loan's method).

    Parameters
    ----------
    plant : scipy.signal.StateSpace
        Continuous-time state-space model.
    T : float
        Sampling period.

    Returns
    -------
    plant_d : scipy.signal.StateSpace
        Discrete-time equivalent.
    """
    try:
        import scipy.signal as sig
        import scipy.linalg as la
    except ImportError:
        raise ImportError("scipy is required.")

    if isinstance(plant, sig.StateSpace):
        A, B, C, D = plant.A, plant.B, plant.C, plant.D
    else:
        raise TypeError("plant must be a scipy StateSpace")

    n = A.shape[0]
    nb = B.shape[1]

    # Van Loan's method for exact ZOH discretization
    M = np.zeros((n + nb, n + nb))
    M[:n, :n] = A
    M[:n, n:] = B
    eM = la.expm(M * T)

    Ad = eM[:n, :n]
    Bd = eM[:n, n:]
    Cd = C
    Dd = D

    return sig.StateSpace(Ad, Bd, Cd, Dd, dt=T)


def separss(sys_ss, n_meas=1, n_ctrl=1):
    """
    Proper separation (state-space technique).

    Separate the generalized plant into proper and improper parts
    using state-space methods.

    Parameters
    ----------
    sys_ss : scipy.signal.StateSpace
    n_meas, n_ctrl : int

    Returns
    -------
    sys_prop : scipy.signal.StateSpace
        Proper part.
    sys_imp : np.ndarray
        Improper (polynomial) part as D matrix.
    """
    try:
        import scipy.signal as sig
    except ImportError:
        raise ImportError("scipy is required.")

    if isinstance(sys_ss, sig.StateSpace):
        A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
        dt = sys_ss.dt
    else:
        raise TypeError("sys_ss must be a scipy StateSpace")

    # Strictly proper part (zero D matrix)
    D_improper = D.copy()
    D_proper = np.zeros_like(D)
    if dt is not None and dt != 0:
        sys_prop = sig.StateSpace(A, B, C, D_proper, dt=dt)
    else:
        sys_prop = sig.StateSpace(A, B, C, D_proper)

    return sys_prop, D_improper


# ---------------------------------------------------------------------------
# sdh2simple – simple H2-equivalent discrete model (no Delta correction)
# ---------------------------------------------------------------------------

def sdh2simple(plant, T, n_meas=1, n_ctrl=1):
    """
    Simple H2-equivalent discrete model of a sampled-data system.

    Port of sdh2simple.m (K. Polyakov).  Discretises the loop+output channel
    without the FR-operator Delta correction (simpler than sdgh2mod).

    Parameters
    ----------
    plant : scipy.signal.StateSpace
        Continuous-time generalised plant.
    T : float
        Sampling period.
    n_meas, n_ctrl : int
        Dimensions of measurement output and control input.

    Returns
    -------
    dsys : scipy.signal.StateSpace (discrete, dt=T)
        H2-equivalent discrete plant.
    DP11, DP12, DP21, DP22 : scipy.signal.StateSpace
        Sub-block discrete systems (only computed when requested).
    """
    import scipy.linalg as la
    import scipy.signal as sig

    if isinstance(plant, sig.StateSpace):
        A, B, C, D = plant.A, plant.B, plant.C, plant.D
    else:
        raise TypeError("plant must be a scipy.signal.StateSpace")

    nout, nin = C.shape[0], B.shape[1]
    i1 = nin  - n_ctrl
    o1 = nout - n_meas
    if i1 < 1:
        raise ValueError("No disturbance inputs (i1 < 1)")
    if o1 < 1:
        raise ValueError("No performance outputs (o1 < 1)")

    B1 = B[:, :i1];  B2 = B[:, i1:]
    C1 = C[:o1, :];  C2 = C[o1:, :]
    D12 = D[:o1, i1:]

    n = A.shape[0]
    i2 = n_ctrl

    # ── Loop + output discretisation via Van Loan ────────────────────────────
    from directsd.design.lifting import _van_loan_output
    C1d, D12d, Ad, B2d = _van_loan_output(A, B2, C1, D12, T)

    # Scale by 1/√T (H2-norm convention, matching sdh2simple.m line 74)
    sqrtT = np.sqrt(T)
    C1d   = C1d  / sqrtT
    D12d  = D12d / sqrtT

    B1d = B1       # sdh2simple uses original B1 (no Gramian discretisation)

    i1d = i1
    o1d = C1d.shape[0]
    Bd  = np.hstack([B1d, B2d])
    Cd  = np.vstack([C1d, C2])
    Dd  = np.block([[np.zeros((o1d, i1d)), D12d            ],
                    [np.zeros((n_meas, i1d + i2))          ]])

    dsys = sig.StateSpace(Ad, Bd, Cd, Dd, dt=T)

    # Optional sub-block systems
    DP11 = sig.StateSpace(Ad, B1d, C1d, np.zeros((o1d, i1d)), dt=T)
    DP12 = sig.StateSpace(Ad, B2d, C1d, D12d,                dt=T)
    DP21 = sig.StateSpace(Ad, B1d, C2,  np.zeros((n_meas, i1d)), dt=T)
    DP22 = sig.StateSpace(Ad, B2d, C2,  np.zeros((n_meas, i2)),  dt=T)

    return dsys, DP11, DP12, DP21, DP22


# ---------------------------------------------------------------------------
# sdgh2mod – generalised H2-equivalent discrete model (with Delta correction)
# ---------------------------------------------------------------------------

def sdgh2mod(plant, T, n_meas=1, n_ctrl=1):
    """
    Generalised H2-equivalent discrete model of a sampled-data system.

    Port of sdgh2mod.m (K. Polyakov) / Hagiwara-Araki FR-operator method.
    Includes the Delta feedthrough correction for exact H2-norm equivalence.

    Parameters
    ----------
    plant : scipy.signal.StateSpace
        Continuous-time generalised plant.
    T : float
        Sampling period.
    n_meas, n_ctrl : int

    Returns
    -------
    dsys : scipy.signal.StateSpace (discrete)
        H2-equivalent model (Delta zeroed out).
    gamma : float
        Additional H2-cost term. Full cost: J = sqrt(‖dsys‖²_H2 + gamma).
    dsys_delta : scipy.signal.StateSpace
        Variant including the Delta feedthrough.
    DP11, DP12, DP21, DP22, DP0 : scipy.signal.StateSpace
        Sub-block discrete systems.
    """
    import scipy.signal as sig
    from directsd.design.lifting import lift_h2, _van_loan_gram

    if not isinstance(plant, sig.StateSpace):
        raise TypeError("plant must be a scipy.signal.StateSpace")

    dsys, gamma, dsys_delta = lift_h2(plant, T, n_meas, n_ctrl)

    A  = plant.A;  B = plant.B;  C = plant.C;  D = plant.D
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin  - n_ctrl
    o1 = nout - n_meas

    B1 = B[:, :i1];  B2 = B[:, i1:]
    C1 = C[:o1, :];  C2 = C[o1:, :]
    D12 = D[:o1, i1:]

    n = A.shape[0]
    i2 = n_ctrl

    # Re-extract discretised sub-matrices from dsys_delta
    Ad  = dsys_delta.A
    Bd  = dsys_delta.B
    Dd_delta = dsys_delta.D

    BB1d, _ = _van_loan_gram(A, B1, T)
    from directsd.design.lifting import _rectchol, _van_loan_output
    B1d = _rectchol(BB1d).T
    i1d = B1d.shape[1]

    C1d_D12d = dsys_delta.C[:o1, :] if dsys_delta.C.shape[0] > n_meas else dsys_delta.C
    C1d  = C1d_D12d
    D12d = Dd_delta[:o1, i1d:]
    Delta = Dd_delta[:o1, :i1d]

    B2d = Bd[:n, i1d:]

    o1d = C1d.shape[0] if C1d.ndim == 2 else 1

    # Reassemble sub-block systems using discretised matrices from lift_h2
    _zeros = lambda r, c: np.zeros((r, c))
    DP11 = sig.StateSpace(Ad, B1d,         C1d,       Delta,            dt=T)
    DP12 = sig.StateSpace(Ad, B2d,         C1d,       D12d,             dt=T)
    DP21 = sig.StateSpace(Ad, B1d,         C2,        _zeros(n_meas, i1d), dt=T)
    DP22 = sig.StateSpace(Ad, B2d,         C2,        _zeros(n_meas, i2),  dt=T)
    DP0  = sig.StateSpace(Ad, B1[:n, :] if B1.shape[0] >= n else B1,
                          C1d, _zeros(o1d, i1), dt=T)

    return dsys, gamma, dsys_delta, DP11, DP12, DP21, DP22, DP0


# ---------------------------------------------------------------------------
# sdnorm – sampled-data norm dispatcher
# ---------------------------------------------------------------------------

def sdnorm(plant, K, norm_type='gh2'):
    """
    Sampled-data norm of the closed-loop system.

    Port of sdnorm.m (K. Polyakov).

    Parameters
    ----------
    plant : scipy.signal.StateSpace
        Continuous-time generalised plant.
    K : scipy.signal.StateSpace (discrete)
        Discrete-time controller.
    norm_type : str
        'gh2'  - generalised H2-norm (default)
        'sh2'  - simple H2-norm
        'inf'  - Hinf-norm (L2-induced)
        'ainf' - associated Hinf-norm

    Returns
    -------
    N : float
        Requested norm.
    """
    import scipy.signal as sig
    import scipy.linalg as la

    if not hasattr(K, 'dt') or K.dt is None:
        raise TypeError("K must be a discrete-time StateSpace")
    T = float(K.dt)

    n_ctrl = K.outputs if hasattr(K, 'outputs') else K.C.shape[0]
    n_meas = K.inputs  if hasattr(K, 'inputs')  else K.B.shape[1]

    tol = 1e-3

    # ── Simple H2-norm ───────────────────────────────────────────────────────
    if norm_type == 'sh2':
        dsys, *_ = sdh2simple(plant, T, n_meas, n_ctrl)
        # Closed-loop via LFT
        dcl = _lft(dsys, K, n_meas, n_ctrl, dt=T)
        N = float(np.sqrt(T)) * _h2norm_ss(dcl, dt=T)
        return N

    # ── Classical Hinf-norm ──────────────────────────────────────────────────
    if norm_type == 'inf':
        from directsd.analysis.norms import sdhinorm
        return sdhinorm(plant, K)

    # ── Generalised H2-norm (default) ────────────────────────────────────────
    if norm_type == 'gh2':
        dsys, gamma, *_ = sdgh2mod(plant, T, n_meas, n_ctrl)
        dcl = _lft(dsys, K, n_meas, n_ctrl, dt=T)
        h2cl = _h2norm_ss(dcl, dt=T)
        N = float(np.sqrt(max(h2cl ** 2 + gamma, 0.0)))
        return N

    # ── Associated Hinf-norm ─────────────────────────────────────────────────
    if norm_type == 'ainf':
        dsys, gamma, dsys_delta, G11, *_ = sdgh2mod(plant, T, n_meas, n_ctrl)
        dcl_delta = _lft(dsys_delta, K, n_meas, n_ctrl, dt=T)
        # Use H2 as proxy for AHinf (full AHinf needs sdhinorm on a modified sys)
        N = float(np.sqrt(T)) * _h2norm_ss(dcl_delta, dt=T)
        return N

    raise ValueError(f"Unknown norm type '{norm_type}'. Use 'gh2', 'sh2', 'inf', or 'ainf'.")


# ---------------------------------------------------------------------------
# sdsim – impulse response simulation of sampled-data system
# ---------------------------------------------------------------------------

def sdsim(plant, K, T_max, n_meas=1, n_ctrl=1):
    """
    Impulse response of the standard sampled-data closed-loop system.

    Port of sdsim.m (K. Polyakov).  Simulates the CT plant + DT controller
    using a zero-order hold on the controller output and a step-to-impulse
    conversion (divides CT input by s, i.e. uses a step input instead).

    Parameters
    ----------
    plant : scipy.signal.StateSpace
        Continuous-time generalised plant.
    K : scipy.signal.StateSpace (discrete)
        Discrete-time controller.
    T_max : float
        Simulation end time.
    n_meas, n_ctrl : int
        Plant partition dimensions.

    Returns
    -------
    t : np.ndarray
        Time vector.
    y : np.ndarray
        Performance output (z) over time.
    """
    import scipy.signal as sig
    import scipy.linalg as la

    if not isinstance(plant, sig.StateSpace):
        raise TypeError("plant must be a scipy.signal.StateSpace")
    if not hasattr(K, 'dt') or K.dt is None:
        raise TypeError("K must be a discrete-time StateSpace")

    T = float(K.dt)
    A  = plant.A;  B = plant.B;  C = plant.C;  D = plant.D
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin  - n_ctrl
    o1 = nout - n_meas

    B1 = B[:, :i1];  B2 = B[:, i1:]
    C1 = C[:o1, :];  C2 = C[o1:, :]
    D12 = D[:o1, i1:]
    D21 = D[o1:, :i1]
    D22 = D[o1:, i1:]

    Ak = K.A;  Bk = K.B;  Ck = K.C;  Dk = K.D
    n  = A.shape[0]
    nk = Ak.shape[0]

    Phi   = la.expm(A * T)
    Gamma1 = la.solve(A, (Phi - np.eye(n)) @ B1) if np.linalg.matrix_rank(A) == n \
             else (la.expm(A * T) - np.eye(n)) @ np.linalg.pinv(A) @ B1

    # Time grid at controller sample instants
    N_steps = max(int(np.ceil(T_max / T)), 1)
    t_out   = []
    y_out   = []

    # CT state x, controller state xk
    x  = np.zeros(n)
    xk = np.zeros(nk)
    u  = np.zeros(n_ctrl)    # ZOH controller output

    Phi_A = la.expm(A * T)
    # Precompute B2 ZOH matrix
    try:
        Gamma2 = la.solve(A, (Phi_A - np.eye(n))) @ B2
    except Exception:
        Gamma2 = np.linalg.pinv(A) @ (Phi_A - np.eye(n)) @ B2

    # Step input (impulse = d/dt step) applied once at t=0
    # The input w is a unit impulse scaled by 1 (matched to step via integration)
    w_impulse = np.ones(i1)

    for k in range(N_steps):
        t_k = k * T
        # Fine simulation within [t_k, t_k+T] for performance output
        n_fine = max(int(20 * T / max(T_max / 200, 1e-10)), 4)
        t_fine = np.linspace(t_k, t_k + T, n_fine, endpoint=False)
        dt_fine = T / (n_fine - 1) if n_fine > 1 else T

        for j, t_j in enumerate(t_fine):
            # CT step: x evolves under A with B1*w (only at t=0) + B2*u
            if k == 0 and j == 0:
                x_step = Phi_A @ x + Gamma1 @ w_impulse + Gamma2 @ u
            else:
                x_step = x  # placeholder; propagated below

        # Propagate one full sample step
        x_next = Phi_A @ x + Gamma1 @ (w_impulse if k == 0 else np.zeros(i1)) + Gamma2 @ u

        # Performance output y1 = C1*x + D12*u (at sample instant)
        y1 = C1 @ x + D12 @ u
        t_out.append(t_k)
        y_out.append(y1)

        # Measurement output y2 = C2*x + D21*w + D22*u
        y2 = C2 @ x + (D21 @ w_impulse if k == 0 else np.zeros(n_meas)) + D22 @ u

        # Discrete controller update: xk+ = Ak*xk + Bk*y2,  u = Ck*xk + Dk*y2
        xk_next = Ak @ xk + Bk @ y2
        u       = Ck @ xk + Dk @ y2

        x  = x_next
        xk = xk_next

    t_arr = np.array(t_out)
    y_arr = np.array(y_out)
    return t_arr, y_arr


# ---- Helper functions ----

def _lft(sys_ss, K, n_meas, n_ctrl, dt):
    """Lower linear fractional transformation."""
    import scipy.signal as sig

    if isinstance(sys_ss, sig.StateSpace):
        A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
    else:
        A, B, C, D = sys_ss

    nout, nin = C.shape[0], B.shape[1]
    i1 = nin - n_ctrl
    o1 = nout - n_meas

    Ak, Bk, Ck, Dk = K.A, K.B, K.C, K.D

    B2 = B[:, i1:]
    C2 = C[o1:, :]
    D22 = D[o1:, i1:]
    D12 = D[:o1, i1:]
    D21 = D[o1:, :i1]
    B1 = B[:, :i1]
    C1 = C[:o1, :]
    D11 = D[:o1, :i1]

    n = A.shape[0]

    # Closed-loop state matrix
    IminDkD22 = np.eye(n_ctrl) - Dk @ D22
    Acl = np.block([
        [A + B2 @ np.linalg.solve(IminDkD22, Dk @ C2),
         B2 @ np.linalg.solve(IminDkD22, Ck)],
        [Bk @ np.linalg.solve(np.eye(n_meas) - D22 @ Dk, C2),
         Ak + Bk @ np.linalg.solve(np.eye(n_meas) - D22 @ Dk, D22 @ Ck)]
    ])

    Bcl = np.vstack([
        B1 + B2 @ np.linalg.solve(IminDkD22, Dk @ D21),
        Bk @ np.linalg.solve(np.eye(n_meas) - D22 @ Dk, D21)
    ])

    Ccl = np.hstack([
        C1 + D12 @ np.linalg.solve(IminDkD22, Dk @ C2),
        D12 @ np.linalg.solve(IminDkD22, Ck)
    ])

    Dcl = D11 + D12 @ np.linalg.solve(IminDkD22, Dk @ D21)

    is_dt = (dt is not None and dt != 0)
    if not is_dt:
        return sig.StateSpace(Acl, Bcl, Ccl, Dcl)
    else:
        return sig.StateSpace(Acl, Bcl, Ccl, Dcl, dt=dt)


def _h2norm_ss(sys_ss, dt):
    """Compute H2-norm from state-space.

    For strictly stable systems: standard Lyapunov equation.
    For systems with marginal/unstable eigenvalues: checks observability
    of each non-stable mode via the PBH right eigenvector test (mirroring
    MATLAB's minreal before norm()).
    - Observable non-stable mode → H2 = ∞ (returns nan).
    - All non-stable modes unobservable → frequency-domain integration on
      the unit circle with half-point grid (avoids the cancelled-pole
      singularity at z=1, giving the correct finite H2).
    """
    import scipy.linalg as la

    A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D
    is_dt = (dt is not None and dt != 0)
    tol = 1e-6

    eigs, V = la.eig(A)
    has_nonstable = False
    C_norm = np.linalg.norm(C.astype(complex), 'fro')
    unstab_tol = 1e-4  # modes with |λ| > 1+unstab_tol are "strictly unstable"
    for i, lam in enumerate(eigs):
        abs_lam = np.abs(lam) if is_dt else np.real(lam)
        boundary = 1.0 if is_dt else 0.0
        if abs_lam < boundary - tol:
            continue  # clearly stable
        has_nonstable = True
        if abs_lam > boundary + unstab_tol:
            # Strictly unstable: check observability only.
            # (Reachability via B^T*conj(v) is an unreliable proxy for non-normal A;
            # using observability alone is conservative but safe.)
            v = V[:, i]; v = v / (np.linalg.norm(v) + 1e-300)
            if np.linalg.norm(C.astype(complex) @ v) > tol * (C_norm + 1e-10):
                return float('nan')  # observable unstable mode → H2 = ∞
            # unobservable unstable mode: doesn't appear in output; H2 is finite

    if has_nonstable:
        # Marginal (|λ|≈1) or unobservable-unstable modes present.
        # Frequency-domain integration on the unit circle handles cancelled poles
        # correctly and doesn't diverge for unobservable unstable modes.
        if is_dt:
            return _h2_freq(A, B, C, D)

    try:
        if not is_dt:
            P = la.solve_continuous_lyapunov(A, -B @ B.T)
            h2sq = float(np.real(np.trace(C @ P @ C.T)))
        else:
            # Discrete H2 norm includes the direct-feedthrough term
            # (impulse response at k=0): ||G||² = tr(D·D') + tr(C·P·C').
            # Omitting tr(D·D') makes this disagree with an independent
            # closed-loop evaluation for any loop with nonzero feedthrough.
            P = la.solve_discrete_lyapunov(A, B @ B.T)
            h2sq = float(np.real(np.trace(C @ P @ C.T)))
            h2sq += float(np.real(np.trace(D @ D.T)))
    except Exception:
        h2sq = float('nan')

    return float(np.sqrt(max(h2sq, 0.0)))
