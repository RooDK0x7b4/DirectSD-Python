"""
directsd.design.convex
======================
Convex optimisation-based controller synthesis for sampled-data systems.

Uses CVXPY to formulate and solve problems that have no closed-form
polynomial/Riccati solution:

    • L1 (peak-to-peak) optimal design
    • Mixed H2 / L1 design (energy + amplitude)
    • Template / envelope matching (hard time-domain constraints)
    • General Q-parameterisation (Youla) basis for custom objectives

All methods follow the same lifting → Youla → convex program workflow:

    1. Lift the continuous plant to an exact discrete equivalent via lift_h2.
    2. Compute a stabilising base controller (H2-optimal via h2reg).
    3. Parameterise all stabilising controllers via a stable FIR parameter Q.
    4. Write the closed-loop map as a *linear* function of Q.
    5. Minimise the desired norm / cost subject to constraints.

Install the optional dependency before use::

    pip install cvxpy

References
----------
[1] Dahleh & Diaz-Bobillo, "Control of Uncertain Systems: A Linear
    Programming Approach," Prentice-Hall, 1995.
[2] Chen & Francis, "Optimal Sampled-Data Control Systems," Springer, 1995.
[3] Khammash, "A new approach to the solution of the l1 control problem:
    the scaled-Q method," IEEE TAC, 2000.
"""

import warnings

import numpy as np
import scipy.linalg as la
import scipy.signal as sig

# ── optional CVXPY import ────────────────────────────────────────────────────
try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


def _require_cvxpy():
    """Raise ImportError if cvxpy is not installed."""
    if not _CVXPY_AVAILABLE:
        raise ImportError(
            "cvxpy is required for convex synthesis.\n"
            "Install it with:  pip install cvxpy"
        )


# ── Toeplitz helpers ─────────────────────────────────────────────────────────

def _build_toeplitz_cvxpy(q_vec, N_fir, nc, nm):
    """
    Build a CVXPY lower-triangular block-Toeplitz matrix from FIR coefficients.

    Parameters
    ----------
    q_vec : cp.Variable  shape (N_fir * nc * nm,)
    N_fir : int
    nc    : int  n_ctrl
    nm    : int  n_meas

    Returns
    -------
    Q_mat : cp.Expression  shape (N_fir*nc, N_fir*nm)
    """
    blocks = []
    for i in range(N_fir):
        row_blocks = []
        for j in range(N_fir):
            lag = i - j
            if lag < 0:
                row_blocks.append(np.zeros((nc, nm)))
            else:
                start = lag * nc * nm
                q_block = cp.reshape(
                    q_vec[start: start + nc * nm], (nc, nm), order='C'
                )
                row_blocks.append(q_block)
        blocks.append(row_blocks)
    return cp.bmat(blocks)


def _build_toeplitz_np(Q_fir, nc, nm):
    """
    Build a numpy lower-triangular block-Toeplitz matrix from solved Q.

    Parameters
    ----------
    Q_fir : np.ndarray  shape (N_fir, nc, nm)
    nc, nm : int

    Returns
    -------
    T : np.ndarray  shape (N_fir*nc, N_fir*nm)
    """
    N_fir = Q_fir.shape[0]
    T = np.zeros((N_fir * nc, N_fir * nm))
    for i in range(N_fir):
        for j in range(N_fir):
            lag = i - j
            if lag >= 0:
                T[i*nc:(i+1)*nc, j*nm:(j+1)*nm] = Q_fir[lag]
    return T


# ── Impulse-response helpers ─────────────────────────────────────────────────

def _impulse_response(A, B, C, D, N):
    """
    Compute the first N samples of the discrete impulse response.

    Returns h : (N, n_out, n_in) array.
    """
    n_out, n_in = D.shape
    h = np.zeros((N, n_out, n_in))
    h[0] = D
    x = B.copy()
    for k in range(1, N):
        h[k] = C @ x
        x = A @ x
    return h


def _ss_impulse_matrix(A, B, C, D, N):
    """
    Build the lower-triangular convolution (Toeplitz) matrix of shape
    (N*n_out, N*n_in) such that vec(y) = T_cl @ vec(u).
    """
    h = _impulse_response(A, B, C, D, N)
    N, p, m = h.shape
    T = np.zeros((N * p, N * m))
    for i in range(N):
        for j in range(i + 1):
            T[i*p:(i+1)*p, j*m:(j+1)*m] = h[i - j]
    return T


# ── Youla internals ──────────────────────────────────────────────────────────

def _stabilising_h2(dsys, n_meas, n_ctrl):
    """Return a base controller K0 for Youla parameterization.

    Starts from h2reg's H2-optimal K0. For most plants this already
    strictly stabilizes the plant and is returned unchanged. Some plants
    make h2reg's Chen-Francis discrete GH2 solve "generically SINGULAR"
    (per h2reg's own comments), leaving a closed-loop eigenvalue sitting
    right at |z|=1 instead of strictly inside -- which breaks the Youla-
    Kucera guarantee ("any stable Q gives a stabilizing K") this whole
    module depends on. For that case only, retry with a "prescribed degree
    of stability" discount: solve the regulator/filter Riccati equations on
    (alpha*A, alpha*B2) / (alpha*A, alpha*C2) instead of (A, B2) / (A, C2)
    -- a classical LQG pole-shrinking technique -- then reassemble the
    compensator with the ORIGINAL (undiscounted) plant matrices, exactly as
    h2reg's own Chen-Francis formulas do internally.

    This retry is numerically fragile in practice (verified: closed-loop
    pole location and H2 cost are NOT monotonic in alpha for at least one
    real benchmark plant -- some discount factors land on a well-behaved,
    near-optimal design, nearby ones land on a degenerate or much worse
    one), so this tries several candidate discount factors and keeps only
    the one with the lowest independently-verified H2 cost among those
    that are genuinely, robustly stable (not just barely inside the unit
    circle). If none of the candidates achieve real strict stability, falls
    back to returning the original (possibly marginal) K0 unchanged --
    downstream callers still have the honest _verify_stabilizes check.
    """
    from directsd.sspace.design import h2reg, _lft

    K0, _ = h2reg(dsys, n_meas=n_meas, n_ctrl=n_ctrl)

    def _closed_loop_margin(K):
        dcl = _lft(dsys, K, n_meas, n_ctrl, dt=dsys.dt)
        return dcl, 1.0 - float(np.max(np.abs(np.linalg.eigvals(dcl.A))))

    dcl0, margin0 = _closed_loop_margin(K0)
    STRICT_MARGIN = 1e-3  # "genuinely, robustly stable", not just barely <1
    if margin0 >= STRICT_MARGIN:
        return K0

    K_discounted = _stabilising_h2_discounted(dsys, n_meas, n_ctrl, STRICT_MARGIN)
    return K_discounted if K_discounted is not None else K0


def _stabilising_h2_discounted(dsys, n_meas, n_ctrl, strict_margin):
    """Search a range of LQG discount factors for a strictly-stabilizing K0.

    Returns None (caller falls back to the marginal h2reg K0) if no
    candidate achieves the requested stability margin.
    """
    import scipy.linalg as la
    from directsd.linalg.minreal import Minreal
    from directsd.sspace.design import _ssbal, _lft
    from directsd.linalg.riccati import dare1 as _dare1

    A, B, C, D = Minreal.ss(dsys.A, dsys.B, dsys.C, dsys.D)
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin - n_ctrl
    o1 = nout - n_meas
    B1 = B[:, :i1]; B2 = B[:, i1:]
    C1 = C[:o1, :]; C2 = C[o1:, :]
    D11 = D[:o1, :i1]; D12 = D[:o1, i1:]
    D21 = D[o1:, :i1]

    A, BC_bal, CC_bal = _ssbal(A, np.hstack([B1, B2]), np.vstack([C1, C2]))
    B1 = BC_bal[:, :i1]; B2 = BC_bal[:, i1:]
    C1 = CC_bal[:o1, :]; C2 = CC_bal[o1:, :]

    def build(alpha):
        A_s, B2_s, C2_s = alpha * A, alpha * B2, alpha * C2
        Q_lqr, R_lqr, N_lqr = C1.T @ C1, D12.T @ D12, C1.T @ D12
        X, _, _ = _dare1(A_s, B2_s, Q_lqr, R_lqr, N_lqr)
        RBX = R_lqr + B2_s.T @ X @ B2_s
        F  = -la.solve(RBX, B2_s.T @ X @ A_s + D12.T @ C1)
        F0 = -la.solve(RBX, B2_s.T @ X @ (alpha * B1) + D12.T @ D11)

        Q_kal, R_kal, N_kal = B1 @ B1.T, D21 @ D21.T, B1 @ D21.T
        Y, _, _ = _dare1(A_s.T, C2_s.T, Q_kal, R_kal, N_kal)
        SY = R_kal + C2_s @ Y @ C2_s.T
        L  = -la.solve(SY.T, (A_s @ Y @ C2_s.T + (alpha * B1) @ D21.T).T).T
        L0 =  la.solve(SY.T, (F @ Y @ C2_s.T + F0 @ D21.T).T).T

        # Reassemble with the ORIGINAL (undiscounted) plant matrices --
        # alpha only reshapes the Riccati solves above, not the physical
        # plant the compensator actually has to run against.
        Ac = A + B2 @ F + L @ C2 - B2 @ L0 @ C2
        Bc = L - B2 @ L0
        Cc = L0 @ C2 - F
        Dc = L0
        return sig.StateSpace(Ac, Bc, Cc, Dc, dt=dsys.dt)

    def plain_h2_cost(dcl):
        # Independent Lyapunov-based H2 check -- deliberately bypasses
        # sdnorm's PBH-based marginal-mode special-casing, which is itself
        # unreliable in exactly this near-marginal regime.
        try:
            P = la.solve_discrete_lyapunov(dcl.A, dcl.B @ dcl.B.T)
            h2sq = np.trace(dcl.C @ P @ dcl.C.T) + np.trace(dcl.D @ dcl.D.T)
            return float(np.sqrt(h2sq)) if h2sq > 0 and np.isfinite(h2sq) else float('inf')
        except (np.linalg.LinAlgError, ValueError):
            return float('inf')

    best_K, best_cost = None, float('inf')
    for exp in range(24, 3, -1):
        alpha = 1.0 + 2.0 ** (-exp)
        try:
            K_cand = build(alpha)
        except (np.linalg.LinAlgError, ValueError):
            continue
        dcl, margin = None, None
        try:
            dcl = _lft(dsys, K_cand, n_meas, n_ctrl, dt=dsys.dt)
            margin = 1.0 - float(np.max(np.abs(np.linalg.eigvals(dcl.A))))
        except (np.linalg.LinAlgError, ValueError):
            continue
        if margin < strict_margin:
            continue
        cost = plain_h2_cost(dcl)
        if cost < best_cost:
            best_K, best_cost = K_cand, cost

    return best_K


def _solve_q_program(build_objective, constraints_base, Q_var, q_reg0=1e-6,
                      solver=None, verbose=False, max_tries=6, q_abs_cap=1e4):
    """Solve a Q-parameterized cvxpy Problem with a self-calibrating
    Tikhonov penalty on ||Q||^2.

    Phi12/Phi21 (see _youla_maps) can be genuinely rank-deficient -- checked
    directly on a real MIMO benchmark: Phi21's smallest singular value was
    ~1e-19 (not merely poorly conditioned, actually singular to double
    precision), leaving a real null-space direction in Q with zero effect
    on the objective. Without any penalty on Q's own magnitude, cvxpy
    confirmed the raw LP is genuinely `unbounded` there (not just slow or
    inaccurate) -- the solver is free to drive Q to arbitrarily large,
    physically meaningless values along that direction (observed: Q
    coefficients up to ~6e6 from an `optimal_inaccurate` solve with no
    regularization), which then breaks the closed-loop reconstruction
    numerically.

    `build_objective(q_reg)` must return a `cp.Minimize(base_cost + q_reg *
    cp.sum_squares(Q_var))` expression. Starts at `q_reg0` and retries with
    a geometrically larger q_reg (x100 each time, up to `max_tries`) unless
    the solve both (a) reports a status other than "unbounded"/
    "unbounded_inaccurate"/"user_limit" -- the last one matters too: a
    poorly-conditioned near-null-space direction doesn't just risk an
    unbounded objective, it can also make the solver time out mid-descent
    while still exploring that direction, confirmed on a real benchmark
    (Mixed H2/L1 hit "user_limit" with Q coefficients that, while not
    literally unbounded, were still large enough to leave the closed loop
    unstable) -- and (b) returns a Q with a sane magnitude (capped at
    `q_abs_cap`; the well-regularized pendulum-benchmark solutions this was
    tuned against topped out around ~100, so 1e4 is a generous margin above
    genuine solutions but far below the ~1e6-6e6 pathological blowups seen
    with no regularization at all). This self-calibrates to whatever this
    specific Phi12/Phi21 actually needs instead of relying on a single
    hand-tuned constant that would be wrong for a different plant's scale.

    Returns
    -------
    prob : the solved cp.Problem (whichever q_reg succeeded, or the last
        attempt if none met both criteria)
    q_reg_used : float
    """
    BAD_STATUSES = ("unbounded", "unbounded_inaccurate", "user_limit")
    q_reg = q_reg0
    prob = None
    for _ in range(max_tries):
        objective = build_objective(q_reg)
        prob = cp.Problem(objective, constraints_base)
        prob.solve(solver=solver, verbose=verbose)
        ok_status = prob.status not in BAD_STATUSES and Q_var.value is not None
        ok_magnitude = ok_status and np.max(np.abs(Q_var.value)) < q_abs_cap
        if ok_magnitude:
            return prob, q_reg
        q_reg *= 100.0
    return prob, q_reg


def _verify_stabilizes(dsys, K_opt, n_meas, n_ctrl, context, marginal_tol=1e-4):
    """Raise a clear error if K_opt does not actually stabilize dsys.

    The Q-parameterization used throughout this module (_youla_maps /
    _fir_controller_from_q) is only guaranteed to produce a stabilizing
    K(Q) for every stable Q when the base controller K0 (_stabilising_h2)
    *strictly* stabilizes the plant. `_stabilising_h2` now actively
    regularizes h2reg's K0 for exactly this reason (see its own
    docstring), so that specific precondition failure is largely closed.

    A second, independent gap remains open, and is the more likely cause
    of this error firing even with a robustly-stabilizing K0: the
    finite-horizon FIR-Q linear program (`sdl1_reg`/`sd_mixed_h2_l1`/
    `sd_constrained`) has *no explicit stability constraint at all* -- it
    only minimises a truncated-horizon (`N_fir + n + nk` samples) cost, so
    it is free to select a Q whose truncated cost looks good while the
    true (untruncated) closed loop is genuinely unstable. Confirmed
    reproducible on a MIMO inverted-pendulum benchmark even after fixing
    K0's own marginal eigenvalue (K0 margin ~0.10, robustly stable):
    sdl1_reg/sd_mixed_h2_l1 still returned closed-loop pole magnitude
    ~1.2 at the default N_fir=30. Not fixed at the root -- would need an
    explicit stability-margin constraint in the LP, which isn't naturally
    LP-representable (likely requires reformulating as an SDP with a
    Lyapunov/KYP constraint) -- this only prevents silently returning a
    bad controller.
    """
    from directsd.sspace.design import _lft
    dcl = _lft(dsys, K_opt, n_meas, n_ctrl, dt=dsys.dt)
    mag = np.abs(np.linalg.eigvals(dcl.A))
    unstable = mag >= 1.0 - 1e-9
    marginal = unstable & (np.abs(mag - 1.0) < marginal_tol)
    if np.any(unstable & ~marginal):
        worst = float(mag[unstable & ~marginal].max())
        raise np.linalg.LinAlgError(
            f"{context}: the synthesized controller does not stabilize the actual "
            f"closed loop (worst pole magnitude {worst:.4f} >= 1). _stabilising_h2's "
            f"base controller K0 is regularized to be robustly stable, so this is "
            f"most likely the FIR-Q linear program itself selecting a destabilizing "
            f"Q -- the LP has no explicit stability constraint (see "
            f"_verify_stabilizes' docstring). Try a different N_fir, or check "
            f"eig(_lft(dsys, _stabilising_h2(dsys, n_meas, n_ctrl), n_meas, n_ctrl, "
            f"dt=dsys.dt).A) directly to rule out the base controller itself."
        )


def _youla_maps(dsys, K0, n_meas, n_ctrl, N_fir):
    """
    Compute the three closed-loop maps for Q-parameterisation.

    All stabilising controllers:  K = K0 + Δ(Q)

    For a discrete plant P = [[P11, P12], [P21, P22]] and base
    controller K0, the closed-loop w→z map is:

        T(Q) = T_cl + Phi12 @ Q_mat @ Phi21

    Returns
    -------
    T_cl  : (N*p, N*m)           closed-loop map with K0
    Phi12 : (N*p, N_fir*n_ctrl)  input map for Q
    Phi21 : (N_fir*n_meas, N*m)  output map for Q
    N     : int                  total impulse-response horizon used
    """
    A = dsys.A; B = dsys.B; C = dsys.C; D = dsys.D
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin  - n_ctrl
    o1 = nout - n_meas

    B1  = B[:, :i1];   B2  = B[:, i1:]
    C1  = C[:o1, :];   C2  = C[o1:, :]
    D11 = D[:o1, :i1]; D12 = D[:o1, i1:]
    D21 = D[o1:, :i1]; D22 = D[o1:, i1:]

    Ak, Bk, Ck, Dk = K0.A, K0.B, K0.C, K0.D

    n  = A.shape[0]
    nk = Ak.shape[0]

    Icl = np.eye(n_ctrl) - Dk @ D22
    Jcl = np.eye(n_meas) - D22 @ Dk

    A_cl = np.block([
        [A  + B2 @ np.linalg.solve(Icl, Dk @ C2),
         B2 @ np.linalg.solve(Icl, Ck)],
        [Bk @ np.linalg.solve(Jcl, C2),
         Ak + Bk @ np.linalg.solve(Jcl, D22 @ Ck)]
    ])
    B_cl = np.vstack([
        B1 + B2 @ np.linalg.solve(Icl, Dk @ D21),
        Bk @ np.linalg.solve(Jcl, D21)
    ])
    C_cl = np.hstack([
        C1 + D12 @ np.linalg.solve(Icl, Dk @ C2),
        D12 @ np.linalg.solve(Icl, Ck)
    ])
    D_cl = D11 + D12 @ np.linalg.solve(Icl, Dk @ D21)

    N = N_fir + n + nk
    T_cl = _ss_impulse_matrix(A_cl, B_cl, C_cl, D_cl, N)

    # P12 closed-loop: from Q-input (ctrl-dim) to z-output
    B_12 = np.vstack([
        B2 @ np.linalg.solve(Icl, np.eye(n_ctrl)),
        Bk @ np.linalg.solve(Jcl, D22 @ np.eye(n_ctrl))
    ])
    C_12 = np.hstack([
        C1 + D12 @ np.linalg.solve(Icl, Dk @ C2),
        D12 @ np.linalg.solve(Icl, Ck)
    ])
    D_12 = D12 @ np.linalg.solve(Icl, np.eye(n_ctrl))
    Phi12 = _ss_impulse_matrix(A_cl, B_12, C_12, D_12, N)[:, :N_fir * n_ctrl]

    # P21 closed-loop: from w-input to Q-output (meas-dim)
    B_21 = np.vstack([
        B1 + B2 @ np.linalg.solve(Icl, Dk @ D21),
        Bk @ np.linalg.solve(Jcl, D21)
    ])
    C_21 = np.hstack([
        np.linalg.solve(Jcl, C2),
        np.linalg.solve(Jcl, D22 @ Ck)
    ])
    D_21 = np.linalg.solve(Jcl, D21)
    Phi21 = _ss_impulse_matrix(A_cl, B_21, C_21, D_21, N)[:N_fir * n_meas, :]

    return T_cl, Phi12, Phi21, N


# ── Internal: reconstruct controller from FIR Q ──────────────────────────────

def _fir_controller_from_q(K0, Q_fir, dsys, n_meas, n_ctrl):
    """
    Build the final controller as K0 plus a parallel FIR correction Q.

    The full Youla reconstruction K = K0 + Δ(Q) is approximated here as a
    parallel connection K_total = K0 + Q (valid when ‖Q‖ is small relative
    to K0, i.e. close to the H2-optimal base).
    """
    if np.all(np.abs(Q_fir) < 1e-10):
        return K0

    N_fir, nc, nm = Q_fir.shape
    dt = K0.dt if K0.dt is not None else 1.0

    # FIR state-space: shift register of depth (N_fir-1)*nm
    n_fir_state = (N_fir - 1) * nm
    A_fir = np.zeros((n_fir_state, n_fir_state))
    for i in range(N_fir - 2):
        A_fir[nm*(i+1):nm*(i+2), nm*i:nm*(i+1)] = np.eye(nm)
    B_fir = np.zeros((n_fir_state, nm))
    B_fir[:nm, :] = np.eye(nm)
    C_fir = np.hstack([Q_fir[k] for k in range(1, N_fir)]) if N_fir > 1 \
            else np.zeros((nc, n_fir_state))
    D_fir = Q_fir[0]

    A_tot = la.block_diag(K0.A, A_fir)
    B_tot = np.vstack([K0.B, B_fir])
    C_tot = np.hstack([K0.C, C_fir])
    D_tot = K0.D + D_fir

    return sig.StateSpace(A_tot, B_tot, C_tot, D_tot, dt=dt)


# ── Public API ────────────────────────────────────────────────────────────────

def youla_basis(dsys, K0, N_fir, n_meas=1, n_ctrl=1):
    """
    Compute the Youla (Q) parameterisation basis matrices.

    Every stabilising controller for the discrete plant ``dsys`` can be
    written as K = K0 + Δ(Q) where Q is a stable FIR sequence of length
    N_fir.  The closed-loop w→z map is *affine* in the coefficients of Q:

        T(Q) = T_cl + Phi12 @ diag_block(Q) @ Phi21

    Parameters
    ----------
    dsys : scipy.signal.StateSpace (discrete)
        Lifted discrete-time generalised plant.
    K0 : scipy.signal.StateSpace (discrete)
        Base stabilising controller (typically H2-optimal).
    N_fir : int
        FIR horizon — number of Q taps.
    n_meas, n_ctrl : int

    Returns
    -------
    T_cl  : np.ndarray  (N*p, N*m)   closed-loop impulse-response matrix
    Phi12 : np.ndarray  (N*p, N_fir*n_ctrl)
    Phi21 : np.ndarray  (N_fir*n_meas, N*m)
    N     : int         total impulse-response horizon used
    """
    return _youla_maps(dsys, K0, n_meas, n_ctrl, N_fir)


def sdl1_reg(dsys, N_fir=30, n_meas=1, n_ctrl=1, solver=None, verbose=False, q_reg=1e-6):
    """
    L1-optimal (peak-to-peak) controller synthesis via Linear Programming.

    Minimises the induced ℓ∞→ℓ1 norm of the closed-loop impulse response:

        min_Q  max_j  Σ_i |T(Q)_ij|        (max column-sum of |T|)

    which equals the peak output amplitude for any unit-amplitude bounded
    disturbance signal.  For SISO systems this reduces to Σ_k |h[k]|.

    Parameters
    ----------
    dsys    : scipy.signal.StateSpace (discrete)  lifted plant from lift_h2
    N_fir   : int    FIR horizon for Q (longer = tighter approximation)
    n_meas  : int
    n_ctrl  : int
    solver  : str or None   CVXPY solver ('HIGHS', 'CLARABEL', 'SCS', …)
    verbose : bool
    q_reg   : float   starting Tikhonov weight on ||Q||^2 -- see
        _solve_q_program's docstring. Self-calibrates upward if the raw
        problem turns out to be unbounded, so this default rarely needs
        changing; lower it only if you need Q pushed harder toward the
        true (unregularized) L1 optimum on a well-conditioned plant.

    Returns
    -------
    K_opt   : scipy.signal.StateSpace (discrete)
    l1_norm : float   achieved L1-norm (peak-to-peak gain), computed from
        the actual T(Q) -- excludes the regularization term
    Q_fir   : np.ndarray, shape (N_fir, n_ctrl, n_meas)
    result  : dict   {'status', 'objective', 'solver', 'q_reg'}
    """
    _require_cvxpy()

    K0 = _stabilising_h2(dsys, n_meas, n_ctrl)
    T_cl, Phi12, Phi21, N = _youla_maps(dsys, K0, n_meas, n_ctrl, N_fir)

    p  = T_cl.shape[0] // N
    m  = T_cl.shape[1] // N
    nc, nm = n_ctrl, n_meas

    Q_var = cp.Variable(N_fir * nc * nm, name="Q_fir")
    Q_mat = _build_toeplitz_cvxpy(Q_var, N_fir, nc, nm)
    T_Q   = T_cl + Phi12 @ Q_mat @ Phi21

    # Induced ℓ∞→ℓ1 norm = max column-sum of |T|
    # LP reformulation: introduce t ≥ |T|, then min max_j Σ_i t_ij
    t = cp.Variable((N * p, N * m), name="t_abs")
    col_sums = cp.sum(t, axis=0)          # shape (N*m,)
    constraints = [t >= T_Q, t >= -T_Q]

    def build_objective(reg):
        return cp.Minimize(cp.max(col_sums) + reg * cp.sum_squares(Q_var))

    prob, q_reg_used = _solve_q_program(
        build_objective, constraints, Q_var, q_reg0=q_reg, solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(f"L1 synthesis: solver status '{prob.status}'")

    Q_opt = np.array(Q_var.value).reshape(N_fir, nc, nm) \
            if Q_var.value is not None else np.zeros((N_fir, nc, nm))

    K_opt  = _fir_controller_from_q(K0, Q_opt, dsys, n_meas, n_ctrl)
    _verify_stabilizes(dsys, K_opt, n_meas, n_ctrl, context="sdl1_reg")

    # Report the TRUE L1 cost (excludes the Tikhonov penalty added above).
    T_opt = T_cl + Phi12 @ _build_toeplitz_np(Q_opt, nc, nm) @ Phi21
    l1_val = float(np.max(np.sum(np.abs(T_opt), axis=0)))

    return K_opt, l1_val, Q_opt, {
        "status":    prob.status,
        "objective": l1_val,
        "solver":    prob.solver_stats.solver_name if prob.solver_stats else None,
        "q_reg":     q_reg_used,
    }


def sd_mixed_h2_l1(dsys, N_fir=30, n_meas=1, n_ctrl=1,
                   l1_bound=None, h2_bound=None,
                   solver=None, verbose=False, q_reg=1e-6):
    """
    Mixed H2 / L1 optimal controller synthesis.

    Solves one of three problems depending on which bound is given:

    A) ``l1_bound`` given → **minimise H2** subject to  ‖T‖_L1 ≤ l1_bound
    B) ``h2_bound`` given → **minimise L1** subject to  ‖T‖_H2 ≤ h2_bound
    C) Both or neither   → minimise H2 + L1 (scalarised, equal weight)

    Parameters
    ----------
    dsys      : scipy.signal.StateSpace (discrete)  lifted plant
    N_fir     : int    FIR horizon
    n_meas    : int
    n_ctrl    : int
    l1_bound  : float or None   upper bound on L1-norm
    h2_bound  : float or None   upper bound on H2-norm
    solver    : str or None
    verbose   : bool
    q_reg     : float   starting Tikhonov weight on ||Q||^2 -- see
        sdl1_reg's / _solve_q_program's docstring for why this exists and
        why it self-calibrates.

    Returns
    -------
    K_opt  : scipy.signal.StateSpace
    cost   : dict   {'h2': float, 'l1': float}
    Q_fir  : np.ndarray
    result : dict
    """
    _require_cvxpy()

    K0 = _stabilising_h2(dsys, n_meas, n_ctrl)
    T_cl, Phi12, Phi21, N = _youla_maps(dsys, K0, n_meas, n_ctrl, N_fir)

    p  = T_cl.shape[0] // N
    m  = T_cl.shape[1] // N
    nc, nm = n_ctrl, n_meas

    Q_var = cp.Variable(N_fir * nc * nm, name="Q")
    Q_mat = _build_toeplitz_cvxpy(Q_var, N_fir, nc, nm)
    T_Q   = T_cl + Phi12 @ Q_mat @ Phi21

    # H2 proxy: squared Frobenius norm of impulse-response matrix (= H2² for ZOH)
    h2_sq = cp.sum_squares(T_Q)

    # L1 proxy: induced ℓ∞→ℓ1 norm = max column-sum of |T|
    t_abs    = cp.Variable((N * p, N * m), name="t_abs")
    col_sums = cp.sum(t_abs, axis=0)
    l1_obj   = cp.max(col_sums)

    constraints = [t_abs >= T_Q, t_abs >= -T_Q]
    if l1_bound is not None:
        constraints.append(l1_obj <= l1_bound)
    if h2_bound is not None:
        constraints.append(h2_sq <= h2_bound ** 2)

    scale = float(np.linalg.norm(T_cl, 'fro') + 1e-8)

    def build_objective(reg):
        reg_term = reg * cp.sum_squares(Q_var)
        if l1_bound is not None and h2_bound is None:
            return cp.Minimize(h2_sq + reg_term)
        elif h2_bound is not None and l1_bound is None:
            return cp.Minimize(l1_obj + reg_term)
        else:
            return cp.Minimize(h2_sq / scale ** 2 + l1_obj / scale + reg_term)

    prob, q_reg_used = _solve_q_program(
        build_objective, constraints, Q_var, q_reg0=q_reg, solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(f"Mixed H2/L1: solver status '{prob.status}'")

    Q_opt = np.array(Q_var.value).reshape(N_fir, nc, nm) \
            if Q_var.value is not None else np.zeros((N_fir, nc, nm))

    T_opt = T_cl + Phi12 @ _build_toeplitz_np(Q_opt, nc, nm) @ Phi21
    K_opt = _fir_controller_from_q(K0, Q_opt, dsys, n_meas, n_ctrl)
    _verify_stabilizes(dsys, K_opt, n_meas, n_ctrl, context="sd_mixed_h2_l1")
    cost  = {
        "h2": float(np.sqrt(max(np.sum(T_opt ** 2), 0))),
        "l1": float(np.max(np.sum(np.abs(T_opt), axis=0))),
    }

    return K_opt, cost, Q_opt, {"status": prob.status, "q_reg": q_reg_used}


def sd_constrained(dsys, N_fir=30, n_meas=1, n_ctrl=1,
                   objective='h2',
                   envelope=None,
                   output_bound=None,
                   input_bound=None,
                   solver=None, verbose=False, q_reg=1e-6):
    """
    Synthesis with hard time-domain constraints.

    Parameters
    ----------
    dsys         : scipy.signal.StateSpace  lifted plant
    N_fir        : int   FIR horizon
    n_meas, n_ctrl : int
    objective    : str   'h2' | 'l1' | 'linf'
        'h2'  — minimise squared Frobenius norm of impulse response (= H2²)
        'l1'  — minimise induced ℓ∞→ℓ1 norm (peak-to-peak gain)
        'linf'— minimise induced ℓ2→ℓ2 norm proxy (largest singular value
                of the finite-horizon Toeplitz matrix)
    envelope     : (lo, hi) arrays of shape (N_horizon, p) or (N_horizon,)
        Hard upper and lower bounds on the closed-loop impulse response at
        each time step.  Pass ``None`` to skip.
    output_bound : float or None
        Peak output constraint: max_k ‖y[k]‖_∞ ≤ output_bound.
    input_bound  : float or None
        Peak control constraint: max_k ‖u[k]‖_∞ ≤ input_bound.
    solver       : str or None
    verbose      : bool

    Returns
    -------
    K_opt    : scipy.signal.StateSpace
    achieved : dict   {'h2', 'l1', 'linf'} — norms of the achieved closed-loop
    Q_fir    : np.ndarray
    result   : dict
    """
    _require_cvxpy()

    K0 = _stabilising_h2(dsys, n_meas, n_ctrl)
    T_cl, Phi12, Phi21, N = _youla_maps(dsys, K0, n_meas, n_ctrl, N_fir)

    p  = T_cl.shape[0] // N
    m  = T_cl.shape[1] // N
    nc, nm = n_ctrl, n_meas

    Q_var = cp.Variable(N_fir * nc * nm, name="Q")
    Q_mat = _build_toeplitz_cvxpy(Q_var, N_fir, nc, nm)
    T_Q   = T_cl + Phi12 @ Q_mat @ Phi21

    constraints = []

    # ── Envelope / template constraint ───────────────────────────────────────
    if envelope is not None:
        lo_arr, hi_arr = envelope
        lo_arr = np.atleast_2d(lo_arr)
        hi_arr = np.atleast_2d(hi_arr)
        N_env  = min(lo_arr.shape[0], N)
        for k in range(N_env):
            rs, re = k * p, (k + 1) * p
            for col in range(T_Q.shape[1]):
                constraints.append(T_Q[rs:re, col] >= lo_arr[k])
                constraints.append(T_Q[rs:re, col] <= hi_arr[k])

    # ── Peak output bound ────────────────────────────────────────────────────
    if output_bound is not None:
        constraints.append(cp.norm_inf(T_Q) <= output_bound)

    # ── Peak input (control) bound ───────────────────────────────────────────
    if input_bound is not None:
        constraints.append(cp.norm_inf(T_Q[-nc * N:, :]) <= input_bound)

    # ── Objective ────────────────────────────────────────────────────────────
    t_abs    = cp.Variable((N * p, N * m), name="t_abs")
    col_sums = cp.sum(t_abs, axis=0)
    constraints += [t_abs >= T_Q, t_abs >= -T_Q]

    def build_objective(reg):
        reg_term = reg * cp.sum_squares(Q_var)
        if objective == 'h2':
            return cp.Minimize(cp.sum_squares(T_Q) + reg_term)
        elif objective == 'l1':
            return cp.Minimize(cp.max(col_sums) + reg_term)
        elif objective == 'linf':
            # Minimise the largest singular value of the finite-horizon
            # Toeplitz matrix — a convex proxy for the induced l2->l2 norm.
            return cp.Minimize(cp.norm(T_Q, 2) + reg_term)
        else:
            raise ValueError(
                f"Unknown objective '{objective}'. Choose 'h2', 'l1', or 'linf'."
            )

    prob, q_reg_used = _solve_q_program(
        build_objective, constraints, Q_var, q_reg0=q_reg, solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(f"sd_constrained: solver status '{prob.status}'")

    Q_opt = np.array(Q_var.value).reshape(N_fir, nc, nm) \
            if Q_var.value is not None else np.zeros((N_fir, nc, nm))

    T_opt = T_cl + Phi12 @ _build_toeplitz_np(Q_opt, nc, nm) @ Phi21
    K_opt = _fir_controller_from_q(K0, Q_opt, dsys, n_meas, n_ctrl)
    _verify_stabilizes(dsys, K_opt, n_meas, n_ctrl, context="sd_constrained")

    sv = np.linalg.svd(T_opt, compute_uv=False)
    achieved = {
        "h2":   float(np.sqrt(max(np.sum(T_opt ** 2), 0))),
        "l1":   float(np.max(np.sum(np.abs(T_opt), axis=0))),
        "linf": float(sv[0]) if len(sv) > 0 else float('nan'),
    }

    return K_opt, achieved, Q_opt, {"status": prob.status, "q_reg": q_reg_used}


# ── L1-norm analysis (no synthesis) ──────────────────────────────────────────

def sdl1norm(plant, K, T=None, N=200):
    """
    Compute the L1-norm (peak-to-peak gain) of a sampled-data closed loop.

    The L1-norm equals the induced ℓ∞→ℓ1 gain: the maximum, over all
    unit-amplitude bounded disturbances, of the total output amplitude.
    For SISO systems:  ‖G‖_L1 = Σ_k |h[k]|.

    Parameters
    ----------
    plant : (num, den) tuple or scipy.signal.lti
    K     : (num, den) tuple or scipy.signal.dlti
    T     : float  sampling period
    N     : int    impulse response horizon

    Returns
    -------
    l1 : float        L1-norm (induced ℓ∞→ℓ1 gain)
    h  : np.ndarray   impulse response samples (SISO: shape (N,);
                      MIMO: shape (N, n_out, n_in))
    """
    from directsd.analysis.norms import _unpack_lti
    from directsd.polynomial.transforms import dtfm

    plant_num, plant_den, _   = _unpack_lti(plant)
    K_num,     K_den,     dt_k = _unpack_lti(K)
    if T is None:
        T = dt_k
    if T is None:
        raise ValueError("T must be provided")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D22num, D22den = dtfm((plant_num, plant_den), T)

    KD_num = np.polymul(K_num, D22num)
    KD_den = np.polymul(K_den, D22den)
    S_den  = np.polyadd(KD_den, KD_num)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss_obj = sig.dlti(KD_den, S_den, dt=T).to_ss()

    A, B, C, D = ss_obj.A, ss_obj.B, ss_obj.C, ss_obj.D
    h_mat = _impulse_response(A, B, C, D, N)   # (N, p, m)

    h2d = h_mat.reshape(N, -1)
    l1  = float(np.max(np.sum(np.abs(h2d), axis=0)))   # max column-sum

    h_out = h_mat[:, 0, 0] if h_mat.shape[1:] == (1, 1) else h_mat
    return l1, h_out
