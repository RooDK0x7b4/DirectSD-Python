"""
directsd.design.lifting
=======================
Lifting (FR-operator) for sampled-data systems.

Lifting maps the continuous-time generalized plant P(s) to an
*exact* discrete-time equivalent P_d that accounts for inter-sample
behaviour.  The H2-optimal discrete controller for P_d is also
H2-optimal for the original sampled-data loop.

References
----------
[1] T. Hagiwara & M. Araki, "FR-operator approach to H2-analysis and
    synthesis of sampled-data systems," IEEE TAC, 40(8), 1995.
[2] T. Chen & B. Francis, "Optimal Sampled-Data Control Systems,"
    Springer, 1995.
[3] C. F. Van Loan, "Computing integrals involving the matrix
    exponential," IEEE TAC, 23(3), 1978.
"""

import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import warnings


# ── helpers ─────────────────────────────────────────────────────────────────

def _rectchol(M, n_cols=None):
    """
    Rectangular Cholesky factor: find R (n_cols × size) such that R'R ≈ M.
    Uses an eigendecomposition for positive-semi-definite M.
    When n_cols is None, automatically determines rank via threshold.
    """
    M = (M + M.T) / 2          # symmetrise
    eigvals, eigvecs = la.eigh(M)
    # Clip tiny negative eigenvalues caused by floating-point rounding
    eigvals = np.maximum(eigvals, 0.0)
    if n_cols is None:
        # Keep only numerically significant eigenvalues (rank-revealing)
        thresh = eigvals.max() * 1e-10 if eigvals.max() > 0 else 0.0
        n_cols = max(1, int(np.sum(eigvals > thresh)))
    # Keep largest n_cols eigenvalues
    idx = np.argsort(eigvals)[::-1][:n_cols]
    R = np.diag(np.sqrt(eigvals[idx])) @ eigvecs[:, idx].T
    return R


def _van_loan_gram(A, B, T):
    """
    Van Loan's method (1978) to compute the *discrete input Gramian*

        G_d = ∫_0^T e^{A τ} B B' e^{A' τ} dτ

    via the matrix exponential of an augmented 2n×2n system:

        M = expm([−A   B B'; 0  A'] * T)

        G_d = M[n:, n:]' @ M[:n, n:]
    """
    n = A.shape[0]
    Znn = np.zeros((n, n))
    E = np.block([[-A,      B @ B.T],
                  [Znn,     A.T    ]]) * T
    P = la.expm(E)
    Phi_T = P[n:, n:].T          # = e^{A T}
    BB_d  = Phi_T @ P[:n, n:]   # = G_d
    return BB_d, Phi_T


def _van_loan_output(A, B2, C1, D12, T):
    """
    Van Loan-style computation for the output Gramian integral.

    Returns (C1d, D12d, Ad, B2d) where C1d is (o1×n) and D12d is (o1×i2).
    """
    n   = A.shape[0]
    i2  = B2.shape[1]
    o1  = C1.shape[0]
    nbar = n + i2

    Abar = np.block([[A,                   B2                 ],
                     [np.zeros((i2, n)),   np.zeros((i2, i2)) ]])
    C1D12 = np.hstack([C1, D12])         # (o1 × nbar)
    Q = C1D12.T @ C1D12                  # (nbar × nbar)  ≥ 0

    L = np.block([[-Abar.T,                      Q                    ],
                  [np.zeros((nbar, nbar)),        Abar                 ]]) * T
    M    = la.expm(L)
    M22  = M[nbar:, nbar:]               # e^{Abar T}

    CCDD = M22.T @ M[:nbar, nbar:]       # nbar × nbar  ≥ 0

    # Rectangular Cholesky factor F such that F'F ≈ CCDD.
    # We need F to be (o1 × nbar) so C1d = F[:, :n] and D12d = F[:, n:].
    # Use eigendecomposition keeping the top o1 modes.
    CCDD = (CCDD + CCDD.T) / 2
    eigvals, eigvecs = la.eigh(CCDD)
    eigvals = np.maximum(eigvals, 0.0)

    # Keep all numerically significant eigenvalues (rank of CCDD can exceed o1)
    n_keep = np.sum(eigvals > eigvals.max() * 1e-10)
    idx = np.argsort(eigvals)[::-1][:n_keep]
    F   = np.diag(np.sqrt(eigvals[idx])) @ eigvecs[:, idx].T  # (n_keep × nbar)

    C1d  = F[:, :n]     # n_keep × n
    D12d = F[:, n:]     # n_keep × i2

    Ad  = M22[:n, :n]
    B2d = M22[:n, n:]
    return C1d, D12d, Ad, B2d


# ── Public API ───────────────────────────────────────────────────────────────

def lift_h2(plant_ss, T, n_meas=1, n_ctrl=1):
    """
    H2-lifting: construct the exact discrete-time H2-equivalent of a
    continuous-time generalized plant (Hagiwara-Araki / Chen-Francis).

    Parameters
    ----------
    plant_ss : scipy.signal.StateSpace
        Continuous-time plant in *standard form*::

            [z]   [P11  P12] [w]
            [y] = [P21  P22] [u]

        Assumptions (same as MATLAB sdgh2mod):
        • D11 ≈ 0  (no direct disturbance-to-output)
        • D22 ≈ 0  (no direct control-to-measurement)
    T : float
        Sampling period.
    n_meas : int
        Number of measurement outputs (rows of P21/P22).
    n_ctrl : int
        Number of control inputs (cols of P12/P22).

    Returns
    -------
    dsys : scipy.signal.StateSpace
        Exact discrete-time H2-equivalent (without Δ correction term).
    gamma : float
        Additional H2-cost term due to continuous intersample signal energy
        (from B1 channel).  Full cost: J = ‖dsys‖²_H2 + gamma.
    dsys_delta : scipy.signal.StateSpace
        Variant that includes the Δ feedthrough block (FR-operator form).

    Notes
    -----
    The user workflow is::

        from directsd.design.lifting import lift_h2
        from directsd import h2reg

        plant_lifted, gamma, _ = lift_h2(plant_ss, T)
        K, h2n = h2reg(plant_lifted, n_meas=1, n_ctrl=1)
        total_cost = np.sqrt(h2n**2 + gamma)
    """
    if not isinstance(plant_ss, sig.StateSpace):
        raise TypeError("plant_ss must be a scipy.signal.StateSpace")
    if plant_ss.dt not in (None, 0):
        raise ValueError("plant_ss must be a continuous-time system (dt=None or 0)")

    A  = plant_ss.A
    B  = plant_ss.B
    C  = plant_ss.C
    D  = plant_ss.D
    n  = A.shape[0]

    nout, nin = C.shape[0], B.shape[1]
    i1 = nin  - n_ctrl
    o1 = nout - n_meas

    if i1 < 1:
        raise ValueError("No disturbance inputs (i1 < 1)")
    if o1 < 1:
        raise ValueError("No performance outputs (o1 < 1)")

    B1  = B[:, :i1];   B2  = B[:, i1:]
    C1  = C[:o1, :];   C2  = C[o1:, :]
    D11 = D[:o1, :i1]; D12 = D[:o1, i1:]
    D21 = D[o1:, :i1]; D22 = D[o1:, i1:]

    if np.linalg.norm(D11) > 1e-8 * (np.linalg.norm(D) + 1):
        warnings.warn("D11 ≠ 0 – H2-lifting assumes D11 = 0")
    if np.linalg.norm(D22) > 1e-8 * (np.linalg.norm(D) + 1):
        warnings.warn("D22 ≠ 0 – H2-lifting assumes D22 = 0")

    # ── Discretise B1 channel via Van Loan (input Gramian) ──────────────────
    BB1d, Phi_T = _van_loan_gram(A, B1, T)
    # Rectangular Cholesky: B1d B1d' = BB1d
    B1d = _rectchol(BB1d).T           # n × i1d  (i1d ≥ i1)
    i1d = B1d.shape[1]

    # ── Discretise B2/C1 channel (output Gramian) ───────────────────────────
    C1d, D12d, Ad, B2d = _van_loan_output(A, B2, C1, D12, T)

    # Scale by 1/√T (H2-norm convention)
    sqrtT = np.sqrt(T)
    C1d   = C1d  / sqrtT
    D12d  = D12d / sqrtT

    # ── Delta feedthrough (FR-operator correction) ──────────────────────────
    # Δ satisfies:  B1d @ Δ' @ [C1d D12d] = Q_correction
    # Computed via pseudo-inverse (cf. frdelta in sdgh2mod.m)
    nbar  = n + n_ctrl
    ah1 = np.block([[A,           B1 @ B1.T        ],
                    [np.zeros((n, n)), -A.T         ]])
    ah2 = np.block([[A,           B2               ],
                    [np.zeros((n_ctrl, n + n_ctrl)) ]])
    bh  = np.block([[np.zeros((n, n + n_ctrl))              ],
                    [-C1.T @ C1,  -C1.T @ D12       ]])
    H_big = np.block([[ah1,                 bh                       ],
                      [np.zeros((n + n_ctrl, 2*n)), ah2              ]])
    expH = la.expm(H_big * T)
    ind_n   = slice(n, 2*n)
    PhiInvT = expH[ind_n, ind_n]
    q_corr  = (expH[:n, 2*n:]
               + BB1d @ PhiInvT @ C1d.T @ np.hstack([C1d, D12d]))
    # Δ = pinv(B1d) @ q_corr @ pinv([C1d D12d]')
    C1dD12d = np.hstack([C1d, D12d])
    Delta = (np.linalg.pinv(B1d) @ q_corr @ np.linalg.pinv(C1dD12d.T).T).T
    Delta = Delta / sqrtT

    # ── Assemble discrete systems ────────────────────────────────────────────
    Bd = np.hstack([B1d, B2d])
    Cd = np.vstack([C1d, C2])

    # C1d is (o1_d × n), D12d is (o1_d × n_ctrl).  o1_d may equal o1.
    o1_d = C1d.shape[0]

    Dd_plain = np.block([[np.zeros((o1_d, i1d)),              D12d                       ],
                         [np.zeros((n_meas, i1d + n_ctrl))                               ]])
    Dd_delta = np.block([[Delta,                              D12d                       ],
                         [np.zeros((n_meas, Delta.shape[1] + n_ctrl))                    ]])

    # Handle non-zero D21
    if np.linalg.norm(D21) > 1e-10:
        Bd       = np.hstack([B1d, np.zeros((n, i1)), B2d])
        Dd_plain = np.block([[np.zeros((o1_d, i1d)), np.zeros((o1_d, i1)),  D12d],
                             [np.zeros((n_meas, i1d)), D21, np.zeros((n_meas, n_ctrl))]])
        Dd_delta = np.block([[Delta, np.zeros((o1_d, i1)), D12d],
                             [np.zeros((n_meas, Delta.shape[1])), D21, np.zeros((n_meas, n_ctrl))]])

    dsys       = sig.StateSpace(Ad, Bd, Cd, Dd_plain, dt=T)
    dsys_delta = sig.StateSpace(Ad, Bd, Cd, Dd_delta, dt=T)

    # ── Additional H2-cost term γ (inter-sample energy of B1 channel) ───────
    #   γ = (1/T) tr( B1' ∫ e^{A'τ} C1'C1 e^{Aτ} dτ  B1 )
    H_gamma = np.block([[-A.T,           np.eye(n),   np.zeros((n, n))],
                        [np.zeros((n,n)), -A.T,        C1.T @ C1      ],
                        [np.zeros((n,n)), np.zeros((n,n)), A           ]])
    expHg = la.expm(H_gamma * T)
    gamma = float(np.real(np.trace(
        B1.T @ expHg[2*n:, 2*n:].T @ expHg[:n, 2*n:] @ B1
    )) / T)

    return dsys, gamma, dsys_delta


def lift_l2(plant_ss, T, n_meas=1, n_ctrl=1):
    """
    L2-lifting: simplified H2-equivalent (Chen-Francis, 1995).

    Same as lift_h2 but omits the Δ correction (assumes D11 = D21 = D22 = 0
    strictly). Faster and numerically cleaner for L2-optimal design.

    Parameters
    ----------
    plant_ss, T, n_meas, n_ctrl : same as lift_h2.

    Returns
    -------
    dsys : scipy.signal.StateSpace
        Exact discrete-time L2-equivalent.
    """
    if not isinstance(plant_ss, sig.StateSpace):
        raise TypeError("plant_ss must be a scipy.signal.StateSpace")

    A = plant_ss.A; B = plant_ss.B
    C = plant_ss.C; D = plant_ss.D
    n = A.shape[0]

    nout, nin = C.shape[0], B.shape[1]
    i1 = nin - n_ctrl;  o1 = nout - n_meas
    B1  = B[:, :i1];   B2  = B[:, i1:]
    C1  = C[:o1, :];   C2  = C[o1:, :]
    D12 = D[:o1, i1:]

    # Output channel (loop + output discretisation)
    C1d, D12d, Ad, B2d = _van_loan_output(A, B2, C1, D12, T)
    sqrtT = np.sqrt(T)
    C1d  = C1d  / sqrtT
    D12d = D12d / sqrtT

    # Input channel: B1 is passed through as continuous (no lifting needed
    # for L2 – impulse disturbances)
    i1d = i1
    B1d = B1

    o1_d = C1d.shape[0]
    Bd = np.hstack([B1d, B2d])
    Cd = np.vstack([C1d, C2])
    Dd = np.block([[np.zeros((o1_d, i1d)), D12d                     ],
                   [np.zeros((n_meas, i1d + n_ctrl))                ]])

    return sig.StateSpace(Ad, Bd, Cd, Dd, dt=T)


def lft_dt(plant_ss_dt, K_ss_dt, n_meas=1, n_ctrl=1):
    """
    Lower linear fractional transformation: close the sampled-data loop.

    Returns the DT closed-loop StateSpace from exogenous input w to
    performance output z::

        Fl(P, K) = P11 + P12 * K * (I - P22*K)^{-1} * P21

    Parameters
    ----------
    plant_ss_dt : scipy.signal.StateSpace (discrete, dt=T)
        Lifted DT plant with block structure::

            [z]   [P11  P12] [w]
            [y] = [P21  P22] [u]

    K_ss_dt : scipy.signal.StateSpace (discrete, dt=T)
        Discrete-time controller K(z).
    n_meas : int
        Number of measurement outputs (rows of P21/P22).
    n_ctrl : int
        Number of control inputs (cols of P12/P22).

    Returns
    -------
    dcl : scipy.signal.StateSpace
        Closed-loop system (w → z).
    """
    Ap = plant_ss_dt.A;  Bp = plant_ss_dt.B
    Cp = plant_ss_dt.C;  Dp = plant_ss_dt.D
    T  = plant_ss_dt.dt

    nout = Cp.shape[0];  nin = Bp.shape[1]
    o1 = nout - n_meas;  i1 = nin - n_ctrl

    B1  = Bp[:, :i1];   B2  = Bp[:, i1:]
    C1  = Cp[:o1, :];   C2  = Cp[o1:, :]
    D11 = Dp[:o1, :i1]; D12 = Dp[:o1, i1:]
    D21 = Dp[o1:, :i1]; D22 = Dp[o1:, i1:]

    Ak = K_ss_dt.A;  Bk = K_ss_dt.B
    Ck = K_ss_dt.C;  Dk = K_ss_dt.D

    # M1 = (I - Dk @ D22)^{-1},  M2 = (I - D22 @ Dk)^{-1}
    I_k = np.eye(n_ctrl)
    I_m = np.eye(n_meas)
    try:
        M1 = np.linalg.solve(I_k - Dk @ D22, I_k)
    except np.linalg.LinAlgError:
        M1 = I_k                    # fallback when D22 = 0
    try:
        M2 = np.linalg.solve(I_m - D22 @ Dk, I_m)
    except np.linalg.LinAlgError:
        M2 = I_m

    Acl = np.block([
        [Ap + B2 @ Dk @ M2 @ C2,    B2 @ M1 @ Ck             ],
        [Bk @ M2 @ C2,              Ak + Bk @ M2 @ D22 @ Ck  ],
    ])
    Bcl = np.vstack([
        B1 + B2 @ Dk @ M2 @ D21,
        Bk @ M2 @ D21,
    ])
    Ccl = np.hstack([
        C1 + D12 @ Dk @ M2 @ C2,
        D12 @ M1 @ Ck,
    ])
    Dcl = D11 + D12 @ Dk @ M2 @ D21

    return sig.StateSpace(Acl, Bcl, Ccl, Dcl, dt=T)


def compute_gamma(plant_ss, T, n_meas=1, n_ctrl=1):
    """
    Compute the inter-sample correction term γ for H2-optimal design.

    γ = (1/T) tr( B1' * ∫_0^T e^{A'τ} C1'C1 e^{Aτ} dτ * B1 )

    The total H2-cost is  J = √(‖P_lifted‖²_H2 + γ).

    Parameters
    ----------
    plant_ss : scipy.signal.StateSpace
    T : float
    n_meas, n_ctrl : int

    Returns
    -------
    gamma : float
    """
    _, gamma, _ = lift_h2(plant_ss, T, n_meas, n_ctrl)
    return gamma
