"""
Native discrete-time Hinf sampled-data synthesis.

Port of MATLAB's ``dsdsspace/private/sdhimod.m`` (all 5 discretization
formulas) + ``dsdsspace/private/hinfone.m`` and ``hinfone1.m`` (both gamma=1
synthesis methods) + their private dependencies. This replaces the bilinear
DT->CT->DT approximation ``hinfreg`` previously used for discrete plants,
which is only O(T)-accurate and (before a separate fix) also carried a sign
bug in its own CT central-controller assembly.

The key idea, unlike the old approximation: ``sdhimod`` first builds an
EXACT discrete-time equivalent of the sampled-data Hinf problem (the hard,
sampled-data-specific part — correctly accounting for intersample behavior,
via the Mirkin-Palmor discretization). Only AFTER that exact reduction does
``hinfone`` use a bilinear transform internally, and there it is a
standard, EXACT algebraic conjugacy between the unit disk and the
left-half-plane for solving a gamma=1 Hinf problem on an ALREADY-discrete
system — not an approximation of the physical continuous-time plant.

All 5 ``sdhimod`` discretization formulas ('mi'/Mirkin-Palmor default,
'ch'/Chen-Francis, 'ca'/Cantoni-Glover, 'ba'/Bamieh-Pearson,
'ha'/Hayakawa-Hara-Yamamoto) and both ``hinfone`` gamma=1 synthesis methods
('sa'/Safonov-Limebeer-Chiang default, 'gl'/Glover-Doyle via ``hinfone1``)
are ported. They are all exact discretizations/solutions of the SAME
underlying continuous-time sampled-data Hinf problem, so cross-checking
that all 5 types x 2 methods agree with each other (and with the one
MATLAB-documented numeric example available) is itself a strong
correctness test -- see the test suite.

``_weierstr``/``_descr2ss_general`` (Weierstrass canonical form for a
genuinely singular descriptor pencil) and ``_schurdiag``/``_schurord``
(Cantoni-Glover's numerically-robust restructured computation) are also
ported in full -- ``descr2ss`` auto-dispatches to the fast direct-inversion
path when E is well-conditioned and only falls back to the general
weierstr-based algorithm when it isn't, and ``_cantoni_glover`` always runs
the schurdiag-based restructuring (matching MATLAB's own default branch)
with the direct computation as its fallback for the n1==0 or n2==0 edge
case.

One remaining scope reduction: ``regular.m``'s full ``syscheck``/
``separss``/``improper`` diagnostic chain (structural-validity REPORTING
only, not part of the actual numerics) is skipped; the D12/D21 rank-
regularization this codebase's plants actually need is applied directly.

References
----------
[1] L. Mirkin, G. Tadmor, "Yet another Hinf-discretization", IEEE TAC,
    vol. AC-48, no. 5, pp. 891-894, 2003.
[2] M.G. Safonov, D.J.N. Limebeer, R.Y. Chiang, "Simplifying Hinf-theory
    via loop shifting, matrix pencil and descriptor concept", Int. J.
    Control, 1990, vol. 50, pp. 2467-2488.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.signal as sig

from directsd.linalg.riccati import care2

__all__ = ['sdhimod', 'hinfone', 'sdhinfreg_native']


def _rectchol(a, m=None):
    """Rectangular Cholesky-like factor: W (n x m) such that W@W.T = a, for
    a symmetric PSD (possibly rank-deficient) n x n matrix a. Port of
    ``dsdlinalg/rectchol.m`` via SVD (a = V diag(s) V.T for symmetric a)."""
    a = np.atleast_2d(np.asarray(a, float))
    a = (a + a.T) / 2.0  # symmetrize -- matches MATLAB's tolerance to asymmetry
    n = a.shape[0]
    _, s, Vt = np.linalg.svd(a)
    tol = max(a.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
    r = int(np.sum(s > tol))
    w = Vt[:r, :].T * np.sqrt(np.maximum(s[:r], 0.0))[None, :]
    if m is None:
        m = max(1, r)
    if m > r:
        w = np.hstack([w, np.zeros((n, m - r))])
    elif m < r:
        w = w[:, :m]
    return w


def _blocks4(B, C, D, n_meas, n_ctrl):
    """Partition (B, C, D) into the standard 4-block generalized-plant
    matrices. Port of ``dsdsspace/private/blocks4.m``."""
    nin = B.shape[1]
    nout = C.shape[0]
    i1 = nin - n_ctrl
    o1 = nout - n_meas
    B1, B2 = B[:, :i1], B[:, i1:]
    C1, C2 = C[:o1, :], C[o1:, :]
    D11, D12 = D[:o1, :i1], D[:o1, i1:]
    D21, D22 = D[o1:, :i1], D[o1:, i1:]
    return B1, B2, C1, C2, D11, D12, D21, D22


def _mirkin_palmor(a, b1, b2, c1, c2, d12, T, xi2, i2, o2):
    """Port of ``sdhimod.m``'s MirkinPalmor subfunction."""
    n = a.shape[0]
    Zi2i2 = np.zeros((i2, i2))
    Hgamma = np.block([
        [Zi2i2,              xi2 * d12.T @ c1,     b2.T,          xi2 * d12.T @ d12],
        [np.zeros((n, i2)),  a,                    b1 @ b1.T,     b2],
        [np.zeros((n, i2)), -xi2 * c1.T @ c1,      -a.T,          -xi2 * c1.T @ d12],
        [np.zeros((i2, 2 * i2 + 2 * n))],
    ])
    Hgamma0 = np.block([
        [Zi2i2,              np.zeros((i2, n)),    b2.T,          np.zeros((i2, i2))],
        [np.zeros((n, i2)),  a,                    b1 @ b1.T,     b2],
        [np.zeros((n, i2)),  np.zeros((n, n)),     -a.T,          np.zeros((n, i2))],
        [np.zeros((i2, 2 * i2 + 2 * n))],
    ])
    Gamma = la.expm(Hgamma * T)
    Lam = la.expm(Hgamma0 * T)

    ind1 = slice(0, i2)
    ind2 = slice(i2, i2 + n)
    ind3 = slice(i2 + n, i2 + 2 * n)
    ind4 = slice(i2 + 2 * n, 2 * i2 + 2 * n)

    Lam13 = Lam[ind1, ind3]
    Lam22 = Lam[ind2, ind2]
    Lam23 = Lam[ind2, ind3]
    Lam24 = Lam[ind2, ind4]
    Lam33 = Lam[ind3, ind3]
    Gamma13 = Gamma[ind1, ind3]
    Gamma14 = Gamma[ind1, ind4]
    Gamma22 = Gamma[ind2, ind2]
    Gamma23 = Gamma[ind2, ind3]
    Gamma24 = Gamma[ind2, ind4]
    pinvGamma23 = np.linalg.pinv(Gamma23)
    Gamma32 = Gamma[ind3, ind2]
    Gamma33 = Gamma[ind3, ind3]
    Gamma34 = Gamma[ind3, ind4]
    In = np.eye(n)

    Ad = Lam22
    B2d = Lam24
    B1d = _rectchol(Lam23 @ Lam22.T, Lam22.shape[0])

    M12 = (In - Lam22.T @ Gamma33) @ pinvGamma23
    M11 = M12 @ (Lam22 - Gamma22) - Lam22.T @ Gamma32
    M13 = M12 @ (Lam24 - Gamma24) - Lam22.T @ Gamma34
    M22 = Lam33 @ np.linalg.pinv(Lam23) - Gamma33 @ pinvGamma23
    M32 = (Gamma13 - Lam13 @ Lam22.T @ Gamma33) @ pinvGamma23
    M33 = M32 @ (Lam24 - Gamma24) + Gamma14 - Lam13 @ Lam22.T @ Gamma34

    CCDD = np.block([
        [M11,     M12,   M13],
        [M12.T,   M22,   M32.T],
        [M13.T,   M32,   M33],
    ])
    CD = _rectchol(CCDD).T
    C1d = CD[:, :n]
    Dalpha = CD[:, n:2 * n]
    D12d = CD[:, 2 * n:]
    D11d = Dalpha @ B1d

    Bd = np.hstack([B1d, B2d])
    Cd = np.vstack([C1d, c2])
    o2_rows = c2.shape[0]
    Dd = np.block([
        [D11d, D12d],
        [np.zeros((o2_rows, n + i2))],
    ])
    return Ad, Bd, Cd, Dd


def _chen_francis(a, b1, b2, c1, c2, d12, T, xi2, i2, o2):
    """Port of ``sdhimod.m``'s ChenFrancis subfunction."""
    n = a.shape[0]
    av = np.block([[a, b2], [np.zeros((i2, n + i2))]])
    nv = av.shape[0]
    cv = np.hstack([c1, d12])
    n12 = n + i2

    a12 = np.block([
        [-av.T, -cv.T @ cv],
        [np.zeros((n12, n12)), av],
    ])
    phi12 = la.expm(a12 * T)
    ind1v, ind2v = slice(0, n12), slice(n12, 2 * n12)
    Jinf = -phi12[ind2v, ind2v].T @ phi12[ind1v, ind2v]

    E = np.block([
        [-a.T, -c1.T @ c1],
        [xi2 * b1 @ b1.T, a],
    ])
    ne = E.shape[0]
    X = np.hstack([c1, d12]).T @ np.hstack([np.zeros((c1.shape[0], ne - n)), c1])
    Y = np.hstack([c1, np.zeros((c1.shape[0], ne - n))]).T @ np.hstack([c1, d12])

    aex = np.block([
        [-av.T,               X,                    np.zeros((nv, nv))],
        [np.zeros((ne, nv)),  E,                    Y],
        [np.zeros((nv, nv)),  np.zeros((nv, ne)),   av],
    ])
    phiEx = la.expm(aex * T)
    i1_, i2_ = slice(0, nv), slice(nv, nv + ne)
    i3_ = slice(nv + ne, 2 * nv + ne)
    M = phiEx[i1_, i2_]
    L = phiEx[i1_, i3_]
    Q = phiEx[i2_, i2_]
    N = phiEx[i2_, i3_]
    R = phiEx[i3_, i3_]

    Q11 = Q[:n, :n]
    Q21 = Q[n:2 * n, :n]
    R11 = R[:n, :n]
    R12 = R[:n, n:nv]

    Q11inv = np.linalg.inv(Q11)
    n0 = M.shape[1] - n
    F = np.hstack([Q11inv.T, np.zeros((n, n0))]) @ M.T @ R
    Ad = R11 + F[:, :n]
    B2d = R12 + F[:, n:nv]

    BB1d = Q21 @ Q11inv
    B1d = _rectchol(BB1d, BB1d.shape[1])
    i1d = B1d.shape[1]

    RM = R.T @ M
    rN = N.shape[0]
    cRM = RM.shape[1]
    U = np.block([
        [Q11inv, np.zeros((n, rN - n))],
        [np.zeros((cRM - n, rN))],
    ])
    J = RM @ U @ N - R.T @ L + Jinf
    J = np.real(la.sqrtm(J @ J.T))
    o1d = np.linalg.matrix_rank(J)
    v = _rectchol(J).T
    C1d = v[:, :n]
    D12d = v[:, n:n12]

    Bd = np.hstack([B1d, B2d])
    Cd = np.vstack([C1d, c2])
    o2_rows = c2.shape[0]
    Dd = np.block([
        [np.zeros((o1d, i1d)), D12d],
        [np.zeros((o2_rows, i1d + i2))],
    ])
    return Ad, Bd, Cd, Dd


def _schurord(S, sort_type='s', alpha=0.0):
    """Ordered real Schur decomposition. Port of ``dsdlinalg/schurord.m``
    (not in this repo's source tree, but a straightforward standard
    operation): 's'/'i' sort by ascending/descending real part relative
    to alpha, 'z'/'d' by ascending/descending magnitude."""
    if sort_type == 's':
        sort_fn = lambda x: x.real < alpha
    elif sort_type == 'i':
        sort_fn = lambda x: x.real > alpha
    elif sort_type == 'z':
        sort_fn = lambda x: np.abs(x) < alpha
    elif sort_type == 'd':
        sort_fn = lambda x: np.abs(x) > alpha
    else:
        raise ValueError(f"_schurord: unknown sort type '{sort_type}'")
    T, U, sdim = la.schur(S, output='real', sort=sort_fn)
    E = np.linalg.eigvals(T)
    return U, T, E, sdim


def _schurdiag(S, sort_type='s', alpha=None):
    """Block-diagonalize an ordered real Schur form. Port of
    ``dsdlinalg/schurdiag.m`` -- used by Cantoni-Glover's numerically
    robust restructured computation (an alternative to direct matrix
    inversion when Q11 is ill-conditioned).

    Returns
    -------
    U, S, invU : ndarray -- U@S@invU = original input (up to balancing).
    n1, n2 : int -- sizes of the two diagonal blocks.
    E : ndarray -- eigenvalues.
    """
    if alpha is None:
        alpha = 0.0 if sort_type in ('s', 'i') else 1.0
    n = S.shape[0]
    Sbal, Tbal = la.matrix_balance(S)  # scipy: S = Tbal @ Sbal @ inv(Tbal)
    U, Sdiag, E, n1 = _schurord(Sbal, sort_type, alpha)
    n2 = n - n1

    if n1 == 0 or n2 == 0:
        U = Tbal @ U
        invU = U.T @ np.linalg.inv(Tbal)
        return U, Sdiag, invU, n1, n2, E

    ind1 = slice(0, n1)
    ind2 = slice(n1, n1 + n2)
    T11 = Sdiag[ind1, ind1]
    T12 = Sdiag[ind1, ind2]
    T22 = Sdiag[ind2, ind2]
    # Sylvester equation T11*X + X*(-T22) = -T12 (schurdiag.m: X=lyap(T11,-T22,T12))
    X = la.solve_sylvester(T11, -T22, -T12)
    In1, In2 = np.eye(n1), np.eye(n2)
    Y = np.block([[In1, X], [np.zeros((n2, n1)), In2]])
    invY = np.block([[In1, -X], [np.zeros((n2, n1)), In2]])
    invU = invY @ U.T @ np.linalg.inv(Tbal)
    U = Tbal @ U @ Y
    Sdiag = Sdiag.copy()
    Sdiag[ind1, ind2] = 0.0
    return U, Sdiag, invU, n1, n2, E


def _cantoni_glover(a, b1, b2, c1, c2, d12, T, xi2, i2, o2):
    """Port of ``sdhimod.m``'s CantoniGlover subfunction, including the
    numerically-robust schurdiag-based restructured computation (MATLAB's
    own default whenever the Hamiltonian-like Ehat has any eigenvalue with
    Re>1/T, which is generic) alongside the direct-inversion fallback."""
    n = a.shape[0]
    i1 = b1.shape[1]
    o1 = c1.shape[0]
    Rm = np.eye(i1)
    Sm = np.eye(o1)
    c1d12 = np.hstack([c1, d12])
    b1zero = np.vstack([b1, np.zeros((i2, i1))])
    nbar = n + i2
    Abar = np.block([[a, b2], [np.zeros((i2, nbar))]])
    Ahat = Abar
    BBhat = b1zero @ np.linalg.solve(Rm, b1zero.T) * xi2
    CChat = c1d12.T @ np.linalg.solve(Sm, c1d12)

    Ehat = np.block([
        [-Ahat.T, -CChat],
        [BBhat, Ahat],
    ])
    ind1 = slice(0, nbar)
    ind2 = slice(nbar, 2 * nbar)

    M, S, vM, n1, n2, _E = _schurdiag(Ehat, 'i', 1.0 / T)

    if n1 > 0 and n2 > 0:
        q1 = slice(0, n1)
        q2 = slice(n1, n1 + n2)
        M11, M12 = M[ind1, q1], M[ind1, q2]
        M21, M22 = M[ind2, q1], M[ind2, q2]
        vM11, vM12 = vM[q1, ind1], vM[q1, ind2]
        vM21, vM22 = vM[q2, ind1], vM[q2, ind2]

        expS11 = la.expm(-S[q1, q1] * T)
        expS22 = la.expm(S[q2, q2] * T)
        M11_null = la.null_space(M11.T).T
        ML = np.vstack([expS11 @ np.linalg.pinv(M11), M11_null])
        IZL = np.vstack([np.eye(n1), np.zeros((nbar - n1, n1))])
        vM11_null = la.null_space(vM11)
        MR = np.hstack([np.linalg.pinv(vM11.T).T @ expS11, vM11_null])
        IZR = np.hstack([np.eye(n1), np.zeros((n1, nbar - n1))])

        MeM = M12 @ expS22 @ vM21
        OmegaLinv = ML @ MeM + IZL @ vM11
        OmegaRinv = M11 @ IZR + MeM @ MR
        Q11inv = np.linalg.solve(OmegaRinv.T, MR.T).T
        Q21_Q11inv = np.linalg.solve(
            OmegaRinv.T, (M22 @ expS22 @ vM21 @ MR + M21 @ IZR).T).T
        Q11inv_Q12 = np.linalg.solve(OmegaLinv, ML @ M12 @ expS22 @ vM22 + IZL @ vM12)
    else:
        # Direct method -- Q11 inversion without restructuring.
        Qhat = la.expm(Ehat * T)
        Q11 = Qhat[:nbar, :nbar]
        Q11inv = np.linalg.inv(Q11)
        Q12 = Qhat[:nbar, nbar:]
        Q21 = Qhat[nbar:, :nbar]
        Q21_Q11inv = Q21 @ Q11inv
        Q11inv_Q12 = Q11inv @ Q12

    Q11invT = Q11inv.T
    Ad = Q11invT[:n, :n]
    B2d = Q11invT[:n, n:]
    BB1d = Q21_Q11inv[:n, :n]
    B1d = _rectchol(BB1d, BB1d.shape[1])
    CCDD = -Q11inv_Q12
    C1dD12d = _rectchol(CCDD).T
    C1d = C1dD12d[:, :n]
    D12d = C1dD12d[:, n:]

    i1d = B1d.shape[1]
    o1d = C1d.shape[0]
    Bd = np.hstack([B1d, B2d])
    Cd = np.vstack([C1d, c2])
    Dd = np.block([
        [np.zeros((o1d, n)), D12d],
        [np.zeros((o2, n + i2))],
    ])
    return Ad, Bd, Cd, Dd


def _bamieh_pearson(a, b1, b2, c1, c2, d12, T, xi2, i2, o2):
    """Port of ``sdhimod.m``'s BamiehPearson subfunction."""
    n = a.shape[0]
    H = np.block([
        [-a.T, -xi2 * c1.T @ c1],
        [b1 @ b1.T, a],
    ])
    n2 = 2 * n
    H1 = np.block([[H, np.eye(n2)], [np.zeros((n2, 2 * n2))]])
    H2 = np.block([[H1, np.eye(2 * n2)], [np.zeros((2 * n2, 4 * n2))]])
    expH2 = la.expm(H2 * T)
    Gamma = expH2[:n2, :n2]
    Phi = expH2[:n2, n2:2 * n2]
    Omega = expH2[:n2, 3 * n2:4 * n2]

    ind1, ind2 = slice(0, n), slice(n, n2)
    Gamma11, Gamma12 = Gamma[ind1, ind1], Gamma[ind1, ind2]
    Gamma21, Gamma22 = Gamma[ind2, ind1], Gamma[ind2, ind2]
    Phi11, Phi12 = Phi[ind1, ind1], Phi[ind1, ind2]
    Phi22 = Phi[ind2, ind2]
    Omega11, Omega12 = Omega[ind1, ind1], Omega[ind1, ind2]

    Ad = Gamma22 - Gamma21 @ np.linalg.solve(Gamma11, Gamma12)
    BB1d = Gamma21 @ np.linalg.inv(Gamma11)
    B1d = _rectchol(BB1d, BB1d.shape[1])
    B2d = (Phi22 - Gamma21 @ np.linalg.solve(Gamma11, Phi12)) @ b2
    CC = -np.linalg.solve(Gamma11, Gamma12)
    CD_block = -np.linalg.solve(Gamma11, Phi12) @ b2
    DD = b2.T @ (Omega12 - Phi11 @ np.linalg.solve(Gamma11, Phi12)) @ b2
    CCDD = np.block([[CC, CD_block], [CD_block.T, DD]])
    CD_result = _rectchol(CCDD).T
    C1d = CD_result[:, :n]
    D12d = CD_result[:, n:]

    i1d = B1d.shape[1]
    o1d = C1d.shape[0]
    Bd = np.hstack([B1d, B2d])
    Cd = np.vstack([C1d, c2])
    Dd = np.block([
        [np.zeros((o1d, n)), D12d],
        [np.zeros((o2, n + i2))],
    ])
    return Ad, Bd, Cd, Dd


def _hayakawa_hara_yamamoto(a, b1, b2, c1, c2, d12, T, xi, i2, o2):
    """Port of ``sdhimod.m``'s HayakawaHaraYamamoto subfunction. Note this
    branch's discrete equivalent has a genuinely nonzero D11 block (unlike
    the other 4 -- hinfone's general D11 loop-shifting logic, ported in
    full, handles this)."""
    n = a.shape[0]
    c1 = c1 * xi
    d12 = d12 * xi

    Phi = la.expm(np.block([[a, b2], [np.zeros((i2, n + i2))]]) * T)
    Ad = Phi[:n, :n]
    B2d = Phi[:n, n:]

    Psi = la.expm(np.block([[-a.T, np.zeros((n, n))], [b1 @ b1.T, a]]) * T)
    ind1, ind2 = slice(0, n), slice(n, 2 * n)
    W0 = Psi[ind2, ind1] @ Psi[ind2, ind2].T

    Q = np.block([
        [-a.T,       -c1.T @ c1,  -c1.T @ d12,          np.zeros((n, i2))],
        [b1 @ b1.T,   a,           b2,                  np.zeros((n, i2))],
        [np.zeros((i2, 2 * (n + i2)))],
        [b2.T,        d12.T @ c1,  d12.T @ d12,         np.zeros((i2, i2))],
    ])
    Gamma = la.expm(Q * T)
    jnd1, jnd2 = slice(0, n), slice(n, 2 * n)
    jnd3, jnd4 = slice(2 * n, 2 * n + i2), slice(2 * n + i2, 2 * n + 2 * i2)
    invGamma11 = np.linalg.inv(Gamma[jnd1, jnd1])

    W = Gamma[jnd2, ind1] @ invGamma11
    Vcc = -invGamma11 @ Gamma[jnd1, jnd2]
    Vcd = -invGamma11 @ Gamma[jnd1, jnd3]
    Vdd = Gamma[jnd4, jnd3] + Gamma[jnd4, jnd1] @ Vcd
    M1 = invGamma11 - Phi[ind1, ind1].T
    M2 = Gamma[jnd4, jnd1] @ invGamma11 - Phi[ind1, n:].T
    siW = np.real(la.sqrtm(np.linalg.pinv(W)))
    N = siW @ W0 @ siW
    M = siW @ W @ siW
    B1d = np.real(la.sqrtm(W)) @ np.real(la.sqrtm(N))
    Om = np.zeros_like(M)
    Omp = np.zeros((M.shape[0], Vcd.shape[1]))
    Rm = np.block([
        [Vcc, Om, Vcd],
        [Om, M, Omp],
        [Vcd.T, Omp.T, Vdd],
    ])
    sqrtN = np.real(la.sqrtm(N))
    R1 = np.vstack([M1 @ siW, -sqrtN, M2 @ siW])
    CDDd = _rectchol(Rm - R1 @ R1.T).T

    i1d = B1d.shape[1]
    o1d = CDDd.shape[0]
    C1d = CDDd[:, :n]
    D11D12d = CDDd[:, n:]
    Bd = np.hstack([B1d, B2d])
    Cd = np.vstack([C1d, c2])
    Dd = np.vstack([D11D12d, np.zeros((o2, n + i2))])
    return Ad, Bd, Cd, Dd


_SDHIMOD_TYPES = {
    'mi': _mirkin_palmor,
    'ch': _chen_francis,
    'ca': _cantoni_glover,
    'ba': _bamieh_pearson,
    'ha': _hayakawa_hara_yamamoto,
}


def sdhimod(csys, T, n_meas=1, n_ctrl=1, gamma=1.0, type='mi'):
    """
    Hinf-discrete equivalent of a sampled-data system.

    Port of ``sdhimod.m``. ``type`` selects the discretization formula:
    'mi' (Mirkin-Palmor, default), 'ch' (Chen-Francis), 'ca'
    (Cantoni-Glover, direct path only -- see module docstring), 'ba'
    (Bamieh-Pearson), 'ha' (Hayakawa-Hara-Yamamoto). All 5 are exact
    discretizations of the same underlying problem.

    Parameters
    ----------
    csys : scipy.signal.StateSpace (continuous)
        Standard 4-block generalized plant [z;y] = [[P11,P12],[P21,P22]][w;u].
        D21 and D22 are assumed zero (matching MATLAB's own precondition).
    T : float
        Sampling period.
    n_meas, n_ctrl : int
    gamma : float
        Bound on the induced Hinf norm being tested.

    Returns
    -------
    dsys : scipy.signal.StateSpace (discrete, dt=T)
        Discrete-time Hinf-equivalent model.
    """
    if type not in _SDHIMOD_TYPES:
        raise ValueError(f"sdhimod: unknown type '{type}', expected one of "
                          f"{sorted(_SDHIMOD_TYPES)}")

    A, B, C, D = csys.A, csys.B, csys.C, csys.D
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin - n_ctrl
    o1 = nout - n_meas
    if i1 < 1 or o1 < 1:
        raise ValueError("sdhimod: invalid n_meas/n_ctrl for this plant")

    gamma2 = gamma ** 2
    xi = 1.0 / gamma
    xi2 = 1.0 / gamma2

    b1, b2, c1, c2, d11, d12, d21, d22 = _blocks4(B, C, D, n_meas, n_ctrl)

    # D11 loop-shifting (only when D11 != 0 -- most generalized plants here
    # have D11=0, so this is usually a no-op).
    if np.linalg.norm(d11) > np.finfo(float).eps:
        Rxi = np.eye(i1) - xi2 * d11.T @ d11
        sqrtRxi = la.sqrtm(Rxi)
        Sxi = np.eye(o1) - xi2 * d11 @ d11.T
        sqrtSxi = la.sqrtm(Sxi)
        A = A + xi2 * b1 @ d11.T @ np.linalg.solve(Sxi, c1)
        b1 = b1 @ np.linalg.inv(sqrtRxi)
        b2 = b2 + xi2 * b1 @ d11.T @ np.linalg.solve(Sxi, d12)
        c1 = np.linalg.solve(sqrtSxi, c1)
        d12 = np.linalg.solve(sqrtSxi, d12)

    formula = _SDHIMOD_TYPES[type]
    # HayakawaHaraYamamoto uses xi (1/gamma), not xi2 (1/gamma^2) -- matches
    # sdhimod.m's own dispatch comment "% xi instead of xi2".
    xi_arg = xi if type == 'ha' else xi2
    Ad, Bd, Cd, Dd = formula(A, b1, b2, c1, c2, d12, T, xi_arg, n_ctrl, n_meas)
    return sig.StateSpace(Ad, Bd, Cd, Dd, dt=T)


# ─────────────────────────────────────────────────────────────────────────
# hinfone (Safonov-Limebeer-Chiang gamma=1 synthesis) and its dependencies
# ─────────────────────────────────────────────────────────────────────────

def _bilinss(A, B, C, D, is_discrete):
    """Unit (T-independent) bilinear transform, K=[1,-1,1,1] default.
    Port of ``bilinss.m``. Used by ``hinfone`` purely as an algebraic tool
    to reuse continuous-domain Riccati formulas on an already-discrete,
    already gamma-scaled plant -- an exact conjugacy, not an approximation
    of any physical continuous-time system."""
    n = A.shape[0]
    if not is_discrete:
        a, b, c, d = 1.0, -1.0, 1.0, 1.0
    else:
        a, b, c, d = -1.0, -1.0, 1.0, -1.0
    temp = np.linalg.inv(a * np.eye(n) - c * A)
    AB = (d * A - b * np.eye(n)) @ temp
    BB = (a * d - c * b) * (temp @ B)
    CB = C @ temp
    DB = D + c * (C @ temp @ B)
    return AB, BB, CB, DB


def _scaless(b1, b2, c1, c2, d11, d12, d21):
    """D12/D21 SVD-based scaling. Port of ``scaless.m`` (d22 is unused/
    unmodified in the original and is not touched here either)."""
    i1 = b1.shape[1]
    o1 = c1.shape[0]
    i2 = b2.shape[1]

    u12, s12, v12t = np.linalg.svd(d12)
    v12 = v12t.T
    u1 = u12[:, :i2]
    u2 = u12[:, i2:o1]
    su2 = v12 @ np.linalg.inv(np.diag(s12[:i2]))
    sy1 = np.vstack([u2.T, u1.T])

    u21, s21, v21t = np.linalg.svd(d21)
    v21 = v21t.T
    o2 = d21.shape[0]
    v1 = v21[:, :o2]
    v2 = v21[:, o2:i1]
    sy2 = np.linalg.inv(np.diag(s21[:o2])) @ u21.T
    su1 = np.hstack([v2, v1])

    b1n = b1 @ su1
    b2n = b2 @ su2
    c1n = sy1 @ c1
    c2n = sy2 @ c2
    d11n = sy1 @ d11 @ su1
    d12n = sy1 @ d12 @ su2
    d21n = sy2 @ d21 @ su1
    return b1n, b2n, c1n, c2n, d11n, d12n, d21n, su1, su2, sy1, sy2


def _regular(A, B, C, D, n_meas, n_ctrl, tol=1e-6):
    """Regularize D12 (full column rank) / D21 (full row rank) so the
    Riccati-based Hinf synthesis is well-posed. Simplified port of
    ``regular.m`` -- skips its ``syscheck``/``separss``/``improper``
    structural-diagnostic chain (report-message generation only) and
    applies the actual rank-fixing perturbation directly; mathematically
    identical for any plant that reaches this function needing fixing."""
    n = A.shape[0]
    b1, b2, c1, c2, d11, d12, d21, d22 = _blocks4(B, C, D, n_meas, n_ctrl)
    i1 = b1.shape[1]

    if np.linalg.det(d12.T @ d12) == 0:
        r12, c12 = d12.shape
        if r12 < c12:
            c1 = np.vstack([c1, np.zeros((c12, n))])
            d11 = np.vstack([d11, np.zeros((c12, i1))])
            d12 = np.vstack([d12, np.ones((c12, c12)) * tol])
        else:
            d12 = d12 + np.eye(r12, c12) * tol

    if np.linalg.det(d21 @ d21.T) == 0:
        r21, c21 = d21.shape
        if r21 > c21:
            b1 = np.hstack([b1, np.zeros((n, r21))])
            d11 = np.hstack([d11, np.zeros((d11.shape[0], r21))])
            d21 = np.hstack([d21, np.eye(r21) * tol])
        else:
            d21 = d21 + np.eye(r21, c21) * tol

    B_new = np.hstack([b1, b2])
    C_new = np.vstack([c1, c2])
    D_new = np.block([[d11, d12], [d21, d22]])
    return A, B_new, C_new, D_new


def _weierstr(A, E):
    """Canonical Weierstrass form for a matrix pencil sE-A.

    Port of ``dsdlinalg/weierstr.m``. Separates the pencil's finite
    eigenvalues (A11 block) from its eigenvalues at infinity (E22 block,
    nilpotent), via transforms Q, Z such that Q@(sE-A)@Z =
    blkdiag(s*I-A1, s*E1-I).

    MATLAB's own ``qzpencil``+``csf2rsf`` (complex QZ then convert to real
    Schur form) collapses into a single ``scipy.linalg.ordqz(..., output=
    'real')`` call here -- scipy supports real-arithmetic ordered QZ
    natively, so there's no separate complex-to-real conversion step to
    port.

    Returns
    -------
    A11 : (n1, n1) ndarray -- the finite-eigenvalue block.
    E22 : (n2, n2) ndarray -- the nilpotent (infinite-eigenvalue) block.
    Q, Z : (n, n) ndarray -- the full transformation matrices, so callers
        (``_descr2ss_general``) can project B/C without re-deriving them.
    """
    n = A.shape[0]
    tol = 1e-10

    aa, ee, alpha, beta, Q_z, Z_z = la.ordqz(
        A, E, sort=lambda al, be: np.abs(be) > tol, output='real')
    # scipy: A = Q_z @ aa @ Z_z.T:  weierstr's own Q,Z (applied as
    # Q@(sE-A)@Z) are Q_z.T and Z_z respectively.
    Q = Q_z.T
    Z = Z_z

    n1 = int(np.sum(np.abs(beta) > tol))
    n2 = n - n1

    A11 = aa[:n1, :n1]
    A12 = aa[:n1, n1:n]
    A22 = aa[n1:, n1:]
    E11 = ee[:n1, :n1]
    E12 = ee[:n1, n1:n]
    E22 = ee[n1:, n1:]

    In1 = np.eye(n1)
    In2 = np.eye(n2)
    Z21 = np.zeros((n2, n1))

    # Zero the (1,2) blocks via the coupled Sylvester pair
    #   A11*R - L*A22 = -A12,  E11*R - L*E22 = -E12
    # solved as one large linear system, exactly matching weierstr.m's own
    # kron-based reduction (no off-the-shelf coupled-Sylvester solver in
    # scipy for this two-equation/two-unknown form). Trivially a no-op
    # (R, L both empty) when n1==0 or n2==0, but MATLAB's own code doesn't
    # special-case that -- the block system just has an empty side -- so
    # this always runs; only the E11->I / A22->I normalization below is
    # where the real content is when one block is empty.
    if n1 > 0 and n2 > 0:
        M = np.block([
            [np.kron(In2, A11), np.kron(-A22.T, In1)],
            [np.kron(In2, E11), np.kron(-E22.T, In1)],
        ])
        n12 = A12.size
        b = -np.concatenate([A12.reshape(n12, order='F'), E12.reshape(n12, order='F')])
        x = np.linalg.solve(M, b)
        R = x[:n12].reshape(n1, n2, order='F')
        L = x[n12:].reshape(n1, n2, order='F')
        Q = np.block([[In1, -L], [Z21, In2]]) @ Q
        Z = Z @ np.block([[In1, R], [Z21, In2]])

    # Transform A22 and E11 to identity via direct inversion (both are
    # invertible for a regular pencil: E11 spans the finite-eigenvalue
    # subspace, A22 the infinite-eigenvalue one -- a regular pencil can't
    # have both A and E singular in the same direction). This ALWAYS runs,
    # even when n1==0 or n2==0 -- it's what actually converts the raw QZ
    # blocks into the true finite-eigenvalue matrix (E11^-1 @ A11) and the
    # true nilpotent infinite-eigenvalue matrix (A22^-1 @ E22).
    if n1 > 0:
        Q1 = np.block([[np.linalg.inv(E11), Z21.T], [Z21, In2]])
        A11 = np.linalg.solve(E11, A11)
        Q = Q1 @ Q
    if n2 > 0:
        Q2 = np.block([[In1, Z21.T], [Z21, np.linalg.inv(A22)]])
        E22 = np.linalg.solve(A22, E22)
        Q = Q2 @ Q
        E22 = E22.copy()
        np.fill_diagonal(E22, 0.0)  # E22 is nilpotent by construction; snap

    return A11, E22, Q, Z


def _descr2ss_general(AD, BD, CD, DD, ED):
    """Full descriptor-to-state-space conversion via ``_weierstr`` -- port
    of ``descr2ss.m``. Handles a genuinely singular E (unlike
    ``_descr2ss_regular``'s fast path), at the cost of the extra QZ-based
    machinery; raises if the resulting transfer function is improper
    (matching MATLAB's own "erroneous results possible" warning path,
    upgraded to a clear error rather than silently returning garbage)."""
    nin = BD.shape[1]
    nout = CD.shape[0]

    A11, E22, Q, Z = _weierstr(AD, ED)
    n1 = A11.shape[0]
    n2 = E22.shape[0]

    CZ = CD @ Z
    QB = Q @ BD

    if n1 == 0:
        Bm = np.zeros((0, nin))
        Cm = np.zeros((nout, 0))
    else:
        Bm = QB[:n1, :]
        Cm = CZ[:, :n1]

    if n2 == 0:
        D = DD
    else:
        temp = E22.copy()
        k = 0
        while np.linalg.norm(temp) > np.finfo(float).eps:
            temp = temp @ E22
            k += 1
            if k > n2 + 1:
                break  # E22 is nilpotent; this always terminates for a
                       # genuinely proper system -- bail defensively.
        if k > 0:
            raise np.linalg.LinAlgError(
                "descr2ss: transfer matrix is not proper (descriptor pencil "
                "has a nontrivial improper part) -- no state-space realization "
                "exists for this controller")
        B2 = QB[n1:, :]
        C2 = CZ[:, n1:]
        D = C2 @ B2 + DD

    # A11 (from _weierstr) already IS the reduced state matrix: in the
    # transformed basis Q@(sE-A)@Z = blkdiag(sI-A11, sE22-I), so the
    # regular part's dynamics are literally x' = A11 @ x + ... directly.
    return A11, Bm, Cm, D


def _descr2ss_regular(AD, BD, CD, DD, ED, force_general=False):
    """Descriptor-to-state-space conversion.

    Fast path (default): specialized to the E-invertible (regular pencil)
    case -- any valid similarity transform gives the same transfer
    function, so direct inversion is an exact (not approximate) substitute
    for the full ``weierstr``-based algorithm whenever E is genuinely
    invertible, which is the expected case for a well-posed Hinf problem.

    Falls back to ``_descr2ss_general`` (or set ``force_general=True``)
    when E is singular/near-singular -- e.g. the "singular descriptor
    solution" case ``hinfone``/``hinfone1`` already flag via their own
    ``cond(E)`` check before calling this.
    """
    if not force_general and np.linalg.cond(ED) < 1.0 / np.finfo(float).eps:
        Einv = np.linalg.inv(ED)
        return Einv @ AD, Einv @ BD, CD, DD
    return _descr2ss_general(AD, BD, CD, DD, ED)


def hinfone(sys, n_meas=1, n_ctrl=1, method='sa'):
    """
    Sub-Hinf controller with gamma=1 for a discrete-time LTI system.

    Port of ``hinfone.m`` (``method='sa'``, Safonov, Limebeer & Chiang,
    Int. J. Control, 1990 -- loop-shifting + two Riccati equations (P/S) +
    descriptor-form controller assembly) and ``hinfone1.m``
    (``method='gl'``, Glover & Doyle, Systems and Control Letters, 1988 --
    direct D11-aware Riccati formulas, no loop-shifting preamble). Both
    solve the SAME gamma=1 sub-Hinf problem; they should agree on the
    achieved closed loop for any well-posed plant (see the test suite's
    cross-validation).

    Parameters
    ----------
    sys : scipy.signal.StateSpace (discrete)
        Standard 4-block plant, already gamma-normalized (as produced by
        ``sdhimod``) so that a gamma=1 controller is being sought.
    n_meas, n_ctrl : int
    method : str
        'sa' (default, Safonov-Limebeer-Chiang) or 'gl' (Glover-Doyle).

    Returns
    -------
    K : scipy.signal.StateSpace (discrete) or None
        The gamma=1 controller, or None if no solution was found.
    gamma_flag : float
        0 (or the achieved-gamma value) on success; -1 no solution found;
        1 singular descriptor solution; 1e10 P or S not positive definite.
    """
    if method not in ('sa', 'gl'):
        raise ValueError(f"hinfone: unknown method '{method}', expected 'sa' or 'gl'")
    core = _hinfone_ct if method == 'sa' else _hinfone1_ct

    o2, i2 = n_meas, n_ctrl
    A, B, C, D = _regular(sys.A, sys.B, sys.C, sys.D, n_meas, n_ctrl)
    nout, nin = C.shape[0], B.shape[1]
    i1, o1 = nin - i2, nout - o2
    b1, b2, c1, c2, d11, d12, d21, d22 = _blocks4(B, C, D, n_meas, n_ctrl)

    # ── Discrete system: solve via bilinear-transformed CT problem ───────
    Ab, Bb, Cb, Db = _bilinss(A, B, C, D, is_discrete=True)
    K, gamma = core(Ab, Bb, Cb, Db, o2, i2)
    if K is not None:
        Kb = sig.StateSpace(*_bilinss(K.A, K.B, K.C, K.D, is_discrete=False), dt=sys.dt)
        return Kb, gamma
    return None, gamma


def _hinfone_ct(A, B, C, D, o2, i2):
    """Continuous-time core of hinfone (called on the bilinear-transformed
    plant). Mirrors hinfone.m's body after its ``isdt`` branch."""
    nout, nin = C.shape[0], B.shape[1]
    i1, o1 = nin - i2, nout - o2
    b1, b2, c1, c2, d11, d12, d21, d22 = _blocks4(B, C, D, o2, i2)

    d220 = d22
    d22 = np.zeros_like(d22)
    b1, b2, c1, c2, d11, d12, d21, su1, su2, sy1, sy2 = _scaless(b1, b2, c1, c2, d11, d12, d21)

    # ── Lower bound for the Hinf norm from D11 ────────────────────────────
    q11 = d11[:o1 - i2, :i1 - o2]
    q21 = d11[o1 - i2:o1, :i1 - o2]
    q12 = d11[:o1 - i2, i1 - o2:i1]
    q22 = d11[o1 - i2:o1, i1 - o2:i1]

    gammar = max(0.0, np.linalg.svd(np.hstack([q11, q12]), compute_uv=False).max()
                 if min(np.hstack([q11, q12]).shape) > 0 else 0.0)
    gammac = max(0.0, np.linalg.svd(np.vstack([q11, q21]), compute_uv=False).max()
                 if min(np.vstack([q11, q21]).shape) > 0 else 0.0)
    gamma = max(gammar, gammac)
    if gamma > 1:
        return None, gamma

    # ── fopt such that det(I + fopt*sy2*d22_0*su2) != 0 ──────────────────
    if i1 == o2 or o1 == i2:
        fopt = -q22
    else:
        fopt = -(q22 + q21 @ np.linalg.solve(
            np.eye(i1 - o2) - q11.T @ q11, q11.T) @ q12)

    imax = 23
    foptscale = 2.0 ** (-imax)
    fopt1 = fopt
    for _ in range(imax + 1):
        f = np.eye(fopt.shape[0]) + fopt1 @ sy2 @ d220 @ su2
        sigma = np.linalg.svd(f, compute_uv=False)
        if not np.any(sigma < 1e-8):
            break
        sfopt = np.linalg.svd(fopt, compute_uv=False)
        fopt1 = (1 + foptscale * (1 - gamma) / sfopt[0]) * fopt
        foptscale *= 2.0
    fopt = fopt1
    if not (gamma < 1 and foptscale < 1.99):
        return None, -1.0

    # ── Loop-shift out D11 (Safonov-Limebeer-Chiang) ──────────────────────
    A = A + b2 @ fopt @ c2
    b1 = b1 + b2 @ fopt @ d21
    c1 = c1 + d12 @ fopt @ c2
    d11 = d11 + d12 @ fopt @ d21

    X = d11
    IXX1 = np.eye(o1) - X @ X.T
    IXX2 = np.eye(i1) - X.T @ X
    IIXX1 = np.linalg.inv(IXX1)

    A = A + b1 @ X.T @ IIXX1 @ c1
    b2 = b2 + b1 @ X.T @ IIXX1 @ d12
    b1 = b1 @ np.real(la.fractional_matrix_power(IXX2, -0.5))
    c2 = c2 + d21 @ X.T @ IIXX1 @ c1
    c1 = np.real(la.fractional_matrix_power(IXX1, -0.5)) @ c1
    d22 = d21 @ X.T @ IIXX1 @ d12

    d11 = d11 * 0
    d12 = np.real(la.fractional_matrix_power(IXX1, -0.5)) @ d12
    d21 = d21 @ np.real(la.fractional_matrix_power(IXX2, -0.5))

    d22a = d22
    d22 = d22 * 0

    b1, b2, c1, c2, d11, d12, d21, tu1, tu2, ty1, ty2 = _scaless(b1, b2, c1, c2, d11, d12, d21)

    # ── State-feedback (P) Riccati ────────────────────────────────────────
    n = A.shape[0]
    c1til = (np.eye(o1) - d12 @ d12.T) @ c1
    b1til = b1 @ (np.eye(i1) - d21.T @ d21)

    a1 = A - b2 @ d12.T @ c1
    eigA1 = np.linalg.eigvals(a1)

    if o1 == i2 and np.all(eigA1.real < 0):
        P2 = np.zeros((n, n))
        P1 = np.eye(n)
        P = np.zeros((n, n))
    else:
        q1 = c1til.T @ c1til
        r1 = -(b1 @ b1.T - b2 @ b2.T)
        P, poles_p, err_p, P1, P2 = care2(a1, q1, r1)
        # eig(a1*P1 - b2*b2'*P2, P1) -- generalized eigenvalue problem
        eigP_inf = la.eig(a1 @ P1 - b2 @ b2.T @ P2, P1, right=False)
        eigP_inf = np.where(~np.isfinite(eigP_inf), 1.0 / np.finfo(float).eps, eigP_inf)
        if np.max(eigP_inf.real) > 0:
            return None, 1e10

    # ── Output-injection (S) Riccati ──────────────────────────────────────
    a2 = A - b1 @ d21.T @ c2
    eigA2 = np.linalg.eigvals(a2)

    if o2 == i1 and np.all(eigA2.real < 0):
        S2 = np.zeros((n, n))
        S1 = np.eye(n)
        S = np.zeros((n, n))
    else:
        q2 = b1til @ b1til.T
        r2 = -(c1.T @ c1 - c2.T @ c2)
        S, poles_s, err_s, S1t, S2t = care2(a2.T, q2, r2)
        S1 = S1t.T
        S2 = S2t.T
        eigS_inf = la.eig(S1 @ a2 - S2 @ c2.T @ c2, S1, right=False)
        eigS_inf = np.where(~np.isfinite(eigS_inf), 1.0 / np.finfo(float).eps, eigS_inf)
        if np.max(eigS_inf.real) > 0:
            return None, 1e10

    # ── Compatibility check ────────────────────────────────────────────────
    eigPS = la.eig(P2.T @ S2.T, P1.T @ S1.T, right=False)
    eigPS = np.where(~np.isfinite(eigPS), 1.0 / np.finfo(float).eps, eigPS)
    gamma_out = np.max(eigPS.real)
    if gamma_out > 1:
        return None, gamma_out

    # ── Controller parameterization in descriptor form ────────────────────
    E = S1 @ P1 - S2 @ P2

    ak = A - b2 @ d12.T @ c1 - b1 @ d21.T @ c2
    ak = (S1 @ ak @ P1 + S2 @ ak.T @ P2
          + S1 @ (b1til @ b1til.T - b2 @ b2.T) @ P2
          + S2 @ (c1til.T @ c1til - c2.T @ c2) @ P1)

    bk1 = S2 @ c2.T + S1 @ b1 @ d21.T
    bk2 = S1 @ b2 + S2 @ c1.T @ d12
    ck1 = -(b2.T @ P2 + d12.T @ c1 @ P1)
    ck2 = -(c2 @ P1 + d21 @ b1.T @ P2)

    ki1, ki2 = bk1.shape[1], bk2.shape[1]
    ko1, ko2 = ck1.shape[0], ck2.shape[0]

    dk11 = np.zeros((ko1, ki1))
    dk12 = np.eye(ko1)
    dk21 = np.eye(ko2)
    dk22 = np.zeros((ko2, ki2))

    # Reverse controller scaling (stage II)
    bk1 = bk1 @ ty2
    ck1 = tu2 @ ck1
    dk11 = tu2 @ dk11 @ ty2
    dk12 = tu2 @ dk12
    dk21 = dk21 @ ty2

    # Shift D22_A
    temp = np.linalg.inv(np.eye(dk11.shape[0]) + dk11 @ d22a)
    ak = ak - bk1 @ d22a @ temp @ ck1
    bk2 = bk2 - bk1 @ d22a @ temp @ dk12
    ck2 = ck2 - dk21 @ d22a @ temp @ ck1
    ck1 = temp @ ck1
    dk22 = dk22 - dk21 @ d22a @ temp @ dk12
    dk12 = temp @ dk12
    dk11 = temp @ dk11

    temp = np.eye(d22a.shape[0]) - d22a @ dk11
    bk1 = bk1 @ temp
    dk21 = dk21 @ temp

    # Reverse the "fopt" term
    dk11 = dk11 + fopt

    # Reverse controller scaling (stage I)
    bk1 = bk1 @ sy2
    ck1 = su2 @ ck1
    dk11 = su2 @ dk11 @ sy2
    dk12 = su2 @ dk12
    dk21 = dk21 @ sy2

    # Shift the initial D22 term
    temp = np.linalg.inv(np.eye(dk11.shape[0]) + dk11 @ d220)
    ak = ak - bk1 @ d220 @ temp @ ck1
    bk2 = bk2 - bk1 @ d220 @ temp @ dk12
    ck2 = ck2 - dk21 @ d220 @ temp @ ck1
    ck1 = temp @ ck1
    dk22 = dk22 - dk21 @ d220 @ temp @ dk12
    dk12 = temp @ dk12
    dk11 = temp @ dk11

    temp = np.eye(d220.shape[0]) - d220 @ dk11
    bk1 = bk1 @ temp
    dk21 = dk21 @ temp

    # ── Descriptor -> state space ──────────────────────────────────────────
    singular_flag = 0.0
    if np.linalg.cond(E) > 1.0 / np.finfo(float).eps:
        singular_flag = 1.0  # matches MATLAB's "singular solution" gamma=1 flag

    bk_full = np.hstack([bk1, bk2])
    ck_full = np.vstack([ck1, ck2])
    dk_full = np.block([[dk11, dk12], [dk21, dk22]])
    ak_ss, bk_ss, ck_ss, dk_ss = _descr2ss_regular(ak, bk_full, ck_full, dk_full, E)

    bk1_f = bk_ss[:, :ki1]
    ck1_f = ck_ss[:ko1, :]
    dk11_f = dk_ss[:ko1, :ki1]

    K = sig.StateSpace(ak_ss, bk1_f, ck1_f, dk11_f)
    return K, (singular_flag if singular_flag else gamma_out)


def _hinfone1_ct(A, B, C, D, o2, i2):
    """Continuous-time core of hinfone1 (Glover-Doyle, 1988). Mirrors
    hinfone1.m's body after its ``isdt`` branch -- direct D11-aware
    Riccati formulas (no Safonov-Limebeer-Chiang loop-shifting preamble)."""
    nout, nin = C.shape[0], B.shape[1]
    i1, o1 = nin - i2, nout - o2
    n = A.shape[0]
    b1, b2, c1, c2, d11, d12, d21, d22 = _blocks4(B, C, D, o2, i2)

    d220 = d22
    d22 = np.zeros_like(d22)
    b1, b2, c1, c2, d11, d12, d21, su1, su2, sy1, sy2 = _scaless(b1, b2, c1, c2, d11, d12, d21)

    # ── Lower bound for the Hinf norm from D11 ────────────────────────────
    q11 = d11[:o1 - i2, :i1 - o2]
    q21 = d11[o1 - i2:o1, :i1 - o2]
    q12 = d11[:o1 - i2, i1 - o2:i1]
    q22 = d11[o1 - i2:o1, i1 - o2:i1]
    gammar = max(0.0, np.linalg.svd(np.hstack([q11, q12]), compute_uv=False).max()
                 if min(np.hstack([q11, q12]).shape) > 0 else 0.0)
    gammac = max(0.0, np.linalg.svd(np.vstack([q11, q21]), compute_uv=False).max()
                 if min(np.vstack([q11, q21]).shape) > 0 else 0.0)
    gamma = max(gammar, gammac)
    if gamma > 1:
        return None, gamma

    gam = 1.0
    gam2 = gam * gam

    # ── Fix D11 ────────────────────────────────────────────────────────────
    if i1 == o2 or o1 == i2:
        dk11 = -q22
        dk12 = np.eye(i2)
        dk21 = np.eye(o2)
    else:
        gamd1 = gam2 * np.eye(o1 - i2) - q11 @ q11.T
        gamd2 = gam2 * np.eye(i1 - o2) - q11.T @ q11
        dk11 = -q21 @ q11.T @ np.linalg.solve(gamd1, q12) - q22
        dk1212 = np.eye(i2) - q21 @ np.linalg.solve(gamd2, q21.T)
        dk12 = np.linalg.cholesky(dk1212)
        dk2121 = np.eye(o2) - q12.T @ np.linalg.solve(gamd1, q12)
        dk21 = np.linalg.cholesky(dk2121).T

    # ── X Riccati equation ─────────────────────────────────────────────────
    Zr = np.zeros((i1 + i2, i1 + i2))
    Zr[:i1, :i1] = gam2 * np.eye(i1)
    d1d = np.hstack([d11, d12])
    R = d1d.T @ d1d - Zr
    Ri = np.linalg.inv(R)
    Bm = np.hstack([b1, b2])
    ax = A - Bm @ Ri @ d1d.T @ c1
    rx = Bm @ Ri @ Bm.T
    qx = c1.T @ c1 - c1.T @ d1d @ Ri @ d1d.T @ c1
    X, poles_x, err_x, X1, X2 = care2(ax, qx, rx)

    # ── Y Riccati equation ─────────────────────────────────────────────────
    Zrw = np.zeros((o1 + o2, o1 + o2))
    Zrw[:o1, :o1] = gam2 * np.eye(o1)
    dd1 = np.vstack([d11, d21])
    Rw = dd1 @ dd1.T - Zrw
    Rwi = np.linalg.inv(Rw)
    Cm = np.vstack([c1, c2])
    ay = A.T - Cm.T @ Rwi @ dd1 @ b1.T
    ry = Cm.T @ Rwi @ Cm
    qy = b1 @ b1.T - b1 @ dd1.T @ Rwi @ dd1 @ b1.T
    Y, poles_y, err_y, Y1, Y2 = care2(ay, qy, ry)

    # ── Solvability check ──────────────────────────────────────────────────
    gamma_out = np.max(np.linalg.eigvals(X @ Y).real)
    if gamma_out > 1:
        return None, gamma_out

    # ── Auxiliary matrices ──────────────────────────────────────────────────
    F = -Ri @ (d1d.T @ c1 + Bm.T @ X)
    F11 = F[:i1 - o2, :]
    F12 = F[i1 - o2:i1, :]
    F2 = F[i1:i1 + i2, :]

    H = -(b1 @ dd1.T + Y @ Cm.T) @ Rwi
    H11 = H[:, :o1 - i2]
    H12 = H[:, o1 - i2:o1]
    H2 = H[:, o1:o1 + o2]

    # ── Hinf-optimal controller in descriptor form ─────────────────────────
    Z = np.eye(n) - (Y @ X) / gam2
    bk2 = (b2 + H12) @ dk12
    ck2 = -dk21 @ (c2 + F12)
    bk1 = -H2 + bk2 @ np.linalg.inv(dk12) @ dk11
    ck1 = F2 + dk11 @ np.linalg.inv(dk21) @ ck2
    ak = A @ Z + H @ Cm @ Z + bk2 @ np.linalg.inv(dk12) @ ck1

    # ── Reverse controller scaling (stage I) ────────────────────────────────
    bk1 = bk1 @ sy2
    ck1 = su2 @ ck1
    dk11 = su2 @ dk11 @ sy2
    dk12 = su2 @ dk12
    dk21 = dk21 @ sy2

    # ── Shift the initial D22 term ──────────────────────────────────────────
    ki1, ki2 = bk1.shape[1], bk2.shape[1]
    ko1, ko2 = ck1.shape[0], ck2.shape[0]
    dk22 = np.zeros((ko2, ki2))
    temp = np.linalg.inv(np.eye(dk11.shape[0]) + dk11 @ d220)
    ak = ak - bk1 @ d220 @ temp @ ck1
    bk2 = bk2 - bk1 @ d220 @ temp @ dk12
    ck2 = ck2 - dk21 @ d220 @ temp @ ck1
    ck1 = temp @ ck1
    dk22 = dk22 - dk21 @ d220 @ temp @ dk12
    dk12 = temp @ dk12
    dk11 = temp @ dk11

    temp = np.eye(d220.shape[0]) - d220 @ dk11
    bk1 = bk1 @ temp
    dk21 = dk21 @ temp

    # ── Descriptor -> state space ──────────────────────────────────────────
    singular_flag = 0.0
    if np.linalg.cond(Z) > 1.0 / np.finfo(float).eps:
        singular_flag = 1.0

    bk_full = np.hstack([bk1, bk2])
    ck_full = np.vstack([ck1, ck2])
    dk_full = np.block([[dk11, dk12], [dk21, dk22]])
    ak_ss, bk_ss, ck_ss, dk_ss = _descr2ss_regular(ak, bk_full, ck_full, dk_full, Z)

    bk1_f = bk_ss[:, :ki1]
    ck1_f = ck_ss[:ko1, :]
    dk11_f = dk_ss[:ko1, :ki1]

    K = sig.StateSpace(ak_ss, bk1_f, ck1_f, dk11_f)
    return K, (singular_flag if singular_flag else gamma_out)


def _sdxi(csys, T, n_meas, n_ctrl, gamma, sdhimod_type, method):
    """Port of ``sdxi`` (nested in sdhinfreg.m): builds the gamma-scaled
    discrete equivalent and returns (Kgamma, xi0=1/gamma0)."""
    dsys = sdhimod(csys, T, n_meas, n_ctrl, gamma, type=sdhimod_type)
    Kgamma, gamma0 = hinfone(dsys, n_meas, n_ctrl, method=method)
    if gamma0 == 0 or not np.isfinite(gamma0):
        xi0 = np.inf
    else:
        xi0 = 1.0 / gamma0
    return Kgamma, xi0


def sdhinfreg_native(csys, T, n_meas=1, n_ctrl=1, gamma_tol=1e-4, max_outer=50,
                     sdhimod_type='mi', method='sa'):
    """
    Hinf-optimal controller for a sampled-data system -- native discrete
    synthesis (port of ``sdhinfreg.m``'s gamma-iteration, default 'mi'
    discretization + 'sa' (hinfone) method).

    Unlike the bilinear-DT->CT->DT approximation this replaces, this
    builds an EXACT discrete equivalent of the sampled-data Hinf problem
    (``sdhimod``) before ever solving anything, so is not limited to
    O(T)-accuracy for the closed-loop performance it reports.

    Parameters
    ----------
    csys : scipy.signal.StateSpace (continuous)
        Standard 4-block generalized plant. D21 and D22 assumed zero.
    T : float
        Sampling period.
    n_meas, n_ctrl : int
    gamma_tol : float
        Secant-iteration tolerance on gamma.
    max_outer : int
        Cap on the initial bracketing loop's 10x expansions (MATLAB's
        original has no such cap -- added defensively rather than risking
        an infinite loop on a pathological plant).
    sdhimod_type : str
        'mi' (default), 'ch', 'ca', 'ba', or 'ha' -- see ``sdhimod``. All 5
        are exact discretizations of the same problem and agree closely on
        well-conditioned plants; may diverge somewhat on numerically hard
        ones since the gamma-iteration below is a secant search, not a
        true monotonic bisection.
    method : str
        'sa' (default, Safonov-Limebeer-Chiang) or 'gl' (Glover-Doyle) --
        see ``hinfone``.

    Returns
    -------
    K : scipy.signal.StateSpace (discrete)
    gamma : float
        Converged Hinf-norm bound.
    poles : ndarray
        Closed-loop poles (plain ZOH discretization of csys, lft'd with K
        -- matches sdhinfreg.m's own reporting convention).
    """
    gamma1 = 1.0
    Kg, xi1 = _sdxi(csys, T, n_meas, n_ctrl, gamma1, sdhimod_type, method)
    tries = 0
    while xi1 <= 1:
        gamma1 *= 10.0
        Kg, xi1 = _sdxi(csys, T, n_meas, n_ctrl, gamma1, sdhimod_type, method)
        tries += 1
        if tries > max_outer:
            raise np.linalg.LinAlgError(
                "sdhinfreg_native: could not bracket a feasible starting "
                f"gamma after {max_outer} 10x expansions from gamma=1")

    K_opt = Kg
    gamma = gamma1 + 2 * gamma_tol
    Kgamma, xi = _sdxi(csys, T, n_meas, n_ctrl, gamma, sdhimod_type, method)
    if Kgamma is not None:
        K_opt = Kgamma

    n_iter = 0
    while abs(gamma - gamma1) > gamma_tol:
        dXi = (xi - xi1) / (gamma - gamma1)
        gamma1, xi1 = gamma, xi
        if dXi == 0:
            break
        gamma = gamma - (xi - 1) / dXi
        Kgamma, xi = _sdxi(csys, T, n_meas, n_ctrl, gamma, sdhimod_type, method)
        if Kgamma is not None:
            K_opt = Kgamma
        n_iter += 1
        if n_iter > max_outer:
            break

    if K_opt is None:
        raise np.linalg.LinAlgError(
            "sdhinfreg_native: hinfone never returned a controller during "
            "the gamma-iteration")

    # SISO case: sdhinfreg.m does `Kopt = ss(minreal(zpk(Kopt), 1e-3))` --
    # a loose-tolerance pole/zero cancellation. The descriptor-form
    # controller assembly routinely produces near-exactly-cancelling
    # pole/zero pairs (confirmed on the documented dsd_help example: an
    # extra pole at -0.99992, matched by a zero at essentially the same
    # location, that this step removes to recover the true minimal
    # controller).
    if n_ctrl == 1 and n_meas == 1:
        from directsd.zpk.zpk import Zpk
        num, den = sig.ss2tf(K_opt.A, K_opt.B, K_opt.C, K_opt.D)
        zk = Zpk.from_tf(np.atleast_1d(num).ravel(), np.atleast_1d(den).ravel())
        zk = zk.minreal(tol=1e-3)
        num_r, den_r = zk.to_tf()
        Ar, Br, Cr, Dr = sig.tf2ss(num_r, den_r)
        K_opt = sig.StateSpace(Ar, Br, Cr, Dr, dt=T)

    # Final closed-loop poles: plain ZOH-discretized original plant, lft'd
    # with K -- matches sdhinfreg.m's own reporting convention.
    from directsd.sspace.design import _lft
    Pd = sig.StateSpace(csys.A, csys.B, csys.C, csys.D).to_discrete(T, method='zoh')
    dcl = _lft(Pd, K_opt, n_meas, n_ctrl, dt=T)
    poles = np.linalg.eigvals(dcl.A)

    return K_opt, gamma, poles
