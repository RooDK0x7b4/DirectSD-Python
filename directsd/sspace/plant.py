"""
GeneralizedPlant — the standard LFT augmented plant for sampled-data H2/Hinf design.

In the DirectSD toolbox (and H∞ control theory generally), the *generalized plant*
(also called augmented plant) partitions the plant into four sub-blocks::

        [z]   [P11  P12] [w]
        [y] = [P21  P22] [u]

where:
    w — exogenous inputs (disturbances, reference signals)
    u — control inputs
    z — performance outputs (signals to be minimised)
    y — measurement outputs (signals available to the controller)

This class provides the Python equivalent of MATLAB's natural matrix concatenation::

    sys = [F*Fw  F          ← performance rows
           0     rho        ← (can be multiple performance rows)
          -F*Fw  -F]        ← measurement row

Usage::

    from directsd import GeneralizedPlant

    sys = GeneralizedPlant([
        [F*Fw,   F ],
        [0,      rho],
        [-F*Fw, -F ],
    ])                       # 3-output × 2-input, n_meas=1, n_ctrl=1 by default

    K, cost = sdh2(sys, T=0.1)
"""

import numpy as np
import scipy.signal as sig
from directsd.tf.interconnect import to_lti


class GeneralizedPlant:
    """
    Standard generalized (augmented) plant for sampled-data H2/Hinf design.

    Parameters
    ----------
    rows : list[list] or scipy.signal.StateSpace
        Either a 2-D grid of SISO blocks (scalars, ``(num, den)`` tuples, or
        ``scipy.signal.lti`` objects), or a pre-built ``StateSpace``.
        The grid shape determines the total number of outputs and inputs.
    n_perf : int, optional
        Number of performance outputs (z-rows). Default: ``nout - n_meas``.
    n_meas : int, optional
        Number of measurement outputs (y-rows). Default: ``1``.
    n_dist : int, optional
        Number of exogenous inputs (w-columns). Default: ``nin - n_ctrl``.
    n_ctrl : int, optional
        Number of control inputs (u-columns). Default: ``1``.

    Attributes
    ----------
    n_perf, n_meas, n_dist, n_ctrl : int
        Partition sizes.
    A, B, C, D : numpy.ndarray
        State-space matrices (delegated from the internal StateSpace).

    Examples
    --------
    Standard 1-DOF H2 plant (3 outputs, 2 inputs)::

        sys = GeneralizedPlant([
            [_mul(Fw, F),   F ],   # performance: weighted output
            [0,             rho],  # performance: control cost
            [_neg(_mul(Fw, F)), _neg(F)],  # measurement
        ])

    2-DOF plant (3 outputs, 2 inputs, n_meas=2)::

        sys = GeneralizedPlant([
            [_neg(F), _neg(F)],    # performance
            [1,       0      ],    # measurement 1: reference
            [0,       _neg(F)],    # measurement 2: plant output
        ], n_meas=2)
    """

    def __init__(self, rows, *, n_perf=None, n_meas=1, n_dist=None, n_ctrl=1):
        if isinstance(rows, sig.StateSpace):
            ss = rows
        else:
            ss = _build_ss(rows)

        nout = ss.C.shape[0]
        nin = ss.B.shape[1]

        self._ss = ss
        self.n_meas = int(n_meas)
        self.n_ctrl = int(n_ctrl)
        self.n_perf = int(nout - n_meas) if n_perf is None else int(n_perf)
        self.n_dist = int(nin - n_ctrl) if n_dist is None else int(n_dist)

    # ── State-space matrix access ──────────────────────────────────────────────

    @property
    def A(self):
        return self._ss.A

    @property
    def B(self):
        return self._ss.B

    @property
    def C(self):
        return self._ss.C

    @property
    def D(self):
        return self._ss.D

    # ── Interop ────────────────────────────────────────────────────────────────

    def to_statespace(self) -> sig.StateSpace:
        """Return the underlying ``scipy.signal.StateSpace``."""
        return self._ss

    @property
    def P22(self) -> sig.StateSpace:
        """
        Extract P22 sub-block (lower-right corner: y ← u channel).

        Returns a ``StateSpace`` with shared A matrix and the last
        ``n_meas`` output rows and last ``n_ctrl`` input columns.
        """
        A = self._ss.A
        B2 = self._ss.B[:, -self.n_ctrl:]
        C2 = self._ss.C[-self.n_meas:, :]
        D22 = self._ss.D[-self.n_meas:, -self.n_ctrl:]
        return sig.StateSpace(A, B2, C2, D22)

    def __repr__(self) -> str:
        nout = self._ss.C.shape[0]
        nin = self._ss.B.shape[1]
        n = self._ss.A.shape[0]
        return (
            f"GeneralizedPlant({nout}×{nin}, states={n}, "
            f"n_perf={self.n_perf}, n_meas={self.n_meas}, "
            f"n_dist={self.n_dist}, n_ctrl={self.n_ctrl})"
        )


# ── Internal block-matrix assembler ───────────────────────────────────────────

_coerce_lti = to_lti  # alias used by _block_abcd below


def _block_abcd(P):
    """Convert a SISO block to (A, B, C, D) arrays with shape (n,), (n,1), (1,n), (1,1)."""
    if isinstance(P, list) and len(P) == 1:
        P = P[0]
    if np.isscalar(P) or (isinstance(P, np.ndarray) and P.ndim == 0):
        return (np.zeros((0, 0)), np.zeros((0, 1)),
                np.zeros((1, 0)), np.array([[float(P)]]))
    ss = _coerce_lti(P).to_ss()
    n = ss.A.shape[0]
    return (ss.A,
            ss.B.reshape(n, 1),
            ss.C.reshape(1, n),
            ss.D.reshape(1, 1))


def _build_ss(rows) -> sig.StateSpace:
    """
    Assemble a MIMO StateSpace from a 2-D list of SISO blocks.

    Each block gets independent state variables (block-diagonal A matrix).
    """
    nr = len(rows)
    nc = len(rows[0])
    blk = [[_block_abcd(rows[i][j]) for j in range(nc)] for i in range(nr)]
    ns = [[blk[i][j][0].shape[0] for j in range(nc)] for i in range(nr)]
    ntot = sum(ns[i][j] for i in range(nr) for j in range(nc))

    A = np.zeros((ntot, ntot))
    B = np.zeros((ntot, nc))
    C = np.zeros((nr, ntot))
    D = np.zeros((nr, nc))

    off = 0
    for j in range(nc):
        for i in range(nr):
            Aij, Bij, Cij, Dij = blk[i][j]
            nij = ns[i][j]
            if nij > 0:
                A[off:off + nij, off:off + nij] = Aij
                B[off:off + nij, j:j + 1] = Bij
                C[i:i + 1, off:off + nij] = Cij
            D[i, j] = Dij[0, 0]
            off += nij

    return sig.StateSpace(A, B, C, D)
