"""
Integral error functions for sampled-data systems.

Ports of: quaderr, sdl2err, sd2doferr
"""

import numpy as np
from directsd.sspace.plant import GeneralizedPlant

_trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def quaderr(plant_cl, T, N=4096):
    """
    Squared integral error for a sampled-data closed-loop system.

    Computes the H2-norm squared of a discrete-time closed-loop system
    evaluated over one period via frequency-domain integration.

    Parameters
    ----------
    plant_cl : (num, den) tuple or scipy dlti
        Closed-loop discrete-time transfer function.
    T : float
        Sampling period.
    N : int
        Number of frequency points.

    Returns
    -------
    err : float
        Squared integral error (H2-norm squared).
    """
    try:
        import scipy.signal as sig
    except ImportError:
        raise ImportError("scipy is required.")

    if isinstance(plant_cl, sig.dlti):
        num, den = plant_cl.num, plant_cl.den
    elif isinstance(plant_cl, tuple):
        num, den = plant_cl
    else:
        raise TypeError(f"Unsupported type {type(plant_cl)}")

    w = np.linspace(-np.pi, np.pi, N)
    z = np.exp(1j * w)
    H = np.polyval(num, z) / (np.polyval(den, z) + 1e-300)
    err = _trapezoid(np.abs(H) ** 2, w) / (2 * np.pi)
    return float(np.real(err))


def sdl2err(plant, K, T=None):
    """
    Integral quadratic (L2) error for a sampled-data system.

    Mirrors MATLAB sdl2err.m (ss path)::

        dsysL2 = sdh2simple(sys, K.Ts)          % lift_l2
        err    = K.Ts * norm(lft(dsysL2, K))^2  % squared energy, NOT sqrt

    Parameters
    ----------
    plant : scipy.signal.StateSpace or (num, den) tuple or scipy lti
        Continuous-time generalised plant (preferred) or SISO plant.
        Pass the **same** plant that was used for ``sdl2`` design to get
        results comparable to MATLAB's ``sdl2err(sys, K)``.
        For SISO plants a standard regulation generalized plant is constructed
        automatically via ``_parse_plant``.
    K : (num, den) tuple or scipy.signal.StateSpace (discrete)
        Discrete-time controller.
    T : float, optional
        Sampling period (inferred from K when K is a StateSpace/dlti).

    Returns
    -------
    err : float
        Square-root of L2 cost  = sqrt(T · ||Fcl||²_H2).
    """
    import scipy.signal as sig
    from directsd.design.lifting import lift_l2, lft_dt
    from directsd.design.polynomial import _parse_plant
    from directsd.linalg.minreal import Minreal
    from directsd.analysis.norms import _h2norm_dt, _K_to_ss

    K_ss, T = _K_to_ss(K, T)

    plant_ss, n_meas, n_ctrl = _parse_plant(plant)

    n_K_in = K_ss.B.shape[1]
    if n_K_in < n_meas:
        raise ValueError(
            f"sdl2err: controller has {n_K_in} input(s) but plant has "
            f"n_meas={n_meas} measurement output(s). "
            f"For 2-DOF systems call sd2doferr(sys, K_fb, K_ff, T) instead."
        )

    plant_min = Minreal.ss(plant_ss)

    try:
        dsysL2 = lift_l2(plant_min, T, n_meas=n_meas, n_ctrl=n_ctrl)
    except Exception as exc:
        import warnings
        warnings.warn(f"sdl2err: lift_l2 failed ({exc})")
        return float('nan')

    try:
        dcl = lft_dt(dsysL2, K_ss, n_meas=n_meas, n_ctrl=n_ctrl)
        h2n = _h2norm_dt(dcl.A, dcl.B, dcl.C, dcl.D)
    except Exception as exc:
        import warnings
        warnings.warn(f"sdl2err: LFT/norm failed ({exc})")
        return float('nan')

    return float(max(T * h2n ** 2, 0.0))   # L2 cost = T * ‖·‖²_H2  (matches MATLAB sdl2err.m)


def sd2doferr(sys, Kfb, Kff, T=None):
    """
    CT L2 tracking error for a 2-DOF sampled-data system.

    Parameters
    ----------
    sys : scipy.signal.StateSpace
        Full continuous-time generalized plant with structure::

            [z ]   [P11  P12] [d]
            [y1] = [P21   0 ] [u]
            [y2]   [ 0   P22]

        Must have exactly 1 performance output, 2 measurement outputs
        (last 2 rows), 1 exogenous input, and 1 control input (last column).
    Kfb : (num, den) tuple or scipy.signal.StateSpace (discrete)
        Feedback controller acting on measurement y2 (plant-output channel).
    Kff : (num, den) tuple or scipy.signal.StateSpace (discrete)
        Reference feedforward controller acting on measurement y1 (reference
        channel).
    T : float, optional
        Sampling period (inferred from Kfb/Kff when they are StateSpace/dlti).

    Returns
    -------
    err : float
        Squared CT L2 tracking error  T·‖F_cl‖²_H2.
    """
    import scipy.signal as sig
    import scipy.linalg
    from directsd.design.lifting import lift_l2, lft_dt
    from directsd.linalg.minreal import Minreal
    from directsd.analysis.norms import _h2norm_dt, _K_to_ss

    Kff_ss, T = _K_to_ss(Kff, T)
    Kfb_ss, _  = _K_to_ss(Kfb, T)

    # Build 2-input 1-output DT controller:  u = Kff*y1 + Kfb*y2
    n_ff = Kff_ss.A.shape[0]
    n_fb = Kfb_ss.A.shape[0]
    A_k = scipy.linalg.block_diag(Kff_ss.A, Kfb_ss.A)
    B_k = np.block([
        [Kff_ss.B, np.zeros((n_ff, 1))],
        [np.zeros((n_fb, 1)), Kfb_ss.B],
    ])
    C_k = np.hstack([Kff_ss.C, Kfb_ss.C])
    D_k = np.hstack([Kff_ss.D, Kfb_ss.D])
    K_2dof = sig.StateSpace(A_k, B_k, C_k, D_k, dt=T)

    if isinstance(sys, GeneralizedPlant):
        plant_ss = sys.to_statespace()
        n_meas, n_ctrl = sys.n_meas, sys.n_ctrl
    elif isinstance(sys, sig.StateSpace):
        plant_ss = sys
        n_meas, n_ctrl = 2, 1
    elif isinstance(sys, sig.lti):
        plant_ss = sys.to_ss()
        n_meas, n_ctrl = 2, 1
    else:
        raise TypeError(f"sd2doferr: sys must be a StateSpace or lti, got {type(sys)}")

    plant_min = Minreal.ss(plant_ss)

    dsysL2 = lift_l2(plant_min, T, n_meas=n_meas, n_ctrl=n_ctrl)
    dcl    = lft_dt(dsysL2, K_2dof, n_meas=n_meas, n_ctrl=n_ctrl)

    h2n = _h2norm_dt(dcl.A, dcl.B, dcl.C, dcl.D)
    return float(max(T * h2n ** 2, 0.0))
