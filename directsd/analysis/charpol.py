"""
Characteristic polynomial and stability analysis.

Ports of: charpol, sdmargin
"""

import numpy as np


def charpol(plant, K, T=None, hold='zoh'):
    """
    Characteristic polynomial of a sampled-data closed-loop system.

    Parameters
    ----------
    plant : (num, den) tuple or scipy lti
        Continuous-time plant P22 channel.
    K : (num, den) tuple or scipy dlti
        Discrete-time controller.
    T : float, optional
        Sampling period (required when K is a tuple).
    hold : str
        Hold type: 'zoh' (default).

    Returns
    -------
    Delta : np.ndarray
        Monic characteristic polynomial coefficients.
    """
    try:
        import scipy.signal as sig
    except ImportError:
        raise ImportError("scipy is required.")

    if isinstance(K, sig.dlti):
        T = K.dt
        Knum, Kden = K.num, K.den
    elif isinstance(K, tuple):
        Knum, Kden = K
        if T is None:
            raise ValueError("T must be provided when K is a tuple")
    else:
        raise TypeError(f"Unsupported controller type {type(K)}")

    if isinstance(plant, sig.lti):
        Pnum, Pden = plant.num, plant.den
    elif isinstance(plant, tuple):
        Pnum, Pden = plant
    else:
        raise TypeError(f"Unsupported plant type {type(plant)}")

    from directsd.polynomial.transforms import dtfm
    D22num, D22den = dtfm((Pnum, Pden), T, hold=hold)

    # 1 + D22*K  =>  D22_den * K_den + D22_num * K_num  (over D22_den * K_den)
    char_num = np.polyadd(
        np.polymul(D22den, Kden),
        np.polymul(D22num, Knum)
    )

    # Normalize to monic
    char_num = np.real_if_close(char_num / char_num[0], tol=1e6)
    return char_num


def sdmargin(plant, K, T=None):
    """
    Stability margin for a sampled-data system.

    Parameters
    ----------
    plant : (num, den) tuple or scipy lti
    K : (num, den) tuple or scipy dlti
    T : float, optional

    Returns
    -------
    margin : float
        Minimum distance of closed-loop poles from the unit circle boundary
        (positive = stable, negative = unstable).
    poles : np.ndarray
        Closed-loop poles in z-plane.
    """
    Delta = charpol(plant, K, T)
    poles = np.roots(Delta)
    # Distance from unit circle: 1 - |pole| for each pole
    # Stability margin = min(1 - |pole|): positive means all poles inside unit circle
    margin = float(1.0 - np.max(np.abs(poles)))
    return margin, poles
