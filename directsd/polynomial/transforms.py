"""
Discrete transforms for sampled-data systems.

Ports of: ztrm (modified Z-transform), dtfm (discrete Laplace with hold)
"""

import numpy as np


def dtfm(G, T, hold='zoh', method='residue'):
    """
    Compute discrete-time transfer function with a hold from a continuous-time
    transfer function G(s) (modified Z-transform / pulse transfer function).

    Parameters
    ----------
    G : scipy.signal.lti or (num, den) tuple
        Continuous-time transfer function.
    T : float
        Sampling period.
    hold : str
        'zoh' (zero-order hold, default) or 'foh' (first-order hold).
    method : str
        'residue' (default) or 'matrix'.

    Returns
    -------
    Gd : (num_d, den_d) tuple
        Discrete-time numerator and denominator coefficient arrays.
    """
    try:
        import scipy.signal as sig
    except ImportError:
        raise ImportError("scipy is required for dtfm. Install with: pip install scipy")

    # Normalise input to (num, den)
    if isinstance(G, sig.lti):
        num, den = G.num, G.den
    elif isinstance(G, tuple) and len(G) == 2:
        num, den = G
    else:
        raise TypeError("G must be a scipy lti or (num, den) tuple")

    ct_sys = sig.lti(num, den)
    if hold == 'zoh':
        dt_sys = ct_sys.to_discrete(T, method='zoh')
    elif hold == 'foh':
        dt_sys = ct_sys.to_discrete(T, method='foh')
    else:
        raise ValueError(f"Unknown hold type '{hold}'")

    return dt_sys.num, dt_sys.den


def ztrm(G, T, mu=0.0):
    """
    Modified Z-transform (advanced Z-transform) of a transfer matrix.

    Computes Z{G(s) * e^{-mu*T*s}} / the Z-transform evaluated with the
    advance parameter mu in [0, 1].

    For mu=0 this equals the standard Z-transform with ZOH.

    Parameters
    ----------
    G : (num, den) tuple or scipy lti
        Continuous-time SISO transfer function.
    T : float
        Sampling period.
    mu : float
        Advance parameter in [0, 1].

    Returns
    -------
    num_d, den_d : np.ndarray
        Numerator and denominator of discrete transfer function.
    """
    if not (0.0 <= mu <= 1.0):
        raise ValueError("mu must be in [0, 1]")

    try:
        import scipy.signal as sig
        import scipy.linalg as la
    except ImportError:
        raise ImportError("scipy is required for ztrm. Install with: pip install scipy")

    if isinstance(G, sig.lti):
        num, den = G.num, G.den
    elif isinstance(G, tuple):
        num, den = G
    else:
        raise TypeError("G must be a scipy lti or (num, den) tuple")

    # Get poles and residues
    r, p, k = sig.residue(num, den)

    # Compute modified Z-transform via residue method:
    # Z_mu{e^{p*t}} = z * e^{p*T*(1-mu)} / (z - e^{p*T})
    z_poles = np.exp(p * T)
    z_gains = r * np.exp(p * T * (1 - mu))

    # Build discrete transfer function from partial fractions
    # Reconstruct from poles and gains
    num_d = np.array([0.0])
    den_d = np.array([1.0])

    for gi, pi in zip(z_gains, z_poles):
        # Add term gi*z / (z - pi)
        term_num = np.array([gi, 0.0])
        term_den = np.array([1.0, -pi])
        # Add fractions
        new_num = np.polymul(num_d, term_den) + np.polymul(term_num, den_d)
        den_d = np.polymul(den_d, term_den)
        num_d = new_num

    # Add polynomial (direct) part
    if np.any(np.abs(k) > 1e-10):
        k_poly = np.atleast_1d(k)
        num_d = np.polyadd(num_d, np.polymul(k_poly, den_d))

    num_d = np.real_if_close(num_d, tol=1e6)
    den_d = np.real_if_close(den_d, tol=1e6)

    return num_d, den_d
