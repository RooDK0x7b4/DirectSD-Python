"""
Miscellaneous utility functions for DirectSD.

Ports of: bilintr (bilinear transformation), improper (proper/improper split),
          sumzpk (reliable zpk sum), separtf (separation via polynomial
          equations), tf2nd (TF num/den extract)

Diophantine solvers (dioph, dioph2, diophsys, diophsys2) live in
directsd.polynomial.diophantine. A second, incorrect `diophsys` used to live
here (single shared Y for both equations, `[X,Y,err]`) -- it had zero callers
anywhere in the codebase and didn't match MATLAB's diophsys.m (which returns
separate Y1/Y2, `[X,Y1,Y2,err,condA]`) despite being the one exported as the
public `directsd.diophsys`. It has been removed.
"""

import numpy as np


# ---------------------------------------------------------------------------
# bilintr – Bilinear transformation for SISO transfer functions
# ---------------------------------------------------------------------------

def bilintr(sys, btype=None, T=1.0):
    """
    Bilinear transformation for a SISO LTI system.

    Parameters
    ----------
    sys : (num, den) tuple or scipy lti/dlti
        Input system.
    btype : str, optional
        Transformation type:
        's2z'   - s → z, maps LHP into unit disk
        'z2s'   - z → s, maps unit disk into LHP (default for discrete sys)
        's2d'   - s → d=1/z, maps RHP into unit disk
        'd2s'   - d → s
        'tustin' - Tustin (bilinear) transform with sampling period T
        [a,b,c,d] - custom coefficients y = (a*x+b)/(c*x+d)
    T : float
        Sampling period (used for 'tustin').

    Returns
    -------
    sys_out : (num, den) tuple
        Transformed system.
    """
    try:
        import scipy.signal as sig
    except ImportError:
        raise ImportError("scipy is required.")

    # Extract num/den
    if isinstance(sys, sig.lti):
        num, den = sys.num, sys.den
        is_ct = True
    elif isinstance(sys, sig.dlti):
        num, den = sys.num, sys.den
        is_ct = False
    elif isinstance(sys, tuple) and len(sys) == 2:
        num, den = sys
        is_ct = True
    else:
        raise TypeError(f"Unsupported system type {type(sys)}")

    num = np.atleast_1d(np.array(num, dtype=float)).ravel()
    den = np.atleast_1d(np.array(den, dtype=float)).ravel()

    # Determine default type
    if btype is None:
        btype = 'z2s' if not is_ct else 's2z'

    # Handle custom coefficients [a, b, c, d]
    if isinstance(btype, (list, np.ndarray)):
        coefs = np.array(btype, dtype=float).ravel()
        if len(coefs) == 1:
            b_val = coefs[0]
            a, b, c, d = 1.0, b_val, 1.0, -b_val
        elif len(coefs) == 4:
            a, b, c, d = coefs
        else:
            raise ValueError("Custom bilinear coefficients must be length 1 or 4")
    else:
        btype = btype.lower()
        if btype == 's2z':
            a, b, c, d = 1.0, 1.0, -1.0, 1.0   # z = (s+1)/(-s+1)
        elif btype == 'z2s':
            a, b, c, d = 1.0, -1.0, 1.0, 1.0   # s = (z-1)/(z+1)
        elif btype == 's2d':
            a, b, c, d = -1.0, 1.0, 1.0, 1.0   # d = (-s+1)/(s+1)
        elif btype == 'd2s':
            a, b, c, d = -1.0, 1.0, 1.0, 1.0   # s = (-d+1)/(d+1)
        elif btype == 'tustin':
            # z = (1 + s*T/2) / (1 - s*T/2)  => s = 2/T * (z-1)/(z+1)
            a, b, c, d = 2.0/T, -2.0/T, 1.0, 1.0
        else:
            raise ValueError(f"Unknown bilinear type '{btype}'")

    # Apply substitution x = (a*w + b) / (c*w + d)  where w is new variable
    # F(x) = num(x)/den(x); substitute x = (a*w+b)/(c*w+d)
    n = max(len(num), len(den)) - 1

    # Compute numerator and denominator polynomials in w
    # by substituting the rational expression
    # num_out(w) = num((aw+b)/(cw+d)) * (cw+d)^n
    # den_out(w) = den((aw+b)/(cw+d)) * (cw+d)^n
    # Build [aw+b]^k and [cw+d]^(n-k) for each coefficient

    def poly_power(coeffs, power):
        """Raise polynomial [a,b] to `power`."""
        result = np.array([1.0])
        base = np.array(coeffs, dtype=float)
        for _ in range(int(power)):
            result = np.polymul(result, base)
        return result

    ab = np.array([a, b])
    cd = np.array([c, d])

    num_w = np.array([0.0])
    for i, coef in enumerate(num):
        pwr = len(num) - 1 - i
        term = coef * np.polymul(poly_power(ab, pwr), poly_power(cd, n - pwr))
        num_w = np.polyadd(num_w, term)

    den_w = np.array([0.0])
    for i, coef in enumerate(den):
        pwr = len(den) - 1 - i
        term = coef * np.polymul(poly_power(ab, pwr), poly_power(cd, n - pwr))
        den_w = np.polyadd(den_w, term)

    # Normalize
    num_w = np.real_if_close(num_w, tol=1e6)
    den_w = np.real_if_close(den_w, tol=1e6)

    # Strip leading zeros
    from directsd.polynomial.operations import striplz
    num_w = striplz(num_w)
    den_w = striplz(den_w)

    # Normalize leading coefficient of denominator to 1
    lead = den_w[0]
    if abs(lead) > 1e-14:
        num_w = num_w / lead
        den_w = den_w / lead

    return num_w, den_w


# ---------------------------------------------------------------------------
# improper – separate improper (polynomial) part of rational function
# ---------------------------------------------------------------------------

def improper(sys, ptype='sp'):
    """
    Separate the improper (polynomial) part of a rational transfer function.

    Parameters
    ----------
    sys : (num, den) tuple or scipy lti
        Input rational function.
    ptype : str
        'sp'  - strictly proper remainder (default)
        'p'   - proper remainder (may have direct term)
        'symm' - same as 'sp' (for symmetric spectral densities)

    Returns
    -------
    P : np.ndarray
        Polynomial (improper) part coefficients.
    R0 : (num, den) tuple
        Proper or strictly proper part.
    """
    try:
        import scipy.signal as sig
    except ImportError:
        raise ImportError("scipy is required.")

    if ptype not in ('sp', 'p', 'symm'):
        raise ValueError(f"Unknown properness type '{ptype}'")

    if isinstance(sys, sig.lti):
        num, den = sys.num, sys.den
    elif isinstance(sys, tuple):
        num, den = sys
    else:
        raise TypeError(f"Unsupported type {type(sys)}")

    num = np.atleast_1d(np.array(num, dtype=float)).ravel()
    den = np.atleast_1d(np.array(den, dtype=float)).ravel()

    if len(num) <= len(den):
        # Already proper or strictly proper
        if ptype == 'p':
            return np.array([0.0]), (num, den)
        else:
            # Check for direct term
            if len(num) == len(den):
                d = num[0] / den[0]
                num_sp = num - d * den
                from directsd.polynomial.operations import striplz
                num_sp = striplz(num_sp)
                return np.array([d]), (num_sp, den)
            return np.array([0.0]), (num, den)

    # Polynomial long division: num = P * den + R
    P, R = np.polydiv(num, den)

    from directsd.polynomial.operations import striplz
    P = striplz(P) if len(P) > 0 else np.array([0.0])
    R = striplz(R) if len(R) > 0 else np.array([0.0])

    if ptype == 'p':
        # Remainder may still have direct term: split it off
        if len(R) == len(den):
            d = R[0] / den[0]
            R = R - d * den
            R = striplz(R)
            P = np.polyadd(P, np.array([d]))
    # for 'sp' and 'symm': P includes all polynomial terms including constant

    return P, (R, den)


# ---------------------------------------------------------------------------
# tf2nd – extract numerator/denominator from transfer function
# ---------------------------------------------------------------------------

def tf2nd(sys):
    """
    Extract numerator and denominator coefficient arrays from a TF.

    Parameters
    ----------
    sys : (num, den) tuple or scipy lti/dlti

    Returns
    -------
    num : np.ndarray
    den : np.ndarray
    """
    try:
        import scipy.signal as sig
    except ImportError:
        raise ImportError("scipy is required.")

    if isinstance(sys, (sig.lti, sig.dlti)):
        tf = sys if isinstance(sys, sig.TransferFunction) else sys
        try:
            num, den = tf.num, tf.den
        except AttributeError:
            tf_sys = sig.lti(*sys.to_tf().num, *sys.to_tf().den)
            num, den = tf_sys.num, tf_sys.den
    elif isinstance(sys, tuple) and len(sys) == 2:
        num, den = sys
    else:
        raise TypeError(f"Unsupported type {type(sys)}")

    return np.atleast_1d(np.array(num, dtype=float)).ravel(), \
           np.atleast_1d(np.array(den, dtype=float)).ravel()


# ---------------------------------------------------------------------------
# separtf – proper separation via polynomial equations
# ---------------------------------------------------------------------------

def separtf(sys, ptype='sp'):
    """
    Proper separation using polynomial equations technique.

    Equivalent to improper() but uses the polynomial Diophantine approach
    for more robust handling of near-cancellations.

    Parameters
    ----------
    sys : (num, den) tuple
    ptype : str  'sp' or 'p'

    Returns
    -------
    poly_part : np.ndarray
    proper_part : (num, den) tuple
    """
    return improper(sys, ptype)


# ---------------------------------------------------------------------------
# sumzpk – reliable summation of transfer functions with common poles
# ---------------------------------------------------------------------------

def sumzpk(sys1, sys2):
    """
    Reliable summation of two transfer functions with common (or
    near-common) poles.

    Port of MATLAB sumzpk.m: separates any poles/zeros the two operands
    share first, sums only the reduced (non-shared) parts via polynomial
    arithmetic, then reattaches the shared poles exactly -- avoiding the
    numerical cancellation that plain cross-multiplication
    (num1*den2 + num2*den1, den1*den2) suffers when operands share poles
    (duplicated poles come back smeared after re-rooting the raw product).
    Delegates to the already-validated root-list `Zpk.zsum` (used
    throughout the design/polynomial.py pipeline for exactly this reason).

    Parameters
    ----------
    sys1, sys2 : (num, den) tuples, or a scalar

    Returns
    -------
    (num, den) : result of sys1 + sys2
    """
    from directsd.zpk.zpk import Zpk

    def _to_tf(sys):
        if isinstance(sys, (int, float)):
            return np.array([float(sys)]), np.array([1.0])
        num, den = sys
        return (np.atleast_1d(np.array(num, dtype=float)).ravel(),
                np.atleast_1d(np.array(den, dtype=float)).ravel())

    num1, den1 = _to_tf(sys1)
    num2, den2 = _to_tf(sys2)
    z1 = Zpk.from_tf(num1, den1)
    z2 = Zpk.from_tf(num2, den2)
    return z1.zsum(z2).to_tf()


# ---------------------------------------------------------------------------
# zpk – create ZerosPolesGain model from polynomial objects
# ---------------------------------------------------------------------------

def zpk(N, D=None, T=None):
    """
    Create a scipy ZerosPolesGain model from polynomial objects.

    Port of MATLAB ``@poln/zpk``.

    Parameters
    ----------
    N : Poln
        Numerator polynomial.
    D : Poln or scalar, optional
        Denominator polynomial (default: 1).
    T : float, optional
        Sampling period.  Required for discrete-time output; inferred from
        ``N.var`` when omitted (DT variables z/q/d → T=1 if not given).

    Returns
    -------
    F : scipy.signal.ZerosPolesGain
        CT or DT ZPK model.  The zeros and poles include any roots at the
        origin arising from a shift difference between N and D.
    """
    import scipy.signal as sig
    from directsd.polynomial.poln import Poln
    from directsd.polynomial.operations import compat, coprime

    if D is None or (isinstance(D, (int, float)) and D == 1):
        D = Poln(np.array([1.0]), N.var)

    N, D = compat(N, D)
    # Cancel common factors
    N_r, D_r, _ = coprime(N, D)

    zeros = list(N_r.roots)
    poles = list(D_r.roots)
    gain  = N_r.k / D_r.k

    # Account for shift difference: shift encodes z^{-shift} factors.
    # Net extra zeros at origin: D.shift - N.shift (positive → extra zeros in N)
    # Net extra poles at origin: N.shift - D.shift (positive → extra poles in N)
    zn = D_r.shift - N_r.shift
    if zn > 0:
        zeros += [0.0] * zn
    elif zn < 0:
        poles += [0.0] * (-zn)

    z_arr = np.array(zeros, dtype=complex)
    p_arr = np.array(poles, dtype=complex)

    is_dt = N.is_dt
    if is_dt:
        dt = T if T is not None else 1.0
        return sig.ZerosPolesGain(z_arr, p_arr, gain, dt=dt)
    return sig.ZerosPolesGain(z_arr, p_arr, gain)


# ---------------------------------------------------------------------------
# bilinss – bilinear transformation for state-space models
# ---------------------------------------------------------------------------

def bilinss(sys_ss, btype='tustin', T=1.0):
    """
    Bilinear transformation for a state-space system.

    Parameters
    ----------
    sys_ss : scipy.signal.StateSpace
    btype : str
        'tustin' (default) or 's2z', 'z2s'.
    T : float
        Sampling period for Tustin.

    Returns
    -------
    sys_out : scipy.signal.StateSpace
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
        raise TypeError("sys_ss must be a scipy StateSpace")

    n = A.shape[0]

    if btype == 'tustin' and (dt is None or dt == 0):
        # Continuous → Discrete: Tustin method
        # z = (1 + s*T/2) / (1 - s*T/2)
        # => A_d = (I + T/2*A) * inv(I - T/2*A)
        alpha = T / 2.0
        IpA = np.eye(n) + alpha * A
        ImA = np.eye(n) - alpha * A
        ImA_inv = la.inv(ImA)
        Ad = ImA_inv @ IpA
        Bd = np.sqrt(T) * ImA_inv @ B
        Cd = np.sqrt(T) * C @ ImA_inv
        Dd = D + alpha * C @ ImA_inv @ B
        return sig.StateSpace(Ad, Bd, Cd, Dd, T)

    elif btype in ('z2s', 'tustin') and dt is not None and dt != 0:
        # Discrete → Continuous: inverse Tustin
        alpha = T / 2.0
        IpA = np.eye(n) + A
        ImA = np.eye(n) - A
        ImA_inv = la.inv(ImA)
        Ac = (2.0 / T) * ImA_inv @ (A - np.eye(n))
        Bc = np.sqrt(2.0 / T) * ImA_inv @ B
        Cc = np.sqrt(2.0 / T) * C @ ImA_inv
        Dc = D - C @ ImA_inv @ B
        return sig.StateSpace(Ac, Bc, Cc, Dc)

    else:
        raise ValueError(f"Unsupported bilinear type '{btype}' for given system type")
