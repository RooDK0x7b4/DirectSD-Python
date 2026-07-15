"""
Transfer-function block diagram arithmetic.

Provides scalar/tuple/(num,den)/lti-compatible operations for building
interconnected SISO transfer functions before assembling them into a
GeneralizedPlant.

Functions
---------
to_lti   -- coerce any TF-like input to scipy.signal.lti
nd       -- extract (num, den) 1-D arrays from any TF-like input
mul      -- series (cascade) connection: P1 * P2 * ...
neg      -- negate: -P
add      -- parallel (sum) connection: P + Q
feedback -- unity negative feedback: P / (1 + P)
nd_mul   -- series connection, returns (num, den) tuple
nd_neg   -- negate, returns (num, den) tuple
"""

import numpy as np
import scipy.signal as sig


def to_lti(P):
    """
    Coerce any TF-like input to a ``scipy.signal.lti`` object.

    Accepts: scalar, ``(num, den)`` tuple, ``scipy.signal.lti`` subclass,
    ``TransferFunction``, ``ZerosPolesGain``, or ``StateSpace``.
    """
    if isinstance(P, (sig.lti, sig.TransferFunction,
                      sig.ZerosPolesGain, sig.StateSpace)):
        return P
    if isinstance(P, tuple) and len(P) == 2:
        return sig.TransferFunction(*P)
    if np.isscalar(P):
        return sig.TransferFunction([float(P)], [1.0])
    raise TypeError(f"Cannot coerce {type(P)} to lti")


def nd(P):
    """
    Extract ``(num, den)`` 1-D arrays from any TF-like input.

    Accepts the same types as :func:`to_lti`, plus raw ``(num, den)`` tuples.
    """
    if np.isscalar(P):
        return np.array([float(P)]), np.array([1.0])
    if isinstance(P, tuple) and len(P) == 2:
        return (np.atleast_1d(np.array(P[0], float)).ravel(),
                np.atleast_1d(np.array(P[1], float)).ravel())
    if hasattr(P, 'to_tf'):
        P = P.to_tf()
    if isinstance(P, (sig.lti, sig.TransferFunction)):
        return (np.atleast_1d(np.array(P.num, float)).ravel(),
                np.atleast_1d(np.array(P.den, float)).ravel())
    raise TypeError(f"Cannot extract (num, den) from {type(P)}")


def mul(*plants):
    """Series (cascade) connection of two or more transfer functions."""
    n, d = nd(plants[0])
    for p in plants[1:]:
        pn, pd = nd(p)
        n = np.polymul(n, pn)
        d = np.polymul(d, pd)
    return sig.TransferFunction(n, d)


def neg(P):
    """Negate a transfer function: returns ``-P``."""
    n, d = nd(P)
    return sig.TransferFunction(-n, d)


def add(P, Q):
    """Parallel (sum) connection: returns ``P + Q``."""
    pn, pd = nd(P)
    qn, qd = nd(Q)
    return sig.TransferFunction(
        np.polyadd(np.polymul(pn, qd), np.polymul(qn, pd)),
        np.polymul(pd, qd),
    )


def feedback(P):
    """Unity negative feedback: returns ``P / (1 + P)``."""
    pn, pd = nd(P)
    return sig.TransferFunction(pn, np.polyadd(pd, pn))


def nd_mul(*plants):
    """Series connection returning a ``(num, den)`` tuple."""
    n, d = nd(plants[0])
    for p in plants[1:]:
        pn, pd = nd(p)
        n = np.polymul(n, pn)
        d = np.polymul(d, pd)
    return (n, d)


def nd_neg(P):
    """Negate, returning a ``(num, den)`` tuple."""
    n, d = nd(P)
    return (-n, d)
