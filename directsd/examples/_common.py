"""
Shared helpers for the example/demo scripts (``demos.py``, ``help_examples.py``,
``examples.py``).

Centralizes ``_cl_poles``, which was found duplicated verbatim across
``demos.py`` and ``help_examples.py`` -- two independent copies of the same
logic that could silently diverge if one got fixed and not the other.

Also re-exports ``_z2zeta`` from ``design.polynomial`` for the "pre-convert a
whole plant matrix before calling ``dhinf``" pattern (MATLAB's own
``sys = z2zeta(sys)`` before ``demo_dhinf.m``'s calls, since ``dhinf`` applies
``z2zeta`` again internally -- skipping this pre-conversion hands the pipeline
the zeta-domain functions instead of the z-domain ones, flipping every
stability-based factor split). ``_z2zeta`` is one function for any input
shape (a single ``(num, den)`` pair, a ``Zpk``, or a matrix of pairs) -- see
its own docstring in ``design/polynomial.py``.
"""

from __future__ import annotations
import numpy as np

from directsd.analysis.charpol import charpol
from directsd.design.polynomial import _z2zeta


def cl_poles(plant, K, T):
    """Closed-loop z-domain poles via the characteristic polynomial.

    Returns an empty array (rather than raising) if ``charpol`` fails --
    some example plants are pathological enough that this is expected.
    """
    try:
        delta = charpol(plant, K, T)
        return np.roots(delta)
    except Exception:
        return np.array([])
