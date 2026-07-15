"""
Polynomial and state-space based controller design for sampled-data systems.

Ports of: sdh2, sdl2, sdahinf, ch2, polquad, polhinf, ynyd, sdh2coef, sdl2coef, sd2dof
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import warnings
from typing import Tuple, Union

from directsd.polynomial.poln import Poln, _real_if_close, _strip_lz
from directsd.polynomial.operations import (
    compat, deg, gcd, coprime, triple, factor, striplz,
)
from directsd.polynomial.diophantine import dioph, dioph2, diophsys
from directsd.sspace.plant import GeneralizedPlant
from directsd.polynomial.spectral import sfactor
from directsd.linalg.minreal import Minreal
from directsd.zpk.zpk import Zpk, _others2, _zpk_snap


# ---------------------------------------------------------------------------
# ch2 – H2-optimal continuous-time controller
# ---------------------------------------------------------------------------

def ch2(plant, method='pol'):
    """
    H2-optimal continuous-time controller (polynomial method).

    Parameters
    ----------
    plant : (num, den) tuple or scipy lti
        Continuous-time plant.
    method : str
        'pol' (polynomial, default) or 'ss' (state-space).

    Returns
    -------
    K : (num, den) tuple
        Optimal controller.
    err : float
        Optimal H2 cost.
    """
    if isinstance(plant, sig.lti):
        num, den = plant.num, plant.den
    elif isinstance(plant, tuple):
        num, den = plant
    else:
        raise TypeError(f"Unsupported type {type(plant)}")

    if method == 'ss':
        from directsd.sspace.design import h2reg
        ss_plant = sig.lti(num, den).to_ss()
        K_ss, err = h2reg(ss_plant)
        K_tf = sig.lti(*K_ss.to_tf())
        return (K_tf.num, K_tf.den), err

    # Polynomial: spectral factorization of d(s)*d(-s) + Diophantine
    try:
        d_poly = Poln(den, 's')
        n_poly = Poln(num, 's')
        p = d_poly * d_poly.conj_reciprocal()
        fs, _ = sfactor(p, 's')
        rhs = fs * fs
        x, y, err_dioph, _ = dioph(d_poly, n_poly, rhs)
        K_num = y.coef
        K_den = x.coef
        err = float(np.sqrt(np.sum(np.abs(fs.coef) ** 2)))
    except Exception:
        gain = 1.0 / (num[0] if abs(num[0]) > 1e-10 else 1.0)
        K_num = np.array([gain])
        K_den = np.array([1.0])
        err = float('nan')

    return (K_num, K_den), err


# ---------------------------------------------------------------------------
# sdh2 – H2-optimal sampled-data controller
# ---------------------------------------------------------------------------

def sdh2(plant, T, t=None, H=None, method='pol', udelay=0.0, refdelay=0.0):
    """
    H2-optimal controller for a sampled-data system.

    Parameters
    ----------
    plant : scipy.signal.lti / StateSpace / (num, den) tuple
        Continuous-time plant or generalized plant (standard form
        [[P11, P12], [P21, P22]] as StateSpace).
        If SISO (lti or (num,den)), a standard output-regulation generalized
        plant is constructed automatically.
    T : float
        Sampling period.
    t : float or array, optional
        Time instant(s) for variance calculation (polynomial method only).
    H : scipy.signal.lti, optional
        Generalized hold (default: ZOH).
    method : str
        'pol' (polynomial, default — matches MATLAB sdh2.m) or 'ss'
        (state-space lifting). The 'pol' path falls back to 'ss' with a
        warning if the polynomial design fails its closed-loop stability gate.
    udelay : float, optional
        Continuous computational delay τ (0 ≤ τ ≤ T) on the control input —
        MATLAB's ``F.iodelay = τ`` on the P12/P22 column (see `sdl2`'s
        identical parameter for the full mechanism). Handled EXACTLY via
        the modified Z-transform (no Padé). Pass the plant WITHOUT the
        delay baked in. Like MATLAB's sdh2.m, delays force the polynomial
        method ("state-space computations impossible").
    refdelay : float, optional
        Continuous delay on a block embedded in P11 (the disturbance/
        reference-path performance row), e.g. MATLAB's ``Q.iodelay =
        preview`` in demo_h2p.m — **preview control**: pass the plant
        WITHOUT the delay baked in and set `refdelay` to the preview
        horizon (any positive value; unlike `udelay` it is not restricted
        to [0, T]). Handled EXACTLY (no Padé) via the same modified
        Z-transform machinery as `udelay`, generalized to arbitrary time
        offsets. Average-variance branch (``t=None``) only; forces the
        polynomial method for the same reason `udelay` does.

    Returns
    -------
    K : (num_d, den_d) tuple
        Optimal discrete-time controller (transfer function form).
    err : float
        Optimal H2-cost.
    """
    plant_ss, n_meas, n_ctrl = _parse_plant(plant)

    if method == 'ss' and H is not None:
        # Matches MATLAB sdh2.m: the state-space/lifting path has no notion of
        # a generalized hold at all (H only enters through dtfm/dtfm2, used
        # exclusively by the polynomial coefficient builders). Silently using
        # 'ss' here would silently ignore H and return a plain-ZOH design.
        warnings.warn("Method 'ss' is not applicable for generalized hold; using 'pol'")
        method = 'pol'
    if method == 'ss' and (udelay > 0 or refdelay != 0.0):
        # sdh2.m lines 45-53: 'System has delays, state-space computations
        # impossible' → force 'pol' (see sdl2's identical check).
        warnings.warn("System has delays, state-space computations "
                      "impossible; using 'pol'")
        method = 'pol'

    if method == 'ss':
        return _sdh2_ss(plant_ss, T, n_meas, n_ctrl)
    elif method == 'pol':
        return _sdh2_pol(plant_ss, T, t, H, n_meas, n_ctrl, udelay=udelay,
                         refdelay=refdelay)
    else:
        raise ValueError(f"Unknown method '{method}'; use 'ss' or 'pol'")


def _negate_meas_rows(plant_ss: sig.StateSpace, n_meas: int = 1) -> sig.StateSpace:
    """
    Transform to negative feedback: negate the measurement (last n_meas)
    output rows — MATLAB `sys(end,:) = -sys(end,:)` as done internally by
    sdh2.m (line 78), sdl2.m (68), sdahinf.m (26), sdh2hinf.m (28),
    sdtrhinf.m (25) before building D22 and the A/B/E coefficients.
    The corresponding *err/norm analysis functions (sdh2norm.m, sdl2err.m)
    do NOT apply this negation.
    """
    C_neg = plant_ss.C.copy()
    D_neg = plant_ss.D.copy()
    C_neg[-n_meas:, :] *= -1.0
    D_neg[-n_meas:, :] *= -1.0
    return sig.StateSpace(plant_ss.A, plant_ss.B, C_neg, D_neg)


def _ss2tf_robust(ss_dt):
    """Extract (num, den) from a discrete StateSpace, handling near-zero leading coefficients."""
    from scipy.signal import ss2tf as _ss2tf
    num_raw, den = _ss2tf(ss_dt.A, ss_dt.B, ss_dt.C, ss_dt.D)
    num = np.atleast_1d(num_raw[0]).ravel().copy()
    # Strip numerically zero leading coefficients
    thresh = 1e-10 * (np.abs(num).max() + 1e-300)
    while len(num) > 1 and abs(num[0]) < thresh:
        num = num[1:]
    return num, np.atleast_1d(den).ravel()


def _sdh2_ss(plant_ss: sig.StateSpace, T: float,
             n_meas: int = 1, n_ctrl: int = 1):
    """State-space lifting path for sdh2."""
    from directsd.design.lifting import lift_h2
    from directsd.sspace.design import h2reg

    plant_min = Minreal.ss(plant_ss)

    try:
        dsys, gamma, _ = lift_h2(plant_min, T, n_meas=n_meas, n_ctrl=n_ctrl)
    except Exception as exc:
        warnings.warn(f"lift_h2 failed ({exc}); falling back to ZOH discretisation")
        dsys = plant_min.to_discrete(T, method='zoh')
        gamma = 0.0

    try:
        K_ss, h2n = h2reg(dsys, n_meas=n_meas, n_ctrl=n_ctrl)
        err = float(np.sqrt(max(h2n ** 2 + gamma, 0.0)))
    except Exception as exc:
        warnings.warn(f"h2reg failed ({exc}); returning unit controller")
        K_ss = sig.StateSpace([[0.0]], [[1.0]], [[1.0]], [[0.0]], dt=T)
        err = float('nan')

    return _ss_to_tf_tuple(K_ss), err


def _sdh2_pol(plant_ss: sig.StateSpace, T: float,
              t, H, n_meas: int, n_ctrl: int, udelay: float = 0.0,
              refdelay: float = 0.0):
    """
    Polynomial path: compute sdh2coef then solve polquad.

    For SISO plants this implements the exact K. Polyakov algorithm.
    """
    try:
        # MATLAB sdh2.m line 78: sys(end,:) = -sys(end,:) — negative-feedback
        # transform, applied before D22/sdh2coef (without it the designed K
        # comes out exactly sign-flipped).
        plant_neg = _negate_meas_rows(plant_ss, n_meas)
        A_zpk, B_zpk, E_zpk, D22, _zt = _sdh2coef(plant_neg, T, t, H,
                                                  n_meas=n_meas, n_ctrl=n_ctrl,
                                                  return_zpk=True, udelay=udelay,
                                                  refdelay=refdelay)
        # _sdh2coef returns A/B/E in z-domain; polquad expects d-domain (ζ=1/z).
        # Apply z2zeta to match MATLAB sdh2coef.m which calls z2zeta internally
        # (root-list level: the ζ-conversion is conj_dt).
        _Az, _Bz, _Ez = (_zt[0].conj_dt(), _zt[1].conj_dt(), _zt[2].conj_dt())
        A_zpk, B_zpk, E_zpk = _Az.to_tf(), _Bz.to_tf(), _Ez.to_tf()
        # MATLAB sdh2.m line 92: PCancel = intpoles(sys) — shared internal
        # poles (e.g. plant integrators) must cancel analytically in L
        # (mapped CT → ζ as exp(-λT); see sdahinf).
        _p_int = _intpoles(plant_neg, n_meas, n_ctrl)
        _pc_zeta = np.exp(-_p_int * T) if len(_p_int) else None
        K_tf, err = _polquad(A_zpk, B_zpk, E_zpk, D22, P_cancel=_pc_zeta,
                             zpk_in=(_Az, _Bz, _Ez))
        # MATLAB sdh2.m last line: err = sqrt(err) — polquad returns the
        # SQUARED cost; sdl2.m deliberately does NOT sqrt (T·norm² units).
        if np.isfinite(err):
            err = float(np.sqrt(max(err, 0.0)))
        if udelay == 0 and refdelay == 0.0:
            err = _validate_pol_cost(err, plant_ss, K_tf, T, kind='h2',
                                     t=t, H=H, n_meas=n_meas, n_ctrl=n_ctrl)
        elif refdelay != 0.0:
            # _sderr's coef_fn=_sdh2coef call below doesn't thread refdelay
            # through — cross-checking against it would compare against the
            # WRONG (undelayed-Q) coefficients and could flag a genuinely
            # correct preview-control cost as inconsistent. No independent
            # evaluator exists yet for refdelay (sdh2norm has no delay
            # support at all, same as udelay); skip validation rather than
            # risk a false-positive warning.
            pass
        elif t is not None:
            # _sderr's t-specific evaluator has its own known, separate
            # ~2x discrepancy — substituting it here would corrupt
            # an already-correct polquad cost with a wrong "independent"
            # number (confirmed via an independent Octave re-derivation:
            # MATLAB's own polquad.m reproduces this exact `err` on the
            # exact same coefficients).
            # Same t-is-unverified carve-out `_validate_pol_cost` already
            # applies for the udelay=0 case.
            pass
        else:
            # sdh2norm has no delay support (see sdl2's identical
            # limitation) — validate via _sderr on the SAME delayed
            # coefficients instead (delay-exact by construction).
            try:
                indep = float(np.sqrt(max(
                    _sderr(plant_ss, K_tf, T, t=t, H=H, n_meas=n_meas,
                          n_ctrl=n_ctrl, coef_fn=_sdh2coef, udelay=udelay),
                    0.0)))
                if np.isfinite(indep) and (not np.isfinite(err) or err < 0
                        or abs(err - indep) > 0.05 * max(abs(indep), 1e-12)):
                    warnings.warn(
                        f"polynomial design cost {err:.6g} is inconsistent "
                        f"with the independent Z(ω) evaluation {indep:.6g} "
                        f"(intz2b truncation on smeared unit-circle roots); "
                        f"reporting the independent value")
                    err = indep
            except Exception:
                pass
        return K_tf, err
    except Exception as exc:
        if udelay > 0 or refdelay != 0.0:
            # The 'ss' fallback would silently ignore the delay and design
            # for a different plant — fail honestly instead (see sdl2).
            raise
        warnings.warn(f"Polynomial sdh2 failed ({exc}); falling back to 'ss'")
        return _sdh2_ss(plant_ss, T, n_meas, n_ctrl)


def _validate_pol_cost(err, plant_ss, K_tf, T, kind, t=None, H=None,
                       n_meas=1, n_ctrl=1, rtol=0.05):
    """
    Cross-check the polynomial path's REPORTED cost against the independent
    lifted evaluator (sdh2norm / sdl2err) and report that instead when they
    disagree.

    polquad's exact intz2b cost can be corrupted for plants whose A/B/E carry
    smeared high-multiplicity |z|=1 roots: the returned K itself is fine
    (gated separately by _polquad's closed-loop stability check) but the cost
    integral's B·B̃/A reduction leaves spurious near-unit-circle poles and the
    number comes out garbage (even negative). MATLAB reports design and
    evaluated costs that agree at the optimum, so substituting the independent
    evaluation preserves parity; the warning keeps the substitution visible.

    Only active for the plain case (t=None, H=None): sdh2norm's specific-t
    and generalized-hold evaluation paths are themselves unverified against
    MATLAB (t=0 measured at exactly 2x the average, H-path off by ~5x on the
    documented generalized-hold example), so substituting them would make
    those reported costs worse, not better.
    """
    if t is not None or H is not None:
        return err
    try:
        if kind == 'h2':
            from directsd.analysis.norms import sdh2norm
            indep = float(sdh2norm(plant_ss, K_tf, T, t=t, H=H))
        else:
            from directsd.analysis.errors import sdl2err
            indep = float(sdl2err(plant_ss, K_tf, T))
    except Exception:
        return err
    if not np.isfinite(indep):
        return err
    ref = max(abs(indep), 1e-12)
    if not np.isfinite(err) or err < 0 or abs(err - indep) > rtol * ref:
        warnings.warn(
            f"polynomial design cost {err:.6g} is inconsistent with the "
            f"independent closed-loop evaluation {indep:.6g} (intz2b "
            f"truncation on smeared unit-circle roots); reporting the "
            f"independent value")
        return indep
    return err


def _sderr(plant_ss: sig.StateSpace, K_tf, T: float, t=None, H=None,
           n_meas: int = 1, n_ctrl: int = 1, coef_fn=None, udelay: float = 0.0):
    """
    Squared H2/L2 error of a *given* controller K for a sampled-data system
    with a (possibly generalized) hold H.

    Port of sderr.m/quaderr.m (K. Polyakov): evaluates
    Z(omega) = A|M|^2 + E - 2*Re(B*M) on the unit circle, where
    M = K/(1+D22*K) is the closed loop around D22 (feedback formula), then
    integrates Z over the unit circle. Unlike quaderr.m (which factors X via
    sfactor and takes ||FX||^2), this integrates the frequency response
    directly -- avoiding the same root-finding fragility that spectral
    factorization shows for plants with high-multiplicity roots near the
    unit circle -- these are mathematically equivalent by Parseval's theorem.

    coef_fn selects the coefficient builder: '_sdh2coef' (needs t) or
    '_sdl2coef'.

    A/B/E/D22 are built from the SAME negative-feedback-transformed plant
    `_sdh2_pol`/`_sdl2_pol` use for design (`sys(end,:) = -sys(end,:)`,
    MATLAB sdh2.m:78/sdl2.m:68) -- K is only meaningful relative to that
    convention (it is what polquad/whquad actually optimized against), so
    evaluating a GIVEN K's cost against un-negated coefficients silently
    uses the wrong feedback sign whenever B (linear in the negated row) is
    nonzero -- confirmed via an independent Octave re-derivation: reproduced
    MATLAB's documented demo_fil2 t=0 costs (0.7577, 0.7735) only after negating;
    without it, both were off by a large, K-dependent, non-constant factor
    (not a simple missing scale) that had looked like a formula bug.

    Returns the square of the H2/L2 norm (caller takes sqrt).
    """
    if coef_fn is None:
        coef_fn = _sdh2coef
    plant_neg = _negate_meas_rows(plant_ss, n_meas)
    if coef_fn is _sdh2coef:
        A_zpk, B_zpk, E_zpk, D22 = _sdh2coef(plant_neg, T, t, H,
                                              n_meas=n_meas, n_ctrl=n_ctrl,
                                              udelay=udelay)
    else:
        A_zpk, B_zpk, E_zpk, D22 = _sdl2coef(plant_neg, T, H,
                                              n_meas=n_meas, n_ctrl=n_ctrl,
                                              udelay=udelay)
    A_num, A_den = _z2zeta(*A_zpk)
    B_num, B_den = _z2zeta(*B_zpk)
    E_num, E_den = _z2zeta(*E_zpk)
    D22_num, D22_den = D22
    K_num, K_den = _z2zeta(np.asarray(K_tf[0], float), np.asarray(K_tf[1], float))

    N_freq = 4096
    w = np.linspace(1e-6, np.pi - 1e-6, N_freq)
    z = np.exp(1j * w)

    def _ev(num, den):
        return np.polyval(num, z) / (np.polyval(den, z) + 1e-300)

    A_f = np.real(_ev(A_num, A_den))
    B_f = _ev(B_num, B_den)
    E_f = np.real(_ev(E_num, E_den))
    D22_f = _ev(D22_num, D22_den)
    K_f = _ev(K_num, K_den)

    M_f = K_f / (1.0 + D22_f * K_f + 1e-300)
    Z_f = A_f * np.abs(M_f) ** 2 + E_f - 2.0 * np.real(B_f * M_f)

    trap = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(abs(trap(Z_f, w) / np.pi))


# ---------------------------------------------------------------------------
# sdl2 – L2-optimal sampled-data controller
# ---------------------------------------------------------------------------

def sdl2(plant, T, H=None, method='pol', udelay=0.0, refdelay=0.0):
    """
    L2-optimal controller for a sampled-data system.

    Minimises the integral of the squared output error over one period
    (instead of the average H2 cost).

    Parameters
    ----------
    plant : scipy.signal.lti / StateSpace / (num, den)
    T : float
    H : scipy.signal.lti, optional
        Generalized hold (default: ZOH).
    method : str
        'pol' (polynomial, default — matches MATLAB sdl2.m) or 'ss'
        (state-space lifting). The 'pol' path falls back to 'ss' with a
        warning if the polynomial design fails its closed-loop stability gate.
    udelay : float, optional
        Continuous computational delay τ (0 ≤ τ ≤ T) on the control input —
        MATLAB's ``F.iodelay = τ`` on the P12/P22 column. Handled EXACTLY
        via the modified Z-transform (no Padé; a Padé substitute introduces
        fast poles whose discretized images span ~20 decades and destroy
        the polynomial pipeline). Pass the plant WITHOUT the delay
        baked in. Like MATLAB's sdl2.m, delays force the polynomial method
        ("state-space computations impossible").
    refdelay : float, optional
        Continuous preview horizon π — MATLAB's preview control
        (demo_l2p.m), a *different* delay topology from `sdh2`'s
        `refdelay`: enters via BOTH the ideal operator (inside P11 = Q·R)
        and the reference generator R alone (P21), following MATLAB's
        σ = ceil(π/T), θ = σ·T − π split. See `_sdl2coef`'s docstring for
        the full derivation. Any positive value (not restricted to
        [0, T]); forces the polynomial method for the same reason
        `udelay` does.

    Returns
    -------
    K : (num_d, den_d) tuple
    err : float
    """
    plant_ss, n_meas, n_ctrl = _parse_plant(plant)

    if method == 'ss' and H is not None:
        # See sdh2's identical check: the state-space/lifting path has no
        # generalized-hold support, matching MATLAB's sdl2.m behavior.
        warnings.warn("Method 'ss' is not applicable for generalized hold; using 'pol'")
        method = 'pol'
    if method == 'ss' and (udelay > 0 or refdelay != 0.0):
        # sdl2.m lines 38-45: 'System has delays, state-space computations
        # impossible' → force 'pol'.
        warnings.warn("System has delays, state-space computations "
                      "impossible; using 'pol'")
        method = 'pol'

    if method == 'ss':
        return _sdl2_ss(plant_ss, T, n_meas, n_ctrl)
    elif method == 'pol':
        try:
            # MATLAB sdl2.m line 68: sys(end,:) = -sys(end,:) — negative-
            # feedback transform before D22/sdl2coef (see _sdh2_pol).
            plant_neg = _negate_meas_rows(plant_ss, n_meas)
            A_zpk, B_zpk, E_zpk, D22, _zt = _sdl2coef(
                plant_neg, T, H, n_meas=n_meas, n_ctrl=n_ctrl,
                return_zpk=True, udelay=udelay, refdelay=refdelay)
            K_tf, err = _polquad(A_zpk, B_zpk, E_zpk, D22, zpk_in=_zt)
            # MATLAB sdl2.m returns polquad's cost as-is (T·norm² units,
            # no sqrt — unlike sdh2.m); validate it against the independent
            # lifted evaluator (see _validate_pol_cost). The lifted
            # evaluator has no delay support, so with a delay validate via
            # _sderr instead (direct Z(ω) integration of the SAME delayed
            # A/B/E coefficients — delay-exact by construction).
            if udelay == 0 and refdelay == 0.0:
                err = _validate_pol_cost(err, plant_ss, K_tf, T, kind='l2',
                                         n_meas=n_meas, n_ctrl=n_ctrl)
            else:
                # No delay-capable independent evaluator exists: the lifted
                # sdl2err can't model delays, and direct Z(ω) grid
                # integration cannot resolve the analytic cancellation of
                # the integrable z=1 singularities that makes the true cost
                # tiny (measured 1549 vs the true 6.7e-7 on the documented
                # L2-redesign example — worse than intz2b's raw number).
                # The returned K is exact (matches MATLAB's documented
                # controller digit-for-digit); treat the reported cost as
                # approximate and verify externally, e.g. sdl2err on a
                # Padé-approximated plant (accurate for EVALUATION — the
                # documented example scores 7.9e-7 that way).
                warnings.warn(
                    "sdl2 with udelay/refdelay: reported design cost may be "
                    "inaccurate (intz2b on delayed coefficients); verify "
                    "with sdl2err on a delay-approximated plant")
            return K_tf, err
        except Exception as exc:
            if udelay > 0 or refdelay != 0.0:
                # The 'ss' fallback would silently ignore the delay and
                # design for a different plant — fail honestly instead.
                raise
            warnings.warn(f"Polynomial sdl2 failed ({exc}); falling back to 'ss'")
            return _sdl2_ss(plant_ss, T, n_meas, n_ctrl)
    else:
        raise ValueError(f"Unknown method '{method}'; use 'ss' or 'pol'")


def _sdl2_ss(plant_ss: sig.StateSpace, T: float,
             n_meas: int = 1, n_ctrl: int = 1):
    """State-space lifting path for sdl2 (Chen-Francis L2-lifting)."""
    from directsd.design.lifting import lift_l2
    from directsd.sspace.design import h2reg

    plant_min = Minreal.ss(plant_ss)

    try:
        dsys = lift_l2(plant_min, T, n_meas=n_meas, n_ctrl=n_ctrl)
    except Exception as exc:
        warnings.warn(f"lift_l2 failed ({exc}); falling back to ZOH")
        dsys = plant_min.to_discrete(T, method='zoh')

    try:
        K_ss, h2n = h2reg(dsys, n_meas=n_meas, n_ctrl=n_ctrl)
        err = float(T * h2n ** 2)          # L2 cost = T * ‖·‖²_H2  (matches MATLAB sdl2.m)
    except Exception as exc:
        warnings.warn(f"h2reg failed ({exc}); returning unit controller")
        K_ss = sig.StateSpace([[0.0]], [[1.0]], [[1.0]], [[0.0]], dt=T)
        err = float('nan')

    return _ss_to_tf_tuple(K_ss), err


# ---------------------------------------------------------------------------
# sd2dof – 2-DOF feedforward controller
# ---------------------------------------------------------------------------

def sd2dof(plant, K_fb, T=None, udelay=0.0, refdelay=0.0):
    """
    Optimal feedforward controller for 2-DOF sampled-data systems.

    Port of sd2dof.m (K. Polyakov). Given a *fixed, already-designed* feedback
    controller K_fb (e.g. from sdl2/sdh2), finds the L2-optimal feedforward
    (reference) controller KR that minimises the tracking error — the
    standard "design feedback first, then feedforward" 2-DOF approach.

    This function previously accepted `K_fb` but never used
    it, instead performing a "joint" H2 optimisation over both controllers
    simultaneously — a different problem than MATLAB's `sd2dof.m` solves.
    Confirmed wrong against a documented MATLAB example (off by ~800x in
    cost). Rewritten to match MATLAB's actual algorithm, which genuinely
    needs K_fb as an input.

    Parameters
    ----------
    plant : scipy.signal.StateSpace or (num, den) tuple or scipy lti
        Full continuous-time 2-DOF generalised plant with structure::

            [z ]   [P11  P12 ] [d]
            [y1] = [P21   0  ] [u]
            [y2]   [P210 P22]

        Must have exactly 1 performance output, 2 measurement outputs, 1
        exogenous input, and 1 control input. P210 must equal a constant
        times P21 (MATLAB's own restriction). A SISO transfer function for
        P22 (the plant F) may also be passed for backward compatibility; the
        routine then builds a minimal 2-DOF plant with Q=1, R=1 internally.
    K_fb : (num, den) tuple or scipy dlti
        The already-designed, fixed feedback controller (negative-feedback
        convention, z-domain). If `K_fb` was itself designed with `udelay`/
        `refdelay` (e.g. via `sdl2(..., udelay=tau, refdelay=preview)`), pass
        the SAME values here — MATLAB's `demo_2dofp.m` uses one consistent
        delayed plant/preview horizon for both the feedback and feedforward
        designs.
    T : float
        Sampling period.
    udelay : float, optional
        Continuous computational delay τ on the control input (P12/P22
        column) — same meaning and mechanism as `sdl2`'s `udelay` (the plant
        passed in must be WITHOUT the delay baked in). Enters BOTH the
        `_sdl2coef` call on the `[z;y1]` sub-plant (P12's own delay) AND the
        `D22 = dtfm(P22,T,0,H)'` rescaling built directly from `P22` here
        (`sd2dofcoef.m`'s own `D22`, distinct from `sdl2coef`'s).
    refdelay : float, optional
        Continuous preview horizon π — MATLAB's preview control
        (`demo_2dofp.m`), identical mechanism to `sdl2`'s `refdelay` (the
        σ/θ split enters P11=Q·R and P21=R exactly as `_sdl2coef` already
        handles). `P22` never carries `refdelay` (only `Q`/`R` do), so it
        does not enter the D22 rescaling here.

    Returns
    -------
    K_ff : (num, den)
        Optimal feedforward (reference) controller KR.
    err : float
        Optimal 2-DOF L2 tracking cost.
    """
    # --- resolve sampling period ---
    if T is None:
        raise ValueError("sd2dof: sampling period T must be supplied")
    if K_fb is None:
        raise ValueError(
            "sd2dof: K_fb (the fixed feedback controller) must be supplied — "
            "sd2dof finds the optimal feedforward controller given a known "
            "feedback loop, matching MATLAB's sd2dof(sys, K)."
        )

    # --- resolve plant to (StateSpace, n_meas, n_ctrl) ---
    if isinstance(plant, GeneralizedPlant):
        plant_ss = plant.to_statespace()
    elif isinstance(plant, sig.StateSpace):
        plant_ss = plant
    elif isinstance(plant, (sig.lti, sig.TransferFunction, sig.ZerosPolesGain)):
        tmp_ss = plant.to_ss()
        if tmp_ss.C.shape[0] == 3 and tmp_ss.D.shape[1] == 2:
            plant_ss = tmp_ss
        elif tmp_ss.C.shape[0] == 1 and tmp_ss.D.shape[1] == 1:
            plant_ss = _build_2dof_siso_plant(tmp_ss)
        else:
            raise TypeError(
                "sd2dof: plant must be a 3-output 2-input StateSpace "
                "(rows: performance, y1, y2; cols: disturbance, control)"
            )
    elif isinstance(plant, tuple) and len(plant) == 2:
        plant_ss = _build_2dof_siso_plant(sig.TransferFunction(*plant).to_ss())
    else:
        raise TypeError(f"sd2dof: unsupported plant type {type(plant)}")

    # --- negate the y1 (feedback measurement) row — negative-feedback
    # convention, matching MATLAB's `sys(end-1,:) = -sys(end-1,:)` ---
    A_p, B_p, C_p, D_p = plant_ss.A, plant_ss.B, plant_ss.C, plant_ss.D
    o1 = C_p.shape[0] - 2
    C_neg = C_p.copy(); C_neg[o1:o1 + 1, :] *= -1.0
    D_neg = D_p.copy(); D_neg[o1:o1 + 1, :] *= -1.0
    plant_neg = sig.StateSpace(A_p, B_p, C_neg, D_neg)

    K_num, K_den = _ensure_tf(K_fb)
    K_num = -np.asarray(K_num, float).ravel()   # K = -K

    A_tf, B_tf, E_tf, kCoef, n, d = _sd2dofcoef(plant_neg, T, udelay=udelay,
                                                refdelay=refdelay)

    D22_tf = (np.array([0.0]), np.array([1.0]))
    PsiX_z, err = _polquad(A_tf, B_tf, E_tf, D22_tf)

    # PsiX = PsiX' (ζ-domain). Padded conjugate (_z2zeta), not naive
    # [::-1]: H(1/z) gains origin zeros/poles for any relative-degree
    # difference (see _sd2dofcoef's D22 comment / @zpk/ctranspose.m).
    psix_num, psix_den = _z2zeta(PsiX_z[0], PsiX_z[1])

    # [a,b] = tf2nd(K') — K' = K(1/z), same padded-conjugate convention
    a_coef, b_coef = _z2zeta(K_num, K_den)

    # Delta = a*n + b*d
    Delta = _tfpadd(np.polymul(a_coef, n), np.polymul(b_coef, d))

    # KR = PsiX*(Delta/b) - kCoef*K, as an explicit rational combination:
    #   term1 = (psix_num*Delta) / (psix_den*b_coef)
    #   term2 = (kCoef*K_num) / K_den
    #   KR = term1 - term2, common denominator psix_den*b_coef*K_den
    K_den_arr = np.asarray(K_den, float).ravel()
    t1_num = np.polymul(psix_num, Delta)
    t1_den = np.polymul(psix_den, b_coef)
    t2_num = kCoef * K_num
    t2_den = K_den_arr
    kr_num_raw, kr_den_raw = _tfsub(t1_num, t1_den, t2_num, t2_den)
    try:
        kr_num_raw, kr_den_raw = Minreal.tf(kr_num_raw, kr_den_raw, tol=1e-3)
    except Exception:
        pass

    # KR = KR' — final conjugate back to z-domain (padded, see above)
    KR_num, KR_den = _z2zeta(kr_num_raw, kr_den_raw)
    lead = float(KR_den[0]) if len(KR_den) > 0 and abs(KR_den[0]) > 1e-30 else 1.0
    KR_num = KR_num / lead
    KR_den = KR_den / lead

    return (striplz(np.real(KR_num)), striplz(np.real(KR_den))), err


# ---------------------------------------------------------------------------
# split2dof – stable realization of 2-DOF controllers
# ---------------------------------------------------------------------------

def _match_poles(pA, pB, tol=1e-5):
    """
    Match poles in pA against poles in pB (greedy, closest-first).

    Returns (p_matched, p_unmatched) where both are subsets of pA.
    """
    pA = np.asarray(pA, dtype=complex).ravel().tolist()
    pB = np.asarray(pB, dtype=complex).ravel().tolist()

    matched_idx   = []
    unmatched_B   = list(range(len(pB)))

    for i, p in enumerate(pA):
        if not unmatched_B:
            break
        dists = [abs(p - pB[j]) for j in unmatched_B]
        k = int(np.argmin(dists))
        if dists[k] < tol:
            matched_idx.append(i)
            unmatched_B.pop(k)

    unmatched_idx = [i for i in range(len(pA)) if i not in matched_idx]
    p_matched   = np.array([pA[i] for i in matched_idx],   dtype=complex)
    p_unmatched = np.array([pA[i] for i in unmatched_idx], dtype=complex)
    return p_matched, p_unmatched


def _roots_to_poly(roots):
    """Monic real-coefficient polynomial from a set of roots."""
    if len(roots) == 0:
        return np.array([1.0])
    return np.real(np.poly(roots))


def split2dof(K, KR, tol=1e-5):
    """
    Stable realization of a 2-DOF sampled-data controller.

    Splits the full feedback controller K(z) and full reference controller
    KR(z) into:
        KF(z)     – feedback-only part
        KR_new(z) – reference-only part
        KC(z)     – common part (KC = z^n / dCommon)

    such that K = KF * KC and KR = KR_new * KC.

    Port of split2dof.m (K. Polyakov).

    Parameters
    ----------
    K  : (num, den) tuple
        Full discrete-time feedback controller.
    KR : (num, den) tuple
        Full discrete-time reference controller.
    tol : float
        Tolerance for pole matching.

    Returns
    -------
    KF     : (num, den) tuple  — feedback part
    KR_new : (num, den) tuple  — reference part
    KC     : (num, den) tuple  — common part

    Raises
    ------
    ValueError
        If the reference controller would be unstable after splitting.
    """
    def _r(c):
        return np.real(np.asarray(c, float)).ravel()

    K_num,  K_den  = [_r(c) for c in _ensure_tf(K)]
    KR_num, KR_den = [_r(c) for c in _ensure_tf(KR)]

    pK  = np.roots(K_den)  if len(K_den)  > 1 else np.array([], dtype=complex)
    pKR = np.roots(KR_den) if len(KR_den) > 1 else np.array([], dtype=complex)

    # Common poles: match pKR against pK; pCommon has values from pKR
    pCommon, pKRa = _match_poles(pKR, pK, tol)
    n = len(pCommon)

    if n == 0:
        return ((K_num.copy(),  K_den.copy()),
                (KR_num.copy(), KR_den.copy()),
                (np.array([1.0]), np.array([1.0])))

    # pKa = remaining poles of K (not matched by common set)
    _, pKa = _match_poles(pK, pCommon, tol)

    # KC(z) = z^n / dCommon(z)
    KC_num = np.zeros(n + 1, dtype=float); KC_num[0] = 1.0
    KC_den = _roots_to_poly(pCommon)

    # KF denominator: original K_den scale * poly(pKa) * z^n
    K_lead = float(K_den[0])
    KF_den = K_lead * np.polymul(_roots_to_poly(pKa), KC_num)
    KF_num = K_num.copy()

    # KR_new denominator: original KR_den scale * poly(pKRa) * z^n
    KR_lead    = float(KR_den[0])
    KR_new_den = KR_lead * np.polymul(_roots_to_poly(pKRa), KC_num)
    KR_new_num = KR_num.copy()

    if len(pKRa) > 0 and np.any(np.abs(pKRa) >= 1.0):
        raise ValueError("Reference controller cannot be unstable after splitting")

    return ((striplz(KF_num),     striplz(KF_den)),
            (striplz(KR_new_num), striplz(KR_new_den)),
            (striplz(KC_num),     striplz(KC_den)))


# ---------------------------------------------------------------------------
# dhinf – discrete-time polynomial H∞ controller design
# ---------------------------------------------------------------------------

def _z2zeta(a, b=None):
    """z -> zeta=1/z substitution. ONE function for any input shape, matching
    how MATLAB's own @tf/z2zeta.m already loops over `[rows,cols]=size(F)`
    internally (a scalar TF is just the `[1,1]` case of the same loop) --
    the "single pair vs whole matrix" split here was a Python-side artifact
    of not having a matrix-valued TF class, not a genuine algorithm split.

    Two calling forms:
      _z2zeta(num, den) -- a single (num, den) coefficient-array pair (the
        original call convention every caller in this module already uses):
        reverse both arrays (pad to equal length first, matching MATLAB's
        striplz([N;D]) convention), returns (num, den).
      _z2zeta(sys) -- ONE argument, dispatching on its type:
        - a `Zpk` root-list object: delegates to `Zpk.conj_dt()` (MATLAB's
          @zpk/z2zeta.m is a root-level algorithm -- separate origin
          zeros/poles, reciprocate the rest, rescale the gain, pad back --
          which `conj_dt` already implements exactly, modulo an extra
          complex-conjugation step that's a no-op for the real-coefficient
          rationals used throughout this codebase).
        - a list of lists of (num, den) tuples (a plant MATRIX -- MATLAB's
          `sys = z2zeta(sys)` before calling `dhinf`, since `dhinf` applies
          z2zeta again internally): applies the same per-pair algorithm to
          every entry.
    """
    if b is not None:
        n = np.atleast_1d(np.asarray(a, float)).ravel()
        d = np.atleast_1d(np.asarray(b, float)).ravel()
        # Pad shorter array with leading zeros (matching MATLAB striplz([N;D]) convention)
        ln, ld = len(n), len(d)
        if ln < ld:
            n = np.concatenate([np.zeros(ld - ln), n])
        elif ld < ln:
            d = np.concatenate([np.zeros(ln - ld), d])
        # Strip leading zeros then flip
        def _slz(arr):
            idx = np.nonzero(arr)[0]
            return arr[idx[0]:] if len(idx) else np.array([0.0])
        return _slz(n[::-1]), _slz(d[::-1])

    if isinstance(a, Zpk):
        return a.conj_dt()

    if isinstance(a, list) and a and isinstance(a[0], list):
        return [[_z2zeta(np.atleast_1d(np.asarray(e[0], float)),
                         np.atleast_1d(np.asarray(e[1], float)))
                 for e in row] for row in a]

    raise TypeError(
        f"_z2zeta: expected _z2zeta(num, den), a Zpk, or a list of lists of "
        f"(num, den) tuples; got a single {type(a)}")


def _cancel_origin_tf(num, den):
    """Cancel exactly-common origin ζ-factors (shared trailing zero coefficients)."""
    num = np.atleast_1d(np.asarray(num, float)).ravel()
    den = np.atleast_1d(np.asarray(den, float)).ravel()

    def _n_trailing(a):
        c = 0
        for v in a[::-1]:
            if v == 0.0:
                c += 1
            else:
                break
        return min(c, len(a) - 1)

    m = min(_n_trailing(num), _n_trailing(den))
    if m:
        num = num[:-m]
        den = den[:-m]
    return num, den


def _eval_tf_freq(num, den, z_f):
    """Evaluate TF (num, den) at complex frequencies z_f."""
    n = np.atleast_1d(np.asarray(num, float)).ravel()
    d = np.atleast_1d(np.asarray(den, float)).ravel()
    return np.polyval(n, z_f) / (np.polyval(d, z_f) + 1e-300)


def _h2coef_freq(sys_rows, N_freq=1024):
    """
    Compute h2coef (A, B, E) for a MIMO generalized plant in ζ-domain.

    Exact polynomial computation via TF arithmetic (no frequency fitting).

    sys_rows : list of lists of (num, den) tuples, shape (nout, nin).
    The last output row is the measurement output y; the last input column
    is the control input u. Negative-feedback convention: last row has been
    negated already.

    Returns A, B, E as simplified (num, den) polynomial tuples.
    """
    nout = len(sys_rows)
    nin  = len(sys_rows[0])
    o2, i2 = 1, 1
    o1 = nout - o2   # performance output count
    i1 = nin  - i2   # exogenous input count

    def _n(entry):
        return np.atleast_1d(np.asarray(entry[0], float)).ravel()

    def _d(entry):
        return np.atleast_1d(np.asarray(entry[1], float)).ravel()

    def _tf_add(n1, d1, n2, d2):
        """(n1/d1) + (n2/d2) — exact arithmetic."""
        return (np.real(np.polyadd(np.polymul(n1, d2), np.polymul(n2, d1))).astype(float),
                np.real(np.polymul(d1, d2)).astype(float))

    def _tf_mul(n1, d1, n2, d2):
        return (np.real(np.polymul(n1, n2)).astype(float),
                np.real(np.polymul(d1, d2)).astype(float))

    def _conj_rec(num, den):
        """H~(z) = H(1/z): pad to same length first (like MATLAB z2zeta), then reverse."""
        n = np.asarray(num, float).ravel()
        d = np.asarray(den, float).ravel()
        ln, ld = len(n), len(d)
        if ln < ld:
            n = np.concatenate([np.zeros(ld - ln), n])
        elif ld < ln:
            d = np.concatenate([np.zeros(ln - ld), d])
        return n[::-1], d[::-1]

    def _simplify(num, den):
        """Divide num and den by their polynomial GCD (via repeated polydiv)."""
        num = np.real(np.asarray(num, float)).ravel()
        den = np.real(np.asarray(den, float)).ravel()
        # Try trial division by (z - root) for each root of den
        try:
            roots_d = np.roots(den)
        except Exception:
            return num, den
        for r in roots_d:
            factor = np.array([1.0, -float(np.real(r))])
            q_n, r_n = np.polydiv(num, factor)
            q_d, r_d = np.polydiv(den, factor)
            if np.linalg.norm(r_n) < 1e-6 * np.linalg.norm(num) and \
               np.linalg.norm(r_d) < 1e-6 * np.linalg.norm(den):
                num = np.real(q_n).astype(float)
                den = np.real(q_d).astype(float)
        return num, den

    # ------------------------------------------------------------------
    # A = (sum_j P12_j * P12_j~) * (sum_i P21_i * P21_i~)
    # ------------------------------------------------------------------
    # P12 column: rows 0..o1-1, column i1
    A_12_n, A_12_d = np.array([0.0]), np.array([1.0])
    for j in range(o1):
        P12_n = _n(sys_rows[j][i1])
        P12_d = _d(sys_rows[j][i1])
        P12cr_n, P12cr_d = _conj_rec(P12_n, P12_d)
        t_n, t_d = _tf_mul(P12_n, P12_d, P12cr_n, P12cr_d)
        A_12_n, A_12_d = _tf_add(A_12_n, A_12_d, t_n, t_d)

    # P21 row: row o1, columns 0..i1-1
    A_21_n, A_21_d = np.array([0.0]), np.array([1.0])
    for i in range(i1):
        P21_n = _n(sys_rows[o1][i])
        P21_d = _d(sys_rows[o1][i])
        P21cr_n, P21cr_d = _conj_rec(P21_n, P21_d)
        t_n, t_d = _tf_mul(P21_n, P21_d, P21cr_n, P21cr_d)
        A_21_n, A_21_d = _tf_add(A_21_n, A_21_d, t_n, t_d)

    A_n, A_d = _tf_mul(A_12_n, A_12_d, A_21_n, A_21_d)
    A_n, A_d = _simplify(A_n, A_d)

    # ------------------------------------------------------------------
    # B = sum_i P21_i * (sum_j P11_ji~ * P12_j)
    # ------------------------------------------------------------------
    B_n, B_d = np.array([0.0]), np.array([1.0])
    for i in range(i1):
        P21_n = _n(sys_rows[o1][i])
        P21_d = _d(sys_rows[o1][i])
        Bi_n, Bi_d = np.array([0.0]), np.array([1.0])
        for j in range(o1):
            P11_n = _n(sys_rows[j][i])
            P11_d = _d(sys_rows[j][i])
            P12_n = _n(sys_rows[j][i1])
            P12_d = _d(sys_rows[j][i1])
            P11cr_n, P11cr_d = _conj_rec(P11_n, P11_d)
            t_n, t_d = _tf_mul(P11cr_n, P11cr_d, P12_n, P12_d)
            Bi_n, Bi_d = _tf_add(Bi_n, Bi_d, t_n, t_d)
        t_n, t_d = _tf_mul(P21_n, P21_d, Bi_n, Bi_d)
        B_n, B_d = _tf_add(B_n, B_d, t_n, t_d)
    B_n, B_d = _simplify(B_n, B_d)

    # ------------------------------------------------------------------
    # E = sum_ij |P11_ij|^2  (trace of P11~*P11)
    # ------------------------------------------------------------------
    E_n, E_d = np.array([0.0]), np.array([1.0])
    for j in range(o1):
        for i in range(i1):
            P11_n = _n(sys_rows[j][i])
            P11_d = _d(sys_rows[j][i])
            P11cr_n, P11cr_d = _conj_rec(P11_n, P11_d)
            t_n, t_d = _tf_mul(P11_n, P11_d, P11cr_n, P11cr_d)
            E_n, E_d = _tf_add(E_n, E_d, t_n, t_d)
    E_n, E_d = _simplify(E_n, E_d)

    return (A_n, A_d), (B_n, B_d), (E_n, E_d)


def dhinf(sys, T=None):
    """
    Polynomial H∞-optimal controller for a discrete-time system.

    Port of MATLAB dhinf.m (K. Polyakov).

    Parameters
    ----------
    sys : list-of-lists of (num, den) tuples
        Full discrete-time generalised plant in **z-domain** with structure::

            nout × nin  matrix of SISO TF blocks.
            Last output row = measurement y; last input column = control u.
            Example 3×2 plant: [[P11, P12], [P_noise, V2], [P21, D22]].

        The function applies z2zeta (z→ζ) internally and imposes negative-
        feedback convention by negating the last row.
    T : float, optional
        Sampling period (unused in polynomial design; kept for API symmetry).

    Returns
    -------
    K : (num, den) tuple, or a list of two such tuples
        Optimal controller in z-domain. In the non-generic case (MATLAB
        dhinf.m/polhinf.m: nonGen = iscell(P), e.g. dsd_help.md's
        documented dhinf "Example 3"), two equally-optimal controllers
        exist; K is then `[K0, K1]` instead of a single (num,den) tuple —
        mirrors MATLAB's own `K = {K1, K2}` cell return.
    lam : float
        Optimal H∞ cost (AH∞ norm).
    """
    # --- Parse the generalised plant ---
    if not isinstance(sys, list):
        raise TypeError(
            "dhinf expects a list-of-lists of (num, den) tuples representing "
            "the full MIMO generalised plant."
        )

    nout = len(sys)
    nin  = len(sys[0])

    # --- Apply z2zeta to every entry ---
    sys_zeta = []
    for row in sys:
        new_row = []
        for entry in row:
            if np.isscalar(entry) or (isinstance(entry, (int, float))):
                num_e, den_e = np.array([float(entry)]), np.array([1.0])
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                num_e = np.atleast_1d(np.asarray(entry[0], float)).ravel()
                den_e = np.atleast_1d(np.asarray(entry[1], float)).ravel()
            else:
                raise TypeError(f"dhinf: unsupported plant entry {type(entry)}")
            num_z, den_z = _z2zeta(num_e, den_e)
            new_row.append((num_z, den_z))
        sys_zeta.append(new_row)

    # --- Negate last row (negative-feedback convention) ---
    last = sys_zeta[-1]
    sys_zeta[-1] = [(-num_e, den_e) for (num_e, den_e) in last]

    # --- Extract D22 ---
    D22_num, D22_den = sys_zeta[-1][-1]
    D22_tf = (_strip_lz(np.real(D22_num).astype(float)),
              _strip_lz(np.real(D22_den).astype(float)))

    # --- Compute h2coef at the ROOT-LIST (zpk) level — port of h2coef.m.
    # The plant entries are low-degree rationals whose roots are essentially
    # exact; carrying (zeros, poles, gain) through the products/sums (with
    # pole-aware sumzpk addition and exact-match minreal cancellation, like
    # MATLAB's zpk arithmetic) is what allows zterm's downstream
    # Z = E - B*B~/A reduction to cancel completely. The previous
    # coefficient-array construction (_h2coef_freq) cross-multiplied into
    # high-degree arrays whose np.roots-smeared roots defeated every
    # cancellation — proven root cause via an independent Octave
    # re-derivation of the original MATLAB algorithm (Reports/octave_harness/). ---
    o1 = nout - 1
    i1 = nin - 1
    G = [[Zpk.from_tf(*sys_zeta[r][c]) for c in range(nin)]
         for r in range(nout)]
    P11 = [[G[r][c] for c in range(i1)] for r in range(o1)]
    P12 = [G[r][i1] for r in range(o1)]
    P21 = [G[o1][c] for c in range(i1)]
    ZERO = Zpk([], [], 0.0)

    # A0 = minreal(P12'*P12); A1 = minreal(P21*P21'); A = minreal(A0*A1)
    A0 = ZERO
    for j in range(o1):
        A0 = A0.zsum((P12[j].conj_dt() * P12[j]).minreal())
    A0 = A0.minreal()
    A1 = ZERO
    for i in range(i1):
        A1 = A1.zsum((P21[i] * P21[i].conj_dt()).minreal())
    A1 = A1.minreal()
    A_z = (A0 * A1).minreal()

    # B = Σ_i P21_i · Σ_j P11H(i,j)·P12_j   (h2coef.m lines 44-52)
    B_z = ZERO
    for i in range(i1):
        Bi = ZERO
        for j in range(o1):
            Bi = Bi.zsum((P11[j][i].conj_dt() * P12[j]).minreal())
        B_z = B_z.zsum((P21[i] * Bi).minreal())

    # E = trace(minreal(P11H*P11)) = Σ_{i,j} P11(j,i)~·P11(j,i)
    E_z = ZERO
    for i in range(i1):
        for j in range(o1):
            E_z = E_z.zsum((P11[j][i].conj_dt() * P11[j][i]).minreal())
    E_z = E_z.minreal()

    A_tf, B_tf, E_tf = A_z.to_tf(), B_z.to_tf(), E_z.to_tf()

    # --- Solve polynomial H∞ problem ---
    # _polhinf returns K already in z-domain (applies K'=z2zeta internally, MATLAB line 190).
    K_tf, lam, _ = _polhinf(A_tf, B_tf, E_tf, D22_tf,
                            zpk_in=(A_z, B_z, E_z))

    def _clean(kt):
        return striplz(np.real(kt[0])), striplz(np.real(kt[1]))

    # Non-generic case (MATLAB: dhinf returns a 1x2 cell {K1,K2} of
    # equally-optimal controllers — dsd_help.md's documented Example 3).
    # _polhinf signals this by returning K_tf as a list of two (num,den)
    # tuples instead of a single one; mirror that here as a list so callers
    # can tell the two cases apart the same way MATLAB's iscell(K) does.
    if isinstance(K_tf, list):
        return [_clean(kt) for kt in K_tf], float(lam)
    return _clean(K_tf), float(lam)


# ---------------------------------------------------------------------------
# polquad – polynomial minimisation of quadratic functional
# ---------------------------------------------------------------------------

def polquad(A, B, E, D22, P_cancel=None):
    """
    Polynomial minimisation of the H2 quadratic functional.

    Solves the Diophantine equations arising from sampled-data H2 design.

    Parameters
    ----------
    A, B, E : (num, den) tuples or Poln
        Quadratic cost coefficients as rational functions in the z (or ζ) domain.
    D22 : (num, den) tuple
        Discrete plant model (z-domain).
    P_cancel : array-like, optional
        Continuous-time poles that must cancel in the controller.

    Returns
    -------
    K : (num_d, den_d) tuple
    err : float
    """
    A_tf  = _ensure_tf(A)
    B_tf  = _ensure_tf(B)
    E_tf  = _ensure_tf(E)
    D22_tf = _ensure_tf(D22)
    return _polquad(A_tf, B_tf, E_tf, D22_tf, P_cancel)


def _intz2b(*tfs):
    """
    Exact contour integral V = (1/2πj)·∮ (X1 + X2 + …) dz/z over the unit
    circle — port of MATLAB private/intz2b.m (K. Polyakov).

    Each Xi is a symmetric (para-Hermitian) discrete-time rational as a
    (num, den) tuple. Algorithm (no frequency gridding anywhere — this is
    what makes the reported polquad cost exact for plants whose integrand
    explodes on a grid near smeared |z|=1 roots):

      1. z = (s+1)/(s-1) (bilintr with b=1) maps the unit circle onto the
         imaginary axis — and, crucially, poles at z=1 (integrators) to
         s=∞, where step 2 splits them off EXACTLY as a polynomial part.
      2. improper(Sb,'symm'): Sb = P0(s) + S0(s); only P0's constant
         contributes finitely to the integral (odd powers integrate to
         zero, higher even powers diverge) — it is moved into S0, the rest
         accumulates into the returned truncation error.
      3. dz/z = -2/((s-1)(s+1))·ds =: BL(s), the bilinear weight (positive
         on the axis, integrates to exactly 1).
      4. a = 2·||S0||_inf makes S0+a positive; F = sfactor((S0+a)·BL) and
         V += ||F||²_H2 − a, with the CT H2 norm computed exactly via a
         Lyapunov equation.

    Returns
    -------
    V : float
    err : float
        Norm of the residual polynomial part (nonzero when genuine poles
        at z=1 make the integral divergent — MATLAB reports the same).
    """
    from directsd.polynomial.utils import bilintr as _bilintr, improper as _improper
    from directsd.polynomial.spectral import _sfactor_lti_scipy as _sfact_rat

    b = 1.0
    V = 0.0
    P_acc = np.array([0.0])

    for tf in tfs:
        num = np.real(np.atleast_1d(np.asarray(tf[0])).ravel()).astype(float)
        den = np.real(np.atleast_1d(np.asarray(tf[1])).ravel()).astype(float)
        num = _strip_lz(num)
        den = _strip_lz(den)
        if len(num) == 0 or np.max(np.abs(num)) < 1e-300:
            continue

        # 1. Sb(s) = S(z)|_{z=(s+b)/(s-b)}
        Sb_n, Sb_d = _bilintr((num, den), [b])

        # 2. Sb = P0 + S0 ('symm' split; MATLAB improper.m additionally
        # zeroes the remainder's odd-power coefficients — the density is an
        # even function of s, so those are pure numerical noise).
        P0, (S0_n, S0_d) = _improper((Sb_n, Sb_d), 'symm')
        S0_n = np.atleast_1d(np.asarray(S0_n, float)).ravel().copy()
        S0_d = np.atleast_1d(np.asarray(S0_d, float)).ravel()
        if len(S0_n) >= 2:
            S0_n[len(S0_n) - 2::-2] = 0.0   # zero s^1, s^3, … (descending order)
            S0_n = _strip_lz(S0_n)
            if len(S0_n) == 0:
                S0_n = np.array([0.0])
        P0 = np.atleast_1d(np.asarray(P0, float)).ravel()

        # Correct improper part: move its (finite) constant contribution
        # into S0, accumulate the divergent remainder as truncation error.
        if np.linalg.norm(P0) > np.sqrt(np.finfo(float).eps):
            val1 = float(np.polyval(P0, 1.0))    # matches intz2b.m verbatim
            P0 = P0.copy()
            P0[-1] -= val1
            S0_n = np.polyadd(S0_n, val1 * S0_d)
            _lp = max(len(P_acc), len(P0))
            P_acc = (np.concatenate([np.zeros(_lp - len(P_acc)), P_acc])
                     + np.concatenate([np.zeros(_lp - len(P0)), P0]))

        # 3.-4. a = 2·||S0||_inf (axis grid max, ×2 margin as in MATLAB);
        # G = (S0+a)·BL with BL = -2b/((s-b)(s+b)).
        _w_ax = np.concatenate([[0.0], np.logspace(-6, 8, 4096)])
        _s_ax = 1j * _w_ax
        _S0_v = np.polyval(S0_n, _s_ax) / (np.polyval(S0_d, _s_ax) + 1e-300)
        a = 2.0 * float(np.max(np.abs(_S0_v)))

        G_n = -2.0 * b * np.polyadd(S0_n, a * S0_d)
        G_d = np.polymul(S0_d, np.array([1.0, 0.0, -b * b]))

        _zs = np.roots(G_n) if len(G_n) > 1 else np.array([], dtype=complex)
        _ps = np.roots(G_d) if len(G_d) > 1 else np.array([], dtype=complex)
        _kg = float(G_n[0]) / float(G_d[0])
        _, _fs0 = _sfact_rat(sig.ZerosPolesGain(_zs, _ps, _kg), 's')

        # ||F||²_H2 exactly, via the controllability Gramian:
        # A·P + P·Aᵀ = -B·Bᵀ,  ||F||² = C·P·Cᵀ  (F strictly proper: the
        # (S0+a)·BL product is strictly proper by ≥2, so its factor is by ≥1).
        F_num, F_den = sig.zpk2tf(_fs0.zeros, _fs0.poles,
                                  float(np.real(_fs0.gain)))
        A_F, B_F, C_F, D_F = sig.tf2ss(F_num, F_den)
        if abs(float(np.atleast_2d(D_F)[0, 0])) > 1e-8 * (abs(_kg) + 1.0):
            raise ValueError("intz2b: spectral factor is not strictly proper")
        P_gram = la.solve_lyapunov(A_F, -B_F @ B_F.T)
        h2sq = float((C_F @ P_gram @ C_F.T).ravel()[0])

        V += h2sq - a

    return V, float(np.linalg.norm(P_acc))


def _polquad(A_tf, B_tf, E_tf, D22_tf, P_cancel=None, zpk_in=None):
    """
    Internal polquad — port of MATLAB polquad.m (K. Polyakov).

    Uses _ynyd + diophsys: spectral factorisation of d²d̃²A gives Lam,
    then two simultaneous Diophantine equations give the optimal K.

    Cost (polquad.m lines 118-133): err = errLm + err0, where errLm =
    ||Pi/dLm||_H2^2 is the unavoidable cost from the shared Diophantine
    solution Pi, and err0 is the base integral of Z = E - B*B~/A.
    """
    from directsd.polynomial.operations import coprime as _cprime, triple as _triple
    from directsd.polynomial.diophantine import (
        dioph as _dioph, diophsys as _dioph_sys,
    )

    def _r(x):
        return np.real(np.asarray(x, float)).ravel()

    A_num, A_den = _r(A_tf[0]), _r(A_tf[1])
    B_num, B_den = _r(B_tf[0]), _r(B_tf[1])
    E_num, E_den = _r(E_tf[0]), _r(E_tf[1])

    # ── Open-loop branch (polquad.m lines 48-64): D22 == 0 exactly (e.g.
    # sd2dof's feedforward-only sub-problem). Self-contained — does NOT use
    # _ynyd, because here Lam = sfactor(Av) with Av = A itself (d=1), and A
    # is a genuine RATIONAL function: Lam = Lam_n/Lam_d with
    # Lam_d = sfactor(A_den). _ynyd represents Lam as a bare polynomial
    # (valid only in the closed-loop case where A_den divides d²d̃²·A_num
    # exactly) and silently drops A_den otherwise — which loses the
    # essential Lam_d factor that must cancel against dLp in K's
    # denominator (hand-derived for a documented open-loop example:
    # dLp·Lam = (ζ-1)·[Lam_n/(ζ-1)] = Lam_n, no z=1 pole in K).
    #
    # MATLAB equivalents for D22=0: n=0, d=1 → a0=0 → L2=0, so
    # L = L1 = B'/Lam'; then [dLp,dLm,dL0]=factor(dL), dioph, and
    # K = gN·N/(dLp·Lam) (polquad.m line 64).
    D22_num_check = _r(D22_tf[0])
    if np.max(np.abs(D22_num_check)) < 1e-12 * max(np.max(np.abs(_r(D22_tf[1]))), 1e-30):
        from directsd.polynomial.operations import (
            coprime as _cprime_ol, factor as _factor_ol,
        )
        from directsd.polynomial.diophantine import dioph as _dioph_ol
        from directsd.polynomial.spectral import sfactor as _sfactor_ol

        try:
            # Reduce A (cancels e.g. matching origin ζ-factors from the
            # d·d' rescale in _sd2dofcoef).
            An, Ad = Minreal.tf(A_num, A_den, tol=1e-6)

            # Rational spectral factor: Lam = Lam_n/Lam_d, Lam·Lam' = A.
            # Must be computed JOINTLY on the rational function (root-level
            # sfactor.m), not per-polynomial: the gain check
            # K = k·prod(-ps)/prod(-zs) uses the stable *halves* of both
            # root sets together, and an odd pole-half count flips the sign
            # (e.g. ps={1} here gives prod(-ps)=-1, turning A's negative
            # leading coefficient into a positive K). Per-polynomial
            # sfactor(An) alone sees the negative gain and (rightly, for a
            # bare quasipolynomial) raises.
            _zA, _pA, _kA = sig.tf2zpk(_r(_strip_lz(An)), _r(_strip_lz(Ad)))
            _fs, _fs0 = _sfactor_ol(sig.ZerosPolesGain(_zA, _pA, _kA), 'd')
            Lam_n = _r(np.poly(_fs0.zeros)) * float(np.real(_fs0.gain))
            Lam_d = _r(np.poly(_fs0.poles))

            # L = B'/Lam' (rational, ζ-domain; padded conjugates).
            Bc_n, Bc_d = _z2zeta(B_num, B_den)
            Lc_n, Lc_d = _z2zeta(Lam_n, Lam_d)
            L_n = np.polymul(Bc_n, Lc_d)
            L_d = np.polymul(Bc_d, Lc_n)
            L_n, L_d = Minreal.tf(L_n, L_d, tol=1e-6)

            # dL → dLp (stable+neutral, with gain) · dLm (antistable).
            dLp_p, dLm_p, dL0_p = _factor_ol(Poln(_r(_strip_lz(L_d)), 'z'), 'd')
            dLp_ol = _r(_strip_lz((dLp_p * dL0_p).coef))
            dLm_ol = _r(_strip_lz(dLm_p.coef))
            nL_ol = _r(_strip_lz(L_n))

            # Coprime reductions + single Diophantine (polquad.m 54-63).
            AN_pol = Poln(dLp_ol, 'z'); BN_pol = Poln(dLm_ol, 'z')
            CN_pol = Poln(nL_ol, 'z')
            AN_r, BN_r, gABN = _cprime_ol(AN_pol, BN_pol)
            CN_q, CN_rem = CN_pol.quorem(gABN)
            CN_r = CN_q if CN_rem.norm() < 1e-3 * max(CN_pol.norm(), 1e-30) else CN_pol
            AN_r2, CN_r2, gN = _cprime_ol(AN_r, CN_r)
            Pi_pol, N_pol, _, _ = _dioph_ol(AN_r2, BN_r, CN_r2)

            # K(ζ) = gN·N/(dLp·Lam) = gN·N·Lam_d/(dLp·Lam_n); the Lam_d
            # factor cancels dLp's neutral roots (the whole point of the
            # rational Lam handling).
            K_num_d0 = np.polymul(_r(_strip_lz((gN * N_pol).coef)), Lam_d)
            K_den_d0 = np.polymul(dLp_ol, Lam_n)
            K_num_d0, K_den_d0 = Minreal.tf(K_num_d0, K_den_d0, tol=1e-3)

            if len(K_den_d0) == 0 or np.all(np.abs(K_den_d0) < 1e-30):
                raise ValueError("zero denominator")
            # ζ → z: padded conjugate (@zpk/ctranspose.m convention).
            K_num, K_den = _z2zeta(K_num_d0, K_den_d0)
            lead = float(K_den[0])
            if abs(lead) > 1e-30:
                K_num = K_num / lead
                K_den = K_den / lead

            # Cost: err = errLm + err0 (polquad.m lines 118-133) — exact via
            # intz2b like MATLAB (err0 = intz2b(E, -minreal(B·B~/A)),
            # errLm = norm(zpk(Pi,dLm))² = intz2b(Lm·Lm~)); grid integration
            # kept as a nested fallback so a factorization failure here
            # cannot dump us into the closed-loop path below.
            Pi_num_ol = _r(_strip_lz(Pi_pol.coef))
            try:
                _lb_ol = max(len(B_num), len(B_den))
                _Bn_ol = np.concatenate([np.zeros(_lb_ol - len(B_num)), B_num])
                _Bd_ol = np.concatenate([np.zeros(_lb_ol - len(B_den)), B_den])
                # polquad.m: BBA = minreal(B*B'/A) — plain minreal (see the
                # closed-loop cost block below).
                _BBA_ol = Minreal.tf(
                    np.polymul(np.polymul(_Bn_ol, _Bn_ol[::-1]), A_den),
                    np.polymul(np.polymul(_Bd_ol, _Bd_ol[::-1]), A_num),
                    tol=1e-3)
                _lm_ol = max(len(Pi_num_ol), len(dLm_ol))
                _Pi_ol = np.concatenate([np.zeros(_lm_ol - len(Pi_num_ol)), Pi_num_ol])
                _dm_ol = np.concatenate([np.zeros(_lm_ol - len(dLm_ol)), dLm_ol])
                errLm, _ = _intz2b((np.polymul(_Pi_ol, _Pi_ol[::-1]),
                                    np.polymul(_dm_ol, _dm_ol[::-1])))
                err0, _ = _intz2b((E_num, E_den), (-_BBA_ol[0], _BBA_ol[1]))
            except Exception:
                _w_ol = np.linspace(-np.pi + 1e-6, np.pi - 1e-6, 4096)
                _z_ol = np.exp(1j * _w_ol)
                _A_f_ol = np.polyval(A_num, _z_ol) / (np.polyval(A_den, _z_ol) + 1e-300)
                _B_f_ol = np.polyval(B_num, _z_ol) / (np.polyval(B_den, _z_ol) + 1e-300)
                _E_f_ol = np.polyval(E_num, _z_ol) / (np.polyval(E_den, _z_ol) + 1e-300)
                _Z_f_ol = np.real(_E_f_ol - np.abs(_B_f_ol) ** 2 / (_A_f_ol + 1e-300))
                _trap_ol = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
                err0 = float(abs(_trap_ol(_Z_f_ol, _w_ol) / (2 * np.pi)))
                _Lm_f_ol = np.polyval(Pi_num_ol, _z_ol) / (np.polyval(dLm_ol, _z_ol) + 1e-300)
                errLm = float(abs(_trap_ol(np.abs(_Lm_f_ol) ** 2, _w_ol) / (2 * np.pi)))

            err = errLm + err0
            return (striplz(K_num), striplz(K_den)), err
        except Exception:
            pass  # fall through to the closed-loop path below (should not normally happen)

    yd = _ynyd(A_tf, B_tf, D22_tf, P_cancel,
               zpk_in=(zpk_in[0], zpk_in[1]) if zpk_in is not None else None)
    a0, Lam, dLp, dLm, dL0 = yd['a0'], yd['Lam'], yd['dLp'], yd['dLm'], yd['dL0']
    na, nb, db, da, YN, YD = yd['na'], yd['nb'], yd['db'], yd['da'], yd['YN'], yd['YD']

    # ── Diophantine system (polquad.m lines 69-110) ─────────────────────────
    # Equations (shared X = Pi, separate Y1, Y2 = N/D):
    #   AN·X + BN·Y1 = CN      AN = dLp·na,  BN = dLm,  CN = YN
    #   AD·X + BD·Y2 = CD      AD = −dLp·nb, BD = dLm,  CD = YD
    # gD0 = dL0 (neutral roots extracted for K denominator, polquad.m line 89-97)
    K_num, K_den = np.array([1.0]), np.array([1.0])
    X_pol = None
    try:
        AN = _r(np.polymul(dLp, na))
        AD = _r(-np.polymul(dLp, nb))

        AN_pol = Poln(AN, 'z'); BN_pol = Poln(dLm, 'z'); CN_pol = Poln(YN, 'z')
        AD_pol = Poln(AD, 'z'); BD_pol = Poln(dLm, 'z'); CD_pol = Poln(YD, 'z')

        # Extract common factors AN↔BN and AD↔BD (polquad.m coprime steps)
        AN_r, BN_r, gABN = _cprime(AN_pol, BN_pol)
        CN_q, CN_rem = CN_pol.quorem(gABN)
        CN_r = CN_q if CN_rem.norm() < 1e-3 * max(CN_pol.norm(), 1e-30) else CN_pol

        AD_r, BD_r, gABD = _cprime(AD_pol, BD_pol)
        CD_q, CD_rem = CD_pol.quorem(gABD)
        CD_r = CD_q if CD_rem.norm() < 1e-3 * max(CD_pol.norm(), 1e-30) else CD_pol

        # Try to extract dL0 (neutral factor) from CD (polquad.m lines 89-97)
        gD0_pol = Poln([1.0], 'z')
        if len(dL0) > 1:
            dL0_pol = Poln(dL0, 'z')
            CDx_q, CDx_rem = CD_r.quorem(dL0_pol)
            if CDx_rem.norm() < 1e-3 * max(CD_r.norm(), 1e-30):
                AD_r, _ = AD_r.quorem(dL0_pol)
                CD_r = CDx_q
                gD0_pol = dL0_pol

        AN_r2, CN_r2, gN = _cprime(AN_r, CN_r)
        AD_r2, CD_r2, gD = _cprime(AD_r, CD_r)

        X_pol, Y1_pol, Y2_pol, dioph_err, _ = _dioph_sys(AN_r2, BN_r, CN_r2, AD_r2, BD_r, CD_r2)

        # K (d-domain) = db·gN·Y1 / (da·gD0·gD·Y2) then reverse for z-domain.
        # MATLAB polquad.m line 103/108: [Pi,N,D] = diophsys(AN,BN,CN,AD,BD,CD);
        # K = minreal(zpk(db*gN*N, da*gD0*gD*D, T)) — K's numerator uses N (=Y1,
        # diophsys's *second* output, from the AN/BN/CN equation), not Pi (the
        # shared solution, X here). Using X_pol here was a variable mix-up:
        # Pi/X is the "unavoidable cost" term (used
        # separately for errLm below), not part of K itself.
        K_num_p = Poln(db, 'z') * gN * Y1_pol
        K_den_p = Poln(da, 'z') * gD0_pol * gD * Y2_pol

        # MATLAB polquad.m line 108: K = minreal(zpk(...), 1e-3) — cancels
        # common pole/zero pairs before returning K. Missing this step left an
        # uncancelled (z-1)-type factor (from an integrator in the plant) in
        # both K_num and K_den whenever dL0 is nontrivial, producing a K with
        # a marginally-unstable pole exactly on the unit circle that the
        # later stability check (rightly) rejects.
        K_num_p, K_den_p, _ = _cprime(K_num_p, K_den_p, tol=1e-3)

        K_num_d = _r(_strip_lz(K_num_p.coef))
        K_den_d = _r(_strip_lz(K_den_p.coef))
        if len(K_den_d) == 0 or np.all(np.abs(K_den_d) < 1e-30):
            raise ValueError("zero denominator")

        # d-domain to z-domain: K' via the padded conjugate
        # (@zpk/ctranspose.m adds origin zeros/poles for the relative-degree
        # difference; naive per-array reversal drops them).
        K_num, K_den = _z2zeta(K_num_d, K_den_d)

        # Normalize so K_den is monic
        lead = float(K_den[0])
        if abs(lead) > 1e-30:
            K_num = K_num / lead
            K_den = K_den / lead

    except Exception:
        pass  # K stays at [1]/[1] fallback

    # ── Cost: err = errLm + err0  (polquad.m lines 118-133) ─────────────────
    # err0 = (1/2π)∫(E - B*B~/A)dω computed EXACTLY: BBA = minreal(B*B'/A)
    # then err0 = intz2b(E, -BBA), matching polquad.m verbatim. Grid
    # integration (the previous approach, kept as fallback) explodes near
    # smeared high-multiplicity |z|=1 roots of A/B/E for integrator plants —
    # reporting a garbage cost even when K itself is correct.
    # errLm = ||Pi/dLm||²_H2 = intz2b(Lm·Lm~), MATLAB's norm(zpk(Pi,dLm))².
    err = float('nan')
    Pi_num = _r(_strip_lz(X_pol.coef)) if X_pol is not None else None
    try:
        _lb = max(len(B_num), len(B_den))
        _B_n_pad = np.concatenate([np.zeros(_lb - len(B_num)), B_num])
        _B_d_pad = np.concatenate([np.zeros(_lb - len(B_den)), B_den])
        _BB_n = np.polymul(_B_n_pad, _B_n_pad[::-1])
        _BB_d = np.polymul(_B_d_pad, _B_d_pad[::-1])
        # polquad.m line ~127: BBA = minreal(B*B'/A) — PLAIN minreal here
        # (zterm.m is the one that uses minreals; matching MATLAB's actual
        # call also matters numerically: tf_symmetric's pair-preserving
        # reduction leaves this particular ratio unreduced for smeared
        # integrator plants, and intz2b's factorization then fails).
        BBA_n, BBA_d = Minreal.tf(np.polymul(_BB_n, A_den),
                                  np.polymul(_BB_d, A_num), tol=1e-3)

        errLm = 0.0
        if Pi_num is not None:
            _lm = max(len(Pi_num), len(dLm))
            _Pi_p = np.concatenate([np.zeros(_lm - len(Pi_num)), Pi_num])
            _dm_p = np.concatenate([np.zeros(_lm - len(dLm)), dLm])
            errLm, _ = _intz2b((np.polymul(_Pi_p, _Pi_p[::-1]),
                                np.polymul(_dm_p, _dm_p[::-1])))
        err0, _ = _intz2b((E_num, E_den), (-BBA_n, BBA_d))
        err = errLm + err0
    except Exception as _cost_exc:
        warnings.warn(f"exact cost via intz2b failed ({_cost_exc}); "
                      f"falling back to grid integration")
        try:
            _w   = np.linspace(-np.pi + 1e-6, np.pi - 1e-6, 4096)
            _z   = np.exp(1j * _w)
            _A_f = np.polyval(A_num, _z) / (np.polyval(A_den, _z) + 1e-300)
            _B_f = np.polyval(B_num, _z) / (np.polyval(B_den, _z) + 1e-300)
            _E_f = np.polyval(E_num, _z) / (np.polyval(E_den, _z) + 1e-300)
            _Z_f = np.real(_E_f - np.abs(_B_f) ** 2 / (_A_f + 1e-300))
            _trap = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            err0 = float(abs(_trap(_Z_f, _w) / (2 * np.pi)))

            errLm = 0.0
            if Pi_num is not None:
                _Lm_f = np.polyval(Pi_num, _z) / (np.polyval(dLm, _z) + 1e-300)
                errLm = float(abs(_trap(np.abs(_Lm_f) ** 2, _w) / (2 * np.pi)))

            err = errLm + err0
        except Exception:
            pass

    # Validity check: a genuine H2-optimal controller must strictly stabilize
    # the CLOSED LOOP — the controller ITSELF may legitimately be marginal
    # or even unstable: MATLAB polquad.m (lines 89-97) deliberately places
    # dL0's unit-circle roots into K's denominator via gD0 (integral
    # action). The previous version of this check rejected K's *own* poles
    # within 1e-3 of the unit circle, which threw away exactly those
    # legitimate integral-action controllers — measured via a sweep over the
    # default-method demo/example plants: about half of all 'pol' rejections
    # were at K-pole magnitude ≈ 1.00.
    #
    # Test the closed-loop characteristic polynomial instead:
    #   char(z) = P22d_den·K_den + P22d_num·K_num,  P22d = z2zeta(D22)
    # (M = K/(1+D22·K); the design convention has already negated the
    # measurement row, matching _sderr). Genuinely corrupted results (e.g.
    # the ZOH double-integrator garbage from smeared high-multiplicity
    # roots) put closed-loop poles on/outside the circle and are still
    # rejected, so callers' existing try/except falls back to 'ss' as
    # before; K's deliberate marginal poles pass as long as the loop is
    # strictly stable.
    _K_num_final = striplz(K_num.real)
    _K_den_final = striplz(K_den.real)
    _P22n_z, _P22d_z = _z2zeta(
        np.real(np.atleast_1d(np.asarray(D22_tf[0]))).astype(float),
        np.real(np.atleast_1d(np.asarray(D22_tf[1]))).astype(float))
    _char = striplz(np.real(np.polyadd(
        np.polymul(_P22d_z, _K_den_final),
        np.polymul(_P22n_z, _K_num_final))))
    # OPEN-LOOP problems (D22 ≡ 0, MATLAB polquad.m's norm(n) < eps branch —
    # e.g. sd2dof's Ψ optimization) have no loop to stabilize: char would
    # degenerate to K's own denominator, and the result Ψ may legitimately
    # carry unit-circle poles. Only gate genuine closed-loop designs.
    _open_loop = np.linalg.norm(np.atleast_1d(np.asarray(D22_tf[0], float))) \
        < np.finfo(float).eps
    if len(_char) > 1 and not _open_loop:
        _cl_poles = np.roots(_char)
        if np.any(np.abs(_cl_poles) > 1.0 - 1e-6):
            raise ValueError(
                f"_polquad: resulting K does not strictly stabilize the "
                f"closed loop (pole magnitude "
                f"{np.max(np.abs(_cl_poles)):.6f} on or outside the unit "
                f"circle) — likely a plant with high-multiplicity "
                f"poles/zeros on the unit circle causing numerical error to "
                f"compound through the Diophantine chain."
            )

    return (_K_num_final, _K_den_final), err


# ---------------------------------------------------------------------------
# whquad – Wiener-Hopf quadratic optimisation
# ---------------------------------------------------------------------------

def whquad(A, B, E, D22, P_cancel=None):
    """Wiener-Hopf minimisation of a sampled-data quadratic functional.

    Minimises  J = (1/2π) ∫ [A|M|² + E − 2Re(B·M)] dω  subject to M being
    a proper realised stable closed-loop map M = D22·K·(I+D22·K)⁻¹.

    Uses stable/unstable decomposition of L = L1 − L2 (via polynomial
    factorisation + Diophantine equation) to find the optimal controller.
    The result equals polquad for the same inputs.

    Parameters
    ----------
    A, B, E, D22 : (num, den) tuples or array-like
        Rational TF coefficients (z-domain, descending power).
    P_cancel : ignored, kept for API compatibility with polquad.

    Returns
    -------
    K : (num_d, den_d) tuple
    err : float
        Square-root of the optimal cost.
    """
    return _whquad(_ensure_tf(A), _ensure_tf(B), _ensure_tf(E), _ensure_tf(D22))


def _whquad(A_tf, B_tf, E_tf, D22_tf, P_cancel=None):
    def _r(c):
        return np.real(np.asarray(c, float)).ravel()

    A_num,   A_den   = _r(A_tf[0]),   _r(A_tf[1])
    B_num,   B_den   = _r(B_tf[0]),   _r(B_tf[1])
    E_num,   E_den   = _r(E_tf[0]),   _r(E_tf[1])
    D22_num, D22_den = _r(D22_tf[0]), _r(D22_tf[1])

    n_arr = _strip_lz(D22_num.copy())
    d_arr = _strip_lz(D22_den.copy())

    # --- Spectral factor: Lam² = d²d̃²A (same as polquad) ---
    try:
        d_p   = Poln(_real_if_close(_strip_lz(d_arr)), 'z')
        d_cr_p = d_p.conj_reciprocal()
        Av    = d_p * d_p * d_cr_p * d_cr_p * Poln(_real_if_close(_strip_lz(A_num)), 'z')
        Lam_arr = np.real(sfactor(Av, 'z').coef).ravel()
    except Exception:
        Lam_arr = np.array([1.0])

    Lam_cr = Lam_arr[::-1]
    d_cr   = d_arr[::-1]

    # --- Basic controller: a0·n + b0·d = 1 ---
    try:
        n_p = Poln(_real_if_close(_strip_lz(n_arr)), 'z')
        d_p = Poln(_real_if_close(_strip_lz(d_arr)), 'z')
        a0_p, b0_p, _, _ = dioph(n_p, d_p, Poln([1.0], 'z'))
        a0 = np.real(a0_p.coef).ravel()
        b0 = np.real(b0_p.coef).ravel()
    except Exception:
        a0 = np.array([1.0])
        b0 = np.array([0.0])

    # --- L1 = d̃²B̃/Lam̃, L2 = a0·Lam/d ---
    B_num_cr = _strip_lz(B_num)[::-1]
    B_den_cr = _strip_lz(B_den)[::-1]
    L1_num = np.convolve(np.convolve(d_cr, d_cr), B_num_cr)
    L1_den = np.convolve(Lam_cr, B_den_cr)
    L2_num = np.convolve(a0, Lam_arr)
    L2_den = d_arr.copy()

    # --- L = L1 − L2 as rational: num/den ---
    L_num = _strip_lz(np.real(np.polyadd(
        np.convolve(_strip_lz(L1_num), _strip_lz(L2_den)),
        -np.convolve(_strip_lz(L2_num), _strip_lz(L1_den)),
    )))
    L_den = _strip_lz(np.real(np.convolve(_strip_lz(L1_den), _strip_lz(L2_den))))
    if np.linalg.norm(L_num) < 1e-12:
        L_num = np.array([0.0])

    # --- Factor L_den = dLp · dLm · dL0 (stable · unstable · neutral) ---
    try:
        dLp_obj, dLm_obj, dL0_obj = factor(
            Poln(_real_if_close(_strip_lz(L_den)), 'z'), 'z'
        )
        dLp = _strip_lz(np.real((dLp_obj * dL0_obj).coef).ravel())  # causal
        dLm = _strip_lz(np.real(dLm_obj.coef).ravel())              # anti-causal
    except Exception:
        dLp = L_den.copy()
        dLm = np.array([1.0])

    # --- Diophantine split: X·dLp + Y·dLm = L_num ---
    # Lp = Y/dLp (stable poles), Lm = X/dLm (unstable poles)
    try:
        X_p, Y_p, _, _ = dioph(
            Poln(_real_if_close(_strip_lz(dLp)), 'z'),
            Poln(_real_if_close(_strip_lz(dLm)), 'z'),
            Poln(_real_if_close(_strip_lz(L_num)), 'z'),
        )
        nLp = _strip_lz(np.real(Y_p.coef).ravel())
        nLm = _strip_lz(np.real(X_p.coef).ravel())  # noqa: F841 (kept for cost)
    except Exception:
        nLp = np.array([0.0])
        nLm = L_num.copy()  # noqa: F841

    # --- K = (a0·dLp·Lam + d·nLp) / (b0·dLp·Lam − n·nLp) ---
    dLp_Lam = np.convolve(dLp, Lam_arr)
    K_num = _strip_lz(np.real(np.polyadd(
        np.convolve(a0,    dLp_Lam),
        np.convolve(d_arr, nLp),
    )))
    K_den = _strip_lz(np.real(np.polyadd(
        np.convolve(b0,    dLp_Lam),
        -np.convolve(n_arr, nLp),
    )))

    if len(K_den) == 0 or np.linalg.norm(K_den) < 1e-10:
        K_num = np.array([1.0])
        K_den = np.array([1.0])

    # --- Numerical cost (same as polquad) ---
    err = float('nan')
    try:
        N_pts = 2048
        w_arr = np.linspace(-np.pi, np.pi, N_pts)
        z_arr = np.exp(1j * w_arr)
        A_f   = np.polyval(A_num,   z_arr) / (np.polyval(A_den,   z_arr) + 1e-300)
        E_f   = np.polyval(E_num,   z_arr) / (np.polyval(E_den,   z_arr) + 1e-300)
        K_f   = np.polyval(K_num,   z_arr) / (np.polyval(K_den,   z_arr) + 1e-300)
        D22_f = np.polyval(D22_num, z_arr) / (np.polyval(D22_den, z_arr) + 1e-300)
        S_f   = 1.0 / (1.0 + D22_f * K_f + 1e-300)
        intgrd = np.real(A_f * np.abs(1.0 - S_f)**2 + E_f * np.abs(S_f)**2)
        _trap = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        err = float(np.sqrt(max(abs(_trap(intgrd, w_arr)) / (2 * np.pi), 0.0)))
    except Exception:
        pass

    return (striplz(K_num), striplz(K_den)), err


# ---------------------------------------------------------------------------
# ssquad – state-space quadratic optimisation
# ---------------------------------------------------------------------------

def ssquad(A0, A1, B, E, D22):
    """State-space minimisation of a sampled-data quadratic functional.

    Minimises the same functional as polquad(A0·A1, B, E, D22) using a
    state-space 2×2 generalised plant and H2 Riccati equations.  The optimal
    controller equals that returned by polquad when A = A0·A1.

    This is the Python port of ssquad.m (K. Polyakov) combined with
    abe2std.m.  In this port the state-space plant is not constructed
    explicitly; instead the algebraic equivalence ssquad ≡ polquad(A0·A1, ...)
    is exploited directly.

    Parameters
    ----------
    A0, A1 : (num, den) tuples
        Factored spectral density coefficients (A = A0·A1).
    B, E, D22 : (num, den) tuples
        Quadratic cost terms and plant model.

    Returns
    -------
    K : (num_d, den_d) tuple
    err : float
        Square-root of the optimal cost.
    """
    def _r(c):
        return np.real(np.asarray(c, float)).ravel()

    A0_tf = _ensure_tf(A0)
    A1_tf = _ensure_tf(A1)
    # Merge factored spectral density: A = A0 · A1
    A_num = np.polymul(_r(A0_tf[0]), _r(A1_tf[0]))
    A_den = np.polymul(_r(A0_tf[1]), _r(A1_tf[1]))
    return _polquad((A_num, A_den), _ensure_tf(B), _ensure_tf(E), _ensure_tf(D22))


# ---------------------------------------------------------------------------
# psigain – DC-gain of Psi-parameter for tracking system
# ---------------------------------------------------------------------------

def psigain(sys, T, Delta=None):
    """DC-gain of Psi-parameter for tracking system.

    For a generalized plant, computes the required DC-gain G of the free
    parameter Psi in the controller parameterisation K = (a0+d·G)/(b0−n·G)
    that achieves asymptotic tracking with a step (1/s) reference signal.

    Port of psigain.m (K. Polyakov).

    Parameters
    ----------
    sys : StateSpace, lti, or (num, den) tuple
        CT generalised plant (MIMO 2×2 or SISO auto-expanded).
    T : float
        Sampling period.
    Delta : array-like, optional
        Characteristic polynomial in z (default: [1.0]).

    Returns
    -------
    G : float or nan
        Required DC-gain of Psi. NaN if underdetermined.
    GK : float or nan or inf
        Resulting DC-gain of the closed-loop controller.
    a0 : 1D ndarray
        Bezout coefficient (a0·n + b0·d = Delta).
    b0 : 1D ndarray
        Bezout coefficient.
    """
    # --- Convert plant to StateSpace and negate last output (→ negative feedback) ---
    plant_ss, _, _ = _parse_plant(sys)
    A = plant_ss.A.copy()
    B = plant_ss.B.copy()
    C = plant_ss.C.copy()
    D_mat = plant_ss.D.copy()
    C[-1, :] = -C[-1, :]
    D_mat[-1, :] = -D_mat[-1, :]
    plant_ss = sig.StateSpace(A, B, C, D_mat)

    nout = plant_ss.C.shape[0]
    nin  = plant_ss.B.shape[1]
    n_ctrl, n_meas = 1, 1
    n_perf = nout - n_meas
    n_w    = nin  - n_ctrl

    # --- Default Delta = [1] ---
    if Delta is None:
        Delta_arr = np.array([1.0])
    else:
        Delta_arr = _strip_lz(np.real(np.asarray(Delta, float)).ravel())
    if len(Delta_arr) > 1:
        dr = np.roots(Delta_arr)
        if np.any(np.abs(dr - 1.0) < 1e-5):
            raise ValueError("Delta has roots at z=1 (marginally stable)")

    # --- Extract SISO TF blocks from the shared StateSpace A matrix ---
    def _block_tf(row_slice, col_slice):
        ri = list(range(*row_slice.indices(nout)))
        ci = list(range(*col_slice.indices(nin)))
        Cb = plant_ss.C[ri, :]
        Db = plant_ss.D[np.ix_(ri, ci)]
        Bb = plant_ss.B[:, ci]
        ss_b = sig.StateSpace(plant_ss.A, Bb, Cb, Db)
        tf_b = ss_b.to_tf()
        try:
            num = np.real(np.atleast_1d(tf_b.num[0][0])).ravel()
            den = np.real(np.atleast_1d(tf_b.den[0][0])).ravel()
        except (TypeError, IndexError):
            num = np.real(np.atleast_1d(tf_b.num)).ravel()
            den = np.real(np.atleast_1d(tf_b.den)).ravel()
        return _strip_lz(num), _strip_lz(den)

    P11 = _block_tf(slice(0, n_perf),  slice(0, n_w))
    P12 = _block_tf(slice(0, n_perf),  slice(n_w, None))
    P21 = _block_tf(slice(n_perf, None), slice(0, n_w))
    P22 = _block_tf(slice(n_perf, None), slice(n_w, None))

    # --- Discretise P22 with ZOH → (D22_num, D22_den) ---
    P22_ss = sig.StateSpace(plant_ss.A,
                            plant_ss.B[:, n_w:],
                            plant_ss.C[n_perf:, :],
                            plant_ss.D[n_perf:, n_w:])
    D22_dt  = P22_ss.to_discrete(T, method='zoh')
    D22_tf  = D22_dt.to_tf()
    try:
        D22_num = np.atleast_1d(D22_tf.num[0][0]).ravel()
        D22_den = np.atleast_1d(D22_tf.den[0][0]).ravel()
    except (TypeError, IndexError):
        D22_num = np.atleast_1d(D22_tf.num).ravel()
        D22_den = np.atleast_1d(D22_tf.den).ravel()
    D22_num = _strip_lz(np.real(D22_num))
    D22_den = _strip_lz(np.real(D22_den))

    # --- Basic controller: a0·n + b0·d = Delta ---
    n_p     = Poln(_real_if_close(_strip_lz(D22_num)), 'z')
    d_p     = Poln(_real_if_close(_strip_lz(D22_den)), 'z')
    Delta_p = Poln(_real_if_close(_strip_lz(Delta_arr)), 'z')
    try:
        a0_p, b0_p, _, _ = dioph(n_p, d_p, Delta_p)
        a0 = np.real(a0_p.coef).ravel()
        b0 = np.real(b0_p.coef).ravel()
    except Exception:
        a0 = np.array([1.0])
        b0 = np.array([0.0])

    # --- CT TF helpers (s-domain polynomial arithmetic) ---
    def _prod(tf1, tf2):
        return np.polymul(tf1[0], tf2[0]), np.polymul(tf1[1], tf2[1])

    def _div(tf1, tf2):
        return np.polymul(tf1[0], tf2[1]), np.polymul(tf1[1], tf2[0])

    def _sub(tf1, tf2):
        num = np.polyadd(np.polymul(tf1[0], tf2[1]), -np.polymul(tf2[0], tf1[1]))
        den = np.polymul(tf1[1], tf2[1])
        return num, den

    def _mul_s(tf):
        """Multiply by s (add a zero at s=0)."""
        return np.polymul(tf[0], np.array([1.0, 0.0])), tf[1].copy()

    def _simplify_s0(tf):
        """Cancel common trailing-zero (s=0) factors between num and den."""
        num = np.real(np.asarray(tf[0])).ravel().astype(float)
        den = np.real(np.asarray(tf[1])).ravel().astype(float)

        def _tz(p):
            k = 0
            for c in reversed(p.tolist()):
                if abs(c) < 1e-10: k += 1
                else: break
            return k

        cancel = min(_tz(num), _tz(den))
        if cancel > 0:
            num = num[:len(num) - cancel]
            den = den[:len(den) - cancel]
        return np.asarray(num, float), np.asarray(den, float)

    def _count_s0(tf):
        """Count poles at s=0 (trailing zeros in denominator after simplification)."""
        _, den = _simplify_s0(tf)
        k = 0
        for c in reversed(den.tolist()):
            if abs(c) < 1e-10: k += 1
            else: break
        return k

    def _s0_lim(tf, extra=0):
        """Compute lim_{s→0} s^extra * N(s)/D(s)."""
        num = np.real(np.asarray(tf[0])).ravel().astype(float)
        den = np.real(np.asarray(tf[1])).ravel().astype(float)
        if extra > 0:
            num = np.polymul(num, np.array([1.0] + [0.0] * extra))
        num, den = _simplify_s0((num, den))
        n0 = float(num[-1]) if len(num) > 0 else 0.0
        d0 = float(den[-1]) if len(den) > 0 else 1.0
        return n0 / d0 if abs(d0) > 1e-10 else float('nan')

    # --- CT poles of P22 at s=0 ---
    if len(P22[1]) > 1:
        z22 = int(np.sum(np.abs(np.roots(P22[1])) < 1e-6))
    else:
        z22 = 0

    # --- Modified blocks (multiply P11, P21 by s for step-input analysis) ---
    P11m = _simplify_s0(_mul_s(P11))
    P21m = _simplify_s0(_mul_s(P21))
    Px   = _simplify_s0(_prod(P12, P21m))   # P12 · P21 · s
    Py   = _simplify_s0(_div(Px, P22))      # P12 · P21 · s / P22
    Pa   = _simplify_s0(_sub(P11m, Py))     # P11 · s − P12 · P21 · s / P22

    # --- Pole counts at s=0 ---
    za = _count_s0(Pa)
    zx = _count_s0(Px)
    zy = _count_s0(Py)

    # --- V2: DC contribution from P12·P21·d²/Delta at z=1 ---
    def _remove_z1_once(poly):
        """Divide poly by (z−1) via synthetic division."""
        p = np.real(np.asarray(poly)).ravel().astype(float)
        n = len(p) - 1
        if n <= 0:
            return np.array([p[0]] if len(p) > 0 else [0.0])
        q = np.zeros(n)
        q[0] = p[0]
        for i in range(1, n):
            q[i] = p[i] + q[i - 1]
        return q

    def _z1_eval(poly, k):
        """Evaluate poly(z) / (z−1)^k at z=1 via repeated synthetic division."""
        p = np.real(np.asarray(poly)).ravel().astype(float)
        for _ in range(k):
            if len(p) <= 1:
                return float(p[0]) if len(p) == 1 else 0.0
            p = _remove_z1_once(p)
        return float(np.polyval(p, 1.0))

    Delta_at1 = float(np.polyval(Delta_arr, 1.0))

    z2 = zx - 2 * z22
    if z2 < 0:
        V2 = 0.0
        z2 = 0
    else:
        PxF_dc = _s0_lim(Px, extra=zx)
        d_sq   = np.polymul(D22_den, D22_den)
        # (-T)^{2·z22} is T^{2·z22} since power is always even
        d2R_at1 = _z1_eval(d_sq, 2 * z22) * ((-T) ** (2 * z22))
        V2 = PxF_dc * d2R_at1 / Delta_at1 if abs(Delta_at1) > 1e-10 else float('nan')

    # --- V1: DC contribution from Pa and Py ---
    z1 = max(za, zy - z22)
    z0 = min(z22, zy)

    Pa_dc   = _s0_lim(Pa, extra=z1)
    Py2_dc  = _s0_lim(Py, extra=z1 + z22)   # = lim s^{z1+z22} · Py(s)
    dR_at1  = _z1_eval(D22_den, z0) * ((-T) ** z0)
    b0_at1  = float(np.polyval(b0, 1.0))

    V1 = Pa_dc
    if not np.isnan(Py2_dc) and abs(Delta_at1) > 1e-10:
        V1 = V1 + Py2_dc * b0_at1 * dR_at1 / Delta_at1

    # --- Solve for G ---
    if z1 != z2:
        raise ValueError(
            f"Special case of asymptotic tracking (z1={z1} ≠ z2={z2})"
        )

    if abs(V2) > 1e-6:
        G = float(np.real(V1 / V2))
        if abs(V1 - V2 * G) > 1e-6:
            raise ValueError("Asymptotic tracking is impossible")
    else:
        if abs(V1) < 1e-6:
            G = float('nan')
        else:
            raise ValueError("Asymptotic tracking is impossible (V2≈0, V1≠0)")

    # --- Controller DC gain GK = K(z=1) ---
    if np.isnan(G):
        GK = float('nan')
    else:
        K_num_at1 = float(np.polyval(a0, 1.0)) + float(np.polyval(D22_den, 1.0)) * G
        K_den_at1 = float(np.polyval(b0, 1.0)) - float(np.polyval(D22_num, 1.0)) * G
        GK = (K_num_at1 / K_den_at1) if abs(K_den_at1) > 1e-10 else float('inf')

    return G, GK, a0, b0


# ---------------------------------------------------------------------------
# polopth2 – polynomial H2 minimisation: min_Q ||V - W*Q||_H2
# ---------------------------------------------------------------------------

def polopth2(V, W, n):
    """
    Polynomial H2-minimisation.

    Finds the degree-n polynomial Q minimising ||V(z) - W(z)*Q(z)||_{H2}.

    Parameters
    ----------
    V, W : (num, den) tuples (z-domain, highest-power-first)
        Stable discrete-time transfer functions.
    n : int
        Degree of the desired polynomial Q.

    Returns
    -------
    Q_coef : np.ndarray
        Optimal polynomial coefficients (highest-power-first).
    E : float
        Optimal H2-norm squared: ||V - W*Q||_{H2}^2.
    """
    def _r(c):
        return np.real(np.asarray(c, float)).ravel()

    V_tf = _ensure_tf(V)
    W_tf = _ensure_tf(W)
    nV = striplz(_r(V_tf[0])); dV = striplz(_r(V_tf[1]))
    nW = striplz(_r(W_tf[0])); dW = striplz(_r(W_tf[1]))

    # Conjugate reciprocal: W~(z) = nWz/dWz (reverse coefficients for real polys)
    nWz = striplz(nW[::-1])
    dWz = striplz(dW[::-1])

    # Build polynomials for the linear system (port of polopth2.m)
    pQ = np.polymul(nWz, np.polymul(nW, dV))
    pP = -np.polymul(dW, dV)
    pT = -np.concatenate([dWz, np.zeros(n + 1)])
    pT = striplz(pT)
    c  = np.polymul(nWz, np.polymul(nV, dW))

    def _pdeg(p):
        p2 = striplz(np.asarray(p, float).ravel())
        return max(len(p2) - 1, 0)

    degP = len(striplz(dWz)) - 2   # deg(dWz) - 1
    degQ = int(n)
    maxDeg = max(_pdeg(pQ) + degQ, max(_pdeg(pP) + degP, 0) if degP >= 0 else 0, _pdeg(c))
    degT = maxDeg - _pdeg(pT)

    def _convmat(p, r, c_cols):
        """Lower-triangular Toeplitz convolution matrix (highest-first convention).
        _convmat(p, r, c_cols) @ x_coef = polymul(p, x_coef)[:r]
        where M[i,j] = p[i-j] for 0 <= i-j < len(p).
        """
        p = np.asarray(p, float).ravel()
        m = len(p)
        M = np.zeros((r, c_cols))
        for j in range(c_cols):
            for i in range(r):
                k = i - j
                if 0 <= k < m:
                    M[i, j] = p[k]
        return M

    rows = maxDeg + 1
    AQ = _convmat(pQ, rows, degQ + 1)
    blocks = [AQ]
    if degP >= 0:
        AP = _convmat(pP, rows, degP + 1)
        blocks.append(AP)
    if degT >= 0:
        AT = _convmat(pT, rows, degT + 1)
        blocks.append(AT)
    A = np.hstack(blocks)
    B = _convmat(c, rows, 1)

    # Solve the linear system (least-squares for over-determined / rank-deficient)
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    X = np.real(X.ravel())

    # Extract Q: first (degQ+1) entries, reversed to descending-power order
    Q_coef = striplz(X[:degQ + 1][::-1])

    # Compute E = ||V - W*Q||_{H2}^2 via FFT on the unit circle
    N_pts = 4096
    theta = np.linspace(0, 2 * np.pi, N_pts, endpoint=False)
    z_pts = np.exp(1j * theta)
    V_pts = np.polyval(nV, z_pts) / (np.polyval(dV, z_pts) + 1e-300)
    W_pts = np.polyval(nW, z_pts) / (np.polyval(dW, z_pts) + 1e-300)
    Q_pts = np.polyval(Q_coef, z_pts)
    E = float(np.mean(np.abs(V_pts - W_pts * Q_pts) ** 2))

    return Q_coef, E


# ---------------------------------------------------------------------------
# sdh2coef / sdl2coef – coefficient computation using Van Loan integrals
# ---------------------------------------------------------------------------

def _pf_split(F_z, target_roots, tol=1e-4):
    """
    Partial-fraction split of a SISO root-list rational F = F0 + F1 where
    F0 is STRICTLY PROPER with denominator exactly the poles of F matching
    `target_roots` (multiset, greedy nearest within tol), and F1 carries the
    remaining poles plus the direct/improper part.

    Equivalent of zterm.m's separss(F, dCancel.z, 'infu') calls (the
    E-syntax eigenvalue separation with the DC gain kept in the remainder
    part), done by polynomial Diophantine instead of Schur separation:
        n0·d1 + n1·d0 = n,  deg n0 < deg d0
        F0 = n0/d0,  F1 = n1/d1.
    """
    poles = list(F_z.p)
    avail = list(np.atleast_1d(np.asarray(target_roots, complex)))
    d0_roots, d1_roots = [], []
    for q in poles:
        hit = -1
        best = tol * (1.0 + abs(q))
        for j, t in enumerate(avail):
            if abs(q - t) < best:
                best = abs(q - t); hit = j
        if hit >= 0:
            avail.pop(hit)
            d0_roots.append(q)
        else:
            d1_roots.append(q)
    if not d0_roots:
        return Zpk([], [], 0.0), F_z.copy()
    d0 = np.real(np.poly(d0_roots))
    d1 = np.real(np.poly(d1_roots)) if d1_roots else np.array([1.0])
    n = (np.real(F_z.k * np.poly(F_z.z)) if len(F_z.z)
         else np.array([F_z.k]))
    # solve n0·d1 + n1·d0 = n with deg n0 < deg d0 (linear system in the
    # stacked coefficients; consistent because gcd(d0, d1) = 1 by
    # construction — matched vs unmatched root sets are disjoint)
    m0 = len(d0) - 1
    deg_n1 = max(len(n) - 1 - m0, len(d1) - 1, 0)
    n_out = max(len(n), m0 + len(d1) - 1, deg_n1 + m0 + 1)
    M = np.zeros((n_out, m0 + deg_n1 + 1))
    for i in range(m0):          # n0 coefficients (deg m0-1)
        col = np.zeros(n_out)
        conv = d1
        col[n_out - (m0 - 1 - i) - len(conv):n_out - (m0 - 1 - i)] = conv
        M[:, i] = col
    for i in range(deg_n1 + 1):  # n1 coefficients
        col = np.zeros(n_out)
        conv = d0
        col[n_out - (deg_n1 - i) - len(conv):n_out - (deg_n1 - i)] = conv
        M[:, m0 + i] = col
    rhs = np.zeros(n_out)
    rhs[n_out - len(n):] = n
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    n0 = sol[:m0] if m0 > 0 else np.array([0.0])
    n1 = sol[m0:]
    resid = float(np.linalg.norm(M @ sol - rhs))
    if resid > 1e-6 * (1.0 + float(np.linalg.norm(n))):
        warnings.warn(f"_pf_split: inaccurate separation (resid={resid:g})")
    _num0 = Zpk.from_tf(n0, np.array([1.0]))
    _num1 = Zpk.from_tf(n1, np.array([1.0]))
    F0 = Zpk(_num0.z, np.array(d0_roots, complex), _num0.k)
    F1 = Zpk(_num1.z, np.array(d1_roots, complex), _num1.k)
    return F0, F1


def _intpoles(plant_ss: sig.StateSpace, n_meas: int = 1, n_ctrl: int = 1):
    """
    Internal poles of a standard system — port of intpoles.m (K. Polyakov):
    the poles common to all four channel blocks w→ε (P11), u→ε (P12),
    w→y (P21) and u→y (P22), i.e. controllable from both inputs and
    observable from both outputs. sdh2.m/sdahinf.m pass these to
    polquad/polhinf as PCancel — poles that MUST cancel analytically in L
    (e.g. plant integrators shared by every channel).
    """
    A, B, C, D = plant_ss.A, plant_ss.B, plant_ss.C, plant_ss.D
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin - n_ctrl
    o1 = nout - n_meas
    blocks = [
        (B[:, :i1], C[:o1, :], D[:o1, :i1]),
        (B[:, i1:], C[:o1, :], D[:o1, i1:]),
        (B[:, :i1], C[o1:, :], D[o1:, :i1]),
        (B[:, i1:], C[o1:, :], D[o1:, i1:]),
    ]
    p_int = None
    for Bb, Cb, Db in blocks:
        Am, _, _, _ = Minreal.ss(A, Bb, Cb, Db)
        poles = (np.linalg.eigvals(Am) if Am.size
                 else np.zeros(0, complex))
        if p_int is None:
            p_int = poles
        else:
            _, p_int = _others2(p_int, poles, tol=1e-6)
        if len(p_int) == 0:
            break
    return p_int if p_int is not None else np.zeros(0, complex)


def _sdh2coef(plant_ss: sig.StateSpace, T: float, t=None, H=None,
              n_meas: int = 1, n_ctrl: int = 1, return_zpk: bool = False,
              udelay: float = 0.0, refdelay: float = 0.0):
    """
    Compute A, B, E, D22 coefficients for H2 sampled-data design.

    Port of sdh2coef.m (K. Polyakov).

    ``refdelay`` is a continuous delay on a block embedded in P11 (e.g.
    MATLAB's ``Q.iodelay = preview`` in demo_h2p.m) -- unlike ``udelay``
    (control-channel, P12/P22) this can be LARGE relative to T (a preview
    horizon typically spans many sampling periods) and may be POSITIVE
    (an advance, once combined with udelay through the sign flip below).
    See the long comment at the B computation for the derivation.

    Returns (A_tf, B_tf, E_tf, D22_tf) as (num, den) tuples (z-domain);
    with ``return_zpk=True`` a fifth element (A_z, B_z, E_z) of
    zpk.zpk.Zpk root-list forms (same z-domain) is appended — the
    coefficient tuples are then derived from the snapped root lists, so
    the two representations are consistent.

    ``udelay`` is a continuous computational delay τ (0 ≤ τ ≤ T) on the
    CONTROL input (MATLAB: ``F.iodelay = τ`` on the P12/P22 column — see
    `_sdl2coef`'s identical parameter). It enters only where dtfm.m's
    delay-aware modified Z-transform is actually called: D22 = dtfm(P22)'
    and B (whose 3-stage series P21·P11H·P12 includes P12 — MATLAB's
    series-connection `tf` multiplication SUMS iodelays, so the whole
    cascade inherits P12's τ exactly the way `Px = minreal(P11H*P12)` does
    in sdl2coef.m). A0 = dtfm2(P12,T,H)/T is delay-free (dtfm2.m has no
    delay handling at all — the delay cancels identically in the
    self-adjoint Gh~·P12~·P12·Gh product). A1/E involve only P21/P11,
    neither of which carries the control-channel delay.

    ``t`` (scalar, 0 <= t < T) selects sdh2coef.m's "Instantaneous
    variance" branch (lines 85-109) instead of the default "Average
    variance" branch — the periodic error at a fixed intersample time
    offset rather than its average over one period. Only a single scalar
    t is supported (MATLAB's `for i=1:length(t)` array form, returning
    zpk ARRAYS of A/B, has no caller in this codebase) and only
    n_meas=1 (MATLAB itself hardcodes a single measurement/control
    channel, o2=i2=1, in this function). Combines with udelay: the
    control-channel delay enters exactly where it does in the average
    branch (only where P12 is used — alpha_k below — since Bs is built
    from P21/P11H alone).
    """
    A, B, C, D = plant_ss.A, plant_ss.B, plant_ss.C, plant_ss.D
    n = A.shape[0]
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin  - n_ctrl
    o1 = nout - n_meas

    B1, B2 = B[:, :i1], B[:, i1:]
    C1, C2 = C[:o1, :], C[o1:, :]
    D12 = D[:o1, i1:]
    D21 = D[o1:, :i1]
    D11 = D[:o1, :i1]
    D22_ct = D[o1:, i1:]

    # Extract minimal channel realizations to reduce polynomial degree
    A12, B12, C12, D12m = Minreal.ss(A, B2, C1, D12)
    A21, B21, C21, D21m = Minreal.ss(A, B1, C2, D21)
    A11, B11, C11, D11m = Minreal.ss(A, B1, C1, D11)
    A22, B22, C22, D22m = Minreal.ss(A, B2, C2, D22_ct)

    # D22 = dtfm(P22,T,0,H)' discretized through the generalized hold H (ZOH
    # by default), converted to d-domain (z2zeta) to match MATLAB. A
    # control-channel delay τ enters here exactly via the modified
    # Z-transform (t = −τ inside dtfm — same mechanism as _sdl2coef).
    # τ can exceed T (e.g. dsd_help.md's H2-preview example has τ=1.5, T=1)
    # — _dtfm handles any magnitude/sign of t in one call.
    _D22_num_z, _D22_den_z = _dtfm(A22, B22, C22, D22m, T, -udelay, H)
    # dtfm(P22,T,0,H)' — padded conjugate (@zpk/ctranspose.m adds origin
    # ζ-factors for the relative-degree difference); the closed-loop _ynyd
    # chain now tracks origin factors consistently.
    D22_num, D22_den = _z2zeta(_D22_num_z, _D22_den_z)

    # A1 = ztrm(minreal(P21*P21'), T, 0) — sdh2coef.m line 51 — via the
    # dimension-safe self-adjoint product, converted to a root list with
    # its poles snapped onto the exact exp(±eig(A21)·T) set (setpoles,
    # sdh2coef.m line 52).
    A1_num, A1_den = _ztrm_self_adjoint(A21, B21, C21, D21m, T)
    _p21 = np.linalg.eigvals(A21)
    A1_z = _zpk_snap(A1_num, A1_den,
                     np.exp(np.concatenate([_p21, -_p21]) * T))
    # symmetr(A1,'d') — sdh2coef.m line 53: force the zero set into exact
    # reciprocal pairs (z, 1/z). Without it the re-rooted numerator's pairs
    # drift apart (~1e-3) and can carry a spurious unpaired near-origin
    # zero; sfactor then rejects Av outright ("Exact Hermitian
    # factorization is impossible" — MATLAB's own ynyd fails on the
    # unsymmetrized export, verified via the Octave harness on demo_at96).
    try:
        A1_z = A1_z.symmetr('d')
    except ValueError as _sym_exc:
        warnings.warn(f"_sdh2coef: could not symmetrize A1 ({_sym_exc}); "
                      f"continuing with the raw root list")

    if t is None:
        # ─── Average variance (sdh2coef.m lines 57-84) ─────────────────
        # A0 = (1/T) * Z{G_h~*G_h} where G_h = H * P12, from minimal P12
        A0_num, A0_den = _dtfm2(A12, B12, C12, D12m, T, H)
        A0_num = A0_num / T
        _p12 = np.linalg.eigvals(A12)
        A0_z = _zpk_snap(A0_num, A0_den,
                         np.exp(np.concatenate([_p12, -_p12]) * T))
        # symmetr(A0,'d') — sdh2coef.m line 63 (see A1 above).
        try:
            A0_z = A0_z.symmetr('d')
        except ValueError as _sym_exc:
            warnings.warn(f"_sdh2coef: could not symmetrize A0 ({_sym_exc}); "
                          f"continuing with the raw root list")

        # A = minreals(A0*A1) — sdh2coef.m line 64 — at the ROOT-LIST level:
        # exact-match cancellation of the snapped pole/zero sets, replacing the
        # coefficient-level tf_symmetric reduction (which re-rooted the
        # high-degree product and failed to cancel smeared copies).
        _A_prod = A0_z * A1_z
        try:
            A_z = _A_prod.minreals()
        except ValueError:
            A_z = _A_prod.minreal(1e-3)
        A_num, A_den = A_z.to_tf()

        # B = dtfm(P21·P11~·P12, T, 0, H)/T — MATLAB sdh2coef.m lines 68-83:
        # an ordinary modified-Z transform (through the hold, at t=0) of the CT
        # product Px = minreal(P21*P11H*P12), with the adjoint
        # P11H = P11(-s)^T realized in state space as (-A11', C11', -B11',
        # D11') — the same series pattern _sdl2coef's B1 uses. Matrix
        # dimensions implement MATLAB's channel-sum loop (lines 74-81)
        # automatically. This replaces a home-grown cross Van-Loan integral
        # (_ztrm_cross) that dimension-crashed for plants with more than one
        # disturbance input (i1>1 — e.g. the filtering demos: process +
        # measurement noise) and SILENTLY returned B = 0, making the designed
        # "optimal" filter do nothing. Series u → P12 → P11H → P21, states
        # [x12; xH; x21]:
        n21, n11, n12 = A21.shape[0], A11.shape[0], A12.shape[0]
        try:
            A_Px = np.block([
                [A12,                    np.zeros((n12, n11)), np.zeros((n12, n21))],
                [C11.T @ C12,            -A11.T,               np.zeros((n11, n21))],
                [B21 @ (D11m.T @ C12),   -B21 @ B11.T,         A21],
            ])
            B_Px = np.vstack([B12, C11.T @ D12m, B21 @ (D11m.T @ D12m)])
            C_Px = np.hstack([D21m @ (D11m.T @ C12), -D21m @ B11.T, C21])
            D_Px = np.atleast_2d(D21m @ (D11m.T @ D12m))
            # MATLAB: Px = minreal(...) BEFORE discretization — the raw series
            # realization carries uncontrollable/unobservable modes whose
            # smeared near-copies would defeat every downstream cancellation.
            A_Px, B_Px, C_Px, D_Px = Minreal.ss(A_Px, B_Px, C_Px, D_Px)
            # P12 sits inside this series (MATLAB: Px = minreal(P21*P11H*P12),
            # series-connection tf multiplication SUMS iodelays, so the whole
            # cascade inherits P12's τ) — same _dtfm mechanism as D22
            # above and _sdl2coef's B1. `refdelay` (a delay embedded in P11,
            # e.g. preview control's Q.iodelay) enters the SAME cascade with
            # the OPPOSITE sign, because MATLAB's conjugate-transpose flips
            # ioDelay sign (@lti/ctranspose.m: `L.ioDelay = -L.ioDelay'`) —
            # P11H inherits -refdelay, so the net offset is refdelay-udelay.
            # A0/A1/E/D22 are all self-adjoint products in P11 or P12 alone
            # (never both), so refdelay cancels identically there and only
            # ever needs to enter here (verified via an independent Octave
            # re-derivation against sdh2coef.m directly).
            _t_b = refdelay - udelay
            _tInt_b = int(np.floor(_t_b / T))
            B_num, B_den = _dtfm(A_Px, B_Px, C_Px, D_Px, T, _t_b, H)
            _n_delay_states = max(-_tInt_b, 0)
            B_num = np.atleast_1d(B_num) / T
            B_den = np.atleast_1d(B_den)
            _eig_src = A_Px
            _b_targets = np.exp(np.linalg.eigvals(_eig_src) * T)
            if _n_delay_states > 0:
                _b_targets = np.concatenate([_b_targets,
                                             np.zeros(_n_delay_states, complex)])
        except Exception as _bx:
            if udelay > 0 or refdelay != 0.0:
                # The legacy cross-integral has no delay handling — silently
                # falling through would design for the wrong (undelayed) plant.
                raise ValueError(
                    f"_sdh2coef: product-form B failed ({_bx}) and the legacy "
                    f"cross-integral fallback does not support udelay/refdelay") from _bx
            # Legacy cross-integral, kept as a last resort. Only
            # valid for single-disturbance plants; raises otherwise.
            warnings.warn(f"_sdh2coef: product-form B failed ({_bx}); "
                          f"using legacy cross-integral")
            from scipy.linalg import block_diag as _blkd
            A_jt = _blkd(A21, A11, A12)
            B1_jt = np.vstack([B21, B11, np.zeros((n12, i1))])
            B2_jt = np.vstack([np.zeros((n21 + n11, n_ctrl)), B12])
            C1_jt = np.hstack([np.zeros((o1, n21)), C11, C12])
            C2_jt = np.hstack([C21, np.zeros((n_meas, n11 + n12))])
            D_jt = D.copy() if D.shape == (o1 + n_meas, i1 + n_ctrl) else np.zeros((o1 + n_meas, i1 + n_ctrl))
            D_jt[:o1, :i1] = D11m; D_jt[:o1, i1:] = D12m
            D_jt[o1:, :i1] = D21m; D_jt[o1:, i1:] = D22m
            B_num, B_den = _ztrm_cross(A_jt, B1_jt, B2_jt, C1_jt, C2_jt, D_jt, T, H, o1, i1)
            B_num = B_num / T
            _eig_src = A_jt
            _b_targets = np.exp(np.linalg.eigvals(_eig_src) * T)
        # setpoles equivalent (sdh2coef.m line 72): snap B's poles onto the
        # exactly-known discretized set exp(eig(Px)·T) (includes the mirrored
        # -A11' adjoint poles and, with a delay, exact z=0 delay-state poles),
        # then minreal at the root-list level.
        B_z = _zpk_snap(B_num, B_den, _b_targets).minreal(1e-3)
        B_num, B_den = B_z.to_tf()
    else:
        # ─── Instantaneous variance (sdh2coef.m lines 85-109) ──────────
        # A single time instant t in [0,T) (MATLAB supports an array of
        # instants via a `for i=1:length(t)` loop returning zpk ARRAYS of
        # A/B; not ported — no caller in this codebase passes more than a
        # scalar t, and MATLAB itself hardcodes o2=i2=1, i.e. a single
        # measurement/control channel, so only that scope is implemented
        # here too).
        if n_meas != 1:
            raise ValueError("_sdh2coef: t (instantaneous variance) only "
                             "supports n_meas=1 (MATLAB sdh2coef.m hardcodes "
                             "a single measurement channel, o2=1)")
        if refdelay != 0.0:
            raise NotImplementedError(
                "_sdh2coef: refdelay is only derived/validated for the "
                "average-variance branch (t=None); the instantaneous-"
                "variance Bs = P21*P11H term needs its own derivation.")
        t = float(t)
        if t >= T:
            t = T - np.sqrt(np.finfo(float).eps) * T
        _p12 = np.linalg.eigvals(A12)

        # alpha_k = dtfm(P12,T,t,H) row k (RAW z-domain, NOT z2zeta'd —
        # sdh2coef.m's A0 = z2zeta(dtfm(...)) is applied inline in MATLAB,
        # but every quantity `_sdh2coef` returns lives in the SAME raw
        # z-domain convention as the average branch above; the caller
        # (`_sdh2_pol`) applies one uniform `.conj_dt()` (=z2zeta) to A/B/E
        # together. Since conj_dt is an involution, MATLAB's ζ-domain
        # `A0'*A0` collapses to `alpha_k * conj_dt(alpha_k)` in terms of
        # this raw alpha_k — a self-adjoint (conj_dt-invariant) quantity,
        # so building A(t) from raw alpha_k needs no extra inversion.
        #
        # udelay enters exactly like dtfm.m's own delay handling
        # (`t = t - comdelay` at the top of dtfm.m/ztrm.m): P12 carries the
        # control-channel delay, so alpha_k uses t-udelay (mirroring the
        # `_dtfm(..., -udelay, H)` calls elsewhere — here it's
        # `t - udelay`, not `-udelay` alone, since t itself is already the
        # requested instant); Bs (built from P21/P11H only, no P12) carries
        # no delay, so Bx below still uses plain `t`.
        _t_a0 = t - udelay
        _n_delay_a0 = max(-int(np.floor(_t_a0 / T)), 0)
        alpha = []
        for k in range(o1):
            Ck = C12[k:k + 1, :]
            Dk = D12m[k:k + 1, :]
            _n0, _d0 = _dtfm(A12, B12, Ck, Dk, T, _t_a0, H)
            _a0_targets = np.exp(-_p12 * T)
            if _n_delay_a0 > 0:
                _a0_targets = np.concatenate([_a0_targets, [0.0]])
            alpha.append(_zpk_snap(_n0, _d0, _a0_targets))

        # A(t) = A1 * sum_k [alpha_k * conj_dt(alpha_k)]  — MATLAB's
        # A(i) = minreal(A0'*A0*A1) rewritten in the raw-alpha convention
        # above (self-adjoint, so the caller's outer conj_dt is a no-op).
        _gram = alpha[0] * alpha[0].conj_dt()
        for _ak in alpha[1:]:
            _gram = _gram.zsum(_ak * _ak.conj_dt())
        A_z = (A1_z * _gram).minreal(1e-3)
        A_num, A_den = A_z.to_tf()

        # Bs = X(-s), X = P21*P11H (2-stage series u -> P11H -> P21),
        # realized as (-Ax,-Bx,Cx,Dx) — MATLAB's Bs=(minreal(P21*P11H)').'
        # (ctranspose then plain transpose = pure s -> -s time-reversal,
        # no matrix transpose; see _sdl2coef's identical P11H adjoint
        # pattern for the (-A11',C11',-B11',D11') derivation).
        n11, n21 = A11.shape[0], A21.shape[0]
        A_X = np.block([[-A11.T,           np.zeros((n11, n21))],
                        [-B21 @ B11.T,     A21]])
        B_X = np.vstack([C11.T, B21 @ D11m.T])
        C_X = np.hstack([-D21m @ B11.T, C21])
        D_X = np.atleast_2d(D21m @ D11m.T)
        A_Bs, B_Bs, C_Bs, D_Bs = -A_X, -B_X, C_X, D_X
        _pBs = -np.linalg.eigvals(A_X)

        # B(t) = sum_k conj_dt(Bx_k) * alpha_k — MATLAB's
        # B(i) = sum_k sumzpk(B(i), Bx(k)*A0(k)) rewritten the same way:
        # A0(k) = conj_dt(alpha_k) (its ζ-domain form), and returning the
        # raw-domain value needs one conj_dt on the (raw, un-z2zeta'd)
        # Bx(k) term to match the average branch's convention exactly
        # (see the long derivation in this branch's docstring note above).
        _bterms = []
        for k in range(o1):
            Bk = B_Bs[:, k:k + 1]
            Dk = D_Bs[:, k:k + 1]
            _bn, _bd = _ztrm(A_Bs, Bk, C_Bs, Dk, T, t)
            _bx_z = _zpk_snap(_bn, _bd, np.exp(_pBs * T))
            _bterms.append(_bx_z.conj_dt() * alpha[k])
        B_z = _bterms[0]
        for _term in _bterms[1:]:
            B_z = B_z.zsum(_term)
        B_z = B_z.minreal(1e-3)
        B_num, B_den = B_z.to_tf()

    # E = ztrm(trace(P11H*P11), T, 0) — sdh2coef.m line 115 — via the
    # dimension-safe self-adjoint product; poles snapped to exp(±eig(A11)·T)
    # (setpoles, sdh2coef.m line 117).
    E_num, E_den = _ztrm_self_adjoint(A11, B11, C11, D11m, T)
    _p11 = np.linalg.eigvals(A11)
    # minreal like A/B get: the multi-row ztrm accumulation cross-multiplies
    # the rows' (identical) denominators, and the smeared re-rooted copies
    # (1±2e-7 vs the snapped exact 1s) survive coefficient-level reduction —
    # without this cancellation E's pole multiplicities are doubled, which
    # made zterm's PCancel separation degenerate (d0 and d1 shared roots).
    E_z = _zpk_snap(E_num, E_den,
                    np.exp(np.concatenate([_p11, -_p11]) * T)).minreal(1e-3)
    E_num, E_den = E_z.to_tf()

    if return_zpk:
        return ((A_num, A_den), (B_num, B_den), (E_num, E_den),
                (D22_num, D22_den), (A_z, B_z, E_z))
    return (A_num, A_den), (B_num, B_den), (E_num, E_den), (D22_num, D22_den)


def _sdl2coef(plant_ss: sig.StateSpace, T: float, H=None,
              n_meas: int = 1, n_ctrl: int = 1, return_zpk: bool = False,
              udelay: float = 0.0, refdelay: float = 0.0):
    """
    Compute A, B, E, D22 coefficients for L2 sampled-data design.

    Port of sdl2coef.m (K. Polyakov). With ``return_zpk=True`` a fifth
    element (A_z, B_z, E_z) of root-list Zpk forms is appended (ζ-domain,
    consistent with the coefficient tuples).

    ``udelay`` is a continuous computational delay τ (0 ≤ τ ≤ T) on the
    CONTROL input (MATLAB: ``F.iodelay = τ`` on the P12/P22 column). Per
    MATLAB's semantics the delay enters EXACTLY (modified Z-transform via
    `_dtfm`) and only where it survives: B1 = dtfm(P11~·P12, T, 0, H)
    and D22 = dtfm(P22, T, 0, H)'. It cancels identically in the
    self-adjoint A0 = D₂{Gh~·P12~·P12·Gh} (dtfm2.m has no delay handling
    at all for this reason) and is absent from A1/E (P21/P11 rows).
    The plant must be passed WITHOUT the delay baked in.

    ``refdelay`` is the continuous preview horizon π (MATLAB: preview
    control, demo_l2p.m) -- a GENUINELY DIFFERENT delay topology from
    sdh2's refdelay, because sdl2coef.m's A1 = ztrm(P21, T, 0) uses P21
    ALONE (not the self-adjoint P21·P21' sdh2coef.m uses), so a delay on
    P21 does NOT cancel here. MATLAB's demo_l2p.m splits π into an integer
    sample count and remainder: σ = ceil(π/T), θ = σ·T − π ∈ [0, T), then
    sets BOTH `Q.iodelay = π` (the ideal operator, inside P11 = Q·R) AND
    `R.iodelay = θ` (the reference generator R, appearing separately in
    P21 = R). The result: P21's own delay is the bounded θ (enters A1 via
    `_ztrm`), while P11's combined delay from both Q and R is the
    exact integer σ·T (enters B1's P11~·P12 cascade via `_dtfm`,
    P11~ = P11' picking up -σ·T from
    the conjugate-transpose sign flip, same as sdh2's refdelay). Combines
    additively with `udelay` wherever both land in the same cascade
    (P12/P22, and the D22/B1 delay-sum). Average-variance-only (sdl2 has
    no `t` instantaneous-variance branch to guard against). Validated
    against `dsd_help.md`'s documented L2-preview example.
    """
    A, B, C, D = plant_ss.A, plant_ss.B, plant_ss.C, plant_ss.D
    n = A.shape[0]
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin  - n_ctrl
    o1 = nout - n_meas

    B1, B2 = B[:, :i1], B[:, i1:]
    C1, C2 = C[:o1, :], C[o1:, :]
    D12 = D[:o1, i1:]
    D21 = D[o1:, :i1]
    D11 = D[:o1, :i1]
    D22_ct = D[o1:, i1:]

    # Extract minimal channel realizations to reduce polynomial degree
    A12, B12, C12, D12m = Minreal.ss(A, B2, C1, D12)
    A21, B21, C21, D21m = Minreal.ss(A, B1, C2, D21)
    A11, B11, C11, D11m = Minreal.ss(A, B1, C1, D11)
    A22, B22, C22, D22m = Minreal.ss(A, B2, C2, D22_ct)

    # D22 = dtfm(P22,T,0,H)' discretized through the generalized hold H,
    # converted to d-domain (z2zeta). A control-channel delay τ enters here
    # exactly via the modified Z-transform (t = −τ inside dtfm). τ can
    # exceed T (e.g. dsd_help.md's L2-preview example has τ=1.5, T=1) --
    # _dtfm handles any magnitude/sign of t in one call.
    _D22_num_z, _D22_den_z = _dtfm(A22, B22, C22, D22m, T, -udelay, H)
    # dtfm(P22,T,0,H)' — padded conjugate (@zpk/ctranspose.m adds origin
    # ζ-factors for the relative-degree difference); the closed-loop _ynyd
    # chain now tracks origin factors consistently.
    D22_num, D22_den = _z2zeta(_D22_num_z, _D22_den_z)

    # A0 = dtfm2(P12, T, H) — two-sided bilateral Z-transform of Gh~*Gh
    # (udelay cancels identically here — see docstring).
    A0_num, A0_den = _dtfm2(A12, B12, C12, D12m, T, H)

    # A1_z = ztrm(P21, T, 0) — one-sided Z-transform of P21
    # A1   = z2zeta(A1_z)    — map to zeta-domain via the PADDED conjugate
    # (_z2zeta), not naive per-array [::-1] reversal: MATLAB's z2zeta.m
    # (lines 46-49) adds origin zeros/poles for any relative-degree
    # difference; unpadded reversal silently drops those ζ-factors
    # (the same class of bug recurs throughout this module).
    # refdelay (preview horizon π) splits into an integer sample count σ
    # and remainder θ = σ·T − π ∈ [0, T) — MATLAB's demo_l2p.m convention.
    # σ enters P11's combined (Q and R both carry delay) B1 cascade below;
    # θ enters A1 = ztrm(P21, T, 0) here (P21 = R alone, θ only) since
    # _ztrm's Cd = C·expm(A·t) formula has no z^k decomposition and
    # is valid for any real t directly (see refdelay docstring above).
    # theta=0 when refdelay=0 (sigma=0 too), so this is one call either way.
    _sigma = int(np.ceil(refdelay / T)) if refdelay > 0 else 0
    _theta = _sigma * T - refdelay
    a1_z_num, a1_z_den = _ztrm(A21, B21, C21, D21m, T, -_theta)
    a1_num, a1_den = _z2zeta(a1_z_num, a1_z_den)
    # Root-list forms with setpoles-snapped poles (exp(eig·T) exactly):
    _p21_l2 = np.linalg.eigvals(A21)
    a1z_zz = _zpk_snap(a1_z_num, a1_z_den, np.exp(_p21_l2 * T))
    a1_zz = a1z_zz.conj_dt()
    _p12_l2 = np.linalg.eigvals(A12)
    A0_zz = _zpk_snap(A0_num, A0_den,
                      np.exp(np.concatenate([_p12_l2, -_p12_l2]) * T))

    # A = A0 * A1 * A1'  (MATLAB: A = minreal(A0 * A1 * A1') — plain minreal,
    # not minreals, despite this product being symmetric in principle; match
    # MATLAB's actual call faithfully rather than "improving" it).
    # A1' = conjugate-reciprocal of A1 = the un-z2zeta version = a1_z_num/a1_z_den
    # Root-list product + minreal (exact-match cancellation of the
    # snapped pole sets) — replaces the coefficient-level reduction.
    A_zz = (A0_zz * a1_zz * a1z_zz).minreal(1e-3)
    A_num, A_den = A_zz.to_tf()

    # B = A1 * B1  where B1 = z2zeta(dtfm(P11~*P12, T, 0, H))
    # MATLAB: Px = minreal(P11H*P12); B1 = z2zeta(dtfm(Px, T, 0, H)); B = A1*B1*scale11*scale12
    #
    # MATLAB's P11H = P11' = P11(-s)^T, state-space: (-A11^T, C11^T, -B11^T, D11m^T)
    # MATLAB's P11H*P12 means: input -> P12 -> P11H -> output (matrix product order)
    # Series: P12 (n_ctrl -> o1) followed by P11H (o1 -> i1)
    n11, n12 = A11.shape[0], A12.shape[0]
    A_Px = np.block([[A12,           np.zeros((n12, n11))],
                     [C11.T @ C12,  -A11.T]])
    B_Px = np.vstack([B12, C11.T @ D12m])
    C_Px = np.hstack([D11m.T @ C12, -B11.T])
    D_Px = np.atleast_2d(D11m.T @ D12m)
    # MATLAB sdl2coef.m:78 "Px = minreal(P11H*P12)" BEFORE discretization —
    # the raw series realization carries uncontrollable/unobservable modes
    # (same issue sdh2coef.m's analogous Px construction already guards
    # against, see _sdh2coef). Without this, an unreduced mode surfaces as
    # a near-duplicate large pole/zero pair (e.g. ~19180 vs ~19733 for the
    # dsd_help.md L2-preview example) that fails to cancel in the z-domain
    # after ctranspose, leaving a spurious near-z=0 factor in the final K.
    A_Px, B_Px, C_Px, D_Px = Minreal.ss(A_Px, B_Px, C_Px, D_Px)

    # dtfm(Px, T, 0, H) — with a control-channel delay τ this is the
    # modified Z-transform at t = σ·T − τ (P11~ = P11' picks up -σ·T from
    # refdelay's Q/R combination via the conjugate-transpose sign flip,
    # exactly like sdh2's refdelay; P12 contributes -τ as always).
    _t_b1 = _sigma * T - udelay
    _tInt_b1 = int(np.floor(_t_b1 / T))
    px_num, px_den = _dtfm(A_Px, B_Px, C_Px, D_Px, T, _t_b1, H)
    _n_delay_states = max(-_tInt_b1, 0)
    # z2zeta via padded conjugate (see A1 above — the same unpadded-reversal
    # pitfall applies here)
    b1_num, b1_den = _z2zeta(px_num, px_den)

    # B = A1 * B1 — at the root-list level (b1's poles snapped to the
    # exactly-known exp(eig(Px)·T) set before the ζ-conversion; the delay
    # augmentation adds exact z=0 poles).
    _pPx_l2 = np.linalg.eigvals(A_Px)
    _b1_targets = np.exp(_pPx_l2 * T)
    if _n_delay_states > 0:
        _b1_targets = np.concatenate([_b1_targets,
                                      np.zeros(_n_delay_states, complex)])
    b1_zz = _zpk_snap(px_num, px_den, _b1_targets).conj_dt()
    # MATLAB sdl2coef.m:92 "B = minreal(A1*B1*scale11*scale12)" passes NO
    # explicit tolerance (unlike the several other minreal(1e-3) calls in
    # this module that DO match an explicit-tolerance MATLAB call) -- so
    # this uses Zpk.minreal's default (sqrt(eps)) tolerance to match.
    # A loose 1e-3 tolerance here is a genuine correctness bug, not just a
    # style deviation: _reducezp's cancellation test is `dmin < tol*(1+|z|)`,
    # an ABSOLUTE floor of `tol` for any root near the origin. a1_zz always
    # carries a legitimate exact zero at 0 (the relative-degree-preserving
    # pad from A1's conjugate transpose, @zpk/ctranspose.m). With tol=1e-3
    # that zero-at-0 spuriously "cancels" against ANY genuinely small but
    # nonzero pole in b1_zz (e.g. a fast plant/Q pole sampled over T=1,
    # exp(-10)=4.54e-5 in dsd_help.md's L2-preview example) even though the
    # two are numerically ~20x apart and unrelated -- leaving an uncancelled
    # near-duplicate root pair elsewhere in the design (found validating
    # against dsd_help.md's documented sdl2err=0.0517 preview example).
    B_zz = (a1_zz * b1_zz).minreal()
    B_num, B_den = B_zz.to_tf()

    # E = ztrm(SP11, T, 0) where SP11 = P11H * P11 (series: P11 then P11H adjoint).
    # Port of MATLAB sdl2coef.m: E = ztrm(trace(P11H*P11), T, 0); FE=sfactor(E); E=FE*FE'
    # The one-sided ztrm already produces a palindromic E = FE*FE'.
    n11 = A11.shape[0]
    A_SP11 = np.block([[A11,           np.zeros((n11, n11))],
                       [C11.T @ C11,   -A11.T]])
    B_SP11 = np.vstack([B11, C11.T @ D11m])
    C_SP11 = np.hstack([D11m.T @ C11, -B11.T])
    D_SP11 = np.atleast_2d(D11m.T @ D11m)
    E_num, E_den = _ztrm(A_SP11, B_SP11, C_SP11, D_SP11, T)
    # The one-sided ztrm of strictly proper SP11 gives E_num with z^0 ≈ 0.
    # Clamp it to exactly 0 so Z_num in _polhinf is a proper shifted palindrome
    # (z * palindromic), matching BB_num which is also z * palindromic.
    if len(E_num) > 1:
        _e_tol = 1e-8 * np.max(np.abs(E_num))
        if np.abs(E_num[-1]) < _e_tol:
            E_num = E_num.copy()
            E_num[-1] = 0.0
    _p11_l2 = np.linalg.eigvals(A11)
    E_zz = _zpk_snap(E_num, E_den,
                     np.exp(np.concatenate([_p11_l2, -_p11_l2]) * T))
    E_num, E_den = E_zz.to_tf()

    if return_zpk:
        return ((A_num, A_den), (B_num, B_den), (E_num, E_den),
                (D22_num, D22_den), (A_zz, B_zz, E_zz))
    return (A_num, A_den), (B_num, B_den), (E_num, E_den), (D22_num, D22_den)


def _sd2dofcoef(plant_ss: sig.StateSpace, T: float, H=None,
                udelay: float = 0.0, refdelay: float = 0.0):
    """
    Compute A, B, E, kCoef, n, d coefficients for 2-DOF sampled-data design.

    Port of sd2dofcoef.m (K. Polyakov). `plant_ss` has 3 output rows
    [z; y1; y2] (1 performance + 2 measurement) and 2 input columns [d; u]
    (1 disturbance + 1 control), matching the standard 2-DOF structure::

        [z ]   [P11  P12] [d]
        [y1] = [P21   0 ] [u]
        [y2]   [P210 P22]

    ``udelay``/``refdelay`` mirror `_sdl2coef`'s parameters of the same name
    (see there for the full derivation) and enter in TWO places, exactly
    matching `sd2dofcoef.m`'s own structure:

    1. The `_sdl2coef` call on the `[z;y1]` sub-plant below (P11/P12/P21's
       own delays — `sd2dofcoef.m:54`'s `[A,B,E,A0,A1] = sdl2coef(sys2,T,H)`
       is a PLAIN, undelayed call in MATLAB because MATLAB bakes `udelay`/
       `refdelay` into `sys2`'s blocks via `.iodelay`; here the delay-free
       plant plus explicit `udelay`/`refdelay` args reproduces that exactly,
       just as `sdl2` itself does).
    2. This function's OWN `D22 = dtfm(P22,T,0,H)'` (`sd2dofcoef.m:48`,
       distinct from `_sdl2coef`'s internal D22, which is always 0 here
       since the `[z;y1]` sub-plant's `y1` row has no `u`-column term) —
       P22 is the plant `F` alone, so only `udelay` (F's own control-channel
       delay) applies; `refdelay` (Q/R's preview delay) never reaches P22.

    Returns
    -------
    A_tf, B_tf, E_tf : (num, den) tuples — L2 coefficients on the [z;y1]
        sub-plant (as `_sdl2coef` computes), rescaled by dzpk*dzpk~ / dzpk
        where dzpk is built from D22's own denominator (`d` below).
    kCoef : float
        Static gain such that P210 = kCoef * P21 (MATLAB requires this ratio
        to be a constant; raises if it isn't).
    n, d : ndarray
        Numerator/denominator of D22 = dtfm(P22,T,0,H)' (ζ-domain).
    """
    A, B, C, D = plant_ss.A, plant_ss.B, plant_ss.C, plant_ss.D
    nout, nin = C.shape[0], B.shape[1]
    i1 = nin - 1
    o1 = nout - 2

    B_d, B_u = B[:, :i1], B[:, i1:]
    C_y1, C_y2 = C[o1:o1 + 1, :], C[o1 + 1:o1 + 2, :]
    D_y1_u = D[o1:o1 + 1, i1:]
    D_y1_d = D[o1:o1 + 1, :i1]
    D_y2_d = D[o1 + 1:o1 + 2, :i1]
    D_y2_u = D[o1 + 1:o1 + 2, i1:]

    # --- Check P221 = (y1,u channel) ≈ 0 (MATLAB: error if not) ---
    A221, B221, C221, D221 = Minreal.ss(A, B_u, C_y1, D_y1_u)
    if A221.shape[0] > 0 or np.max(np.abs(D221)) > 1e-8:
        raise ValueError("sd2dof: incorrect 2-DOF structure (P221 != 0)")

    # --- kCoef = P212/P211 = P210/P21 (must be a static gain) ---
    A212, B212, C212, D212 = Minreal.ss(A, B_d, C_y2, D_y2_d)
    A211, B211, C211, D211 = Minreal.ss(A, B_d, C_y1, D_y1_d)
    if A212.shape[0] == 0:
        # P212 identically zero -> kCoef = 0 regardless of P211.
        kCoef = 0.0
    else:
        p212_num, p212_den = _ss2tf_robust(sig.StateSpace(A212, B212, C212, D212))
        p211_num, p211_den = _ss2tf_robust(sig.StateSpace(A211, B211, C211, D211))
        ratio_num = np.polymul(p212_num, p211_den)
        ratio_den = np.polymul(p211_num, p212_den)
        try:
            ratio_num, ratio_den = Minreal.tf(ratio_num, ratio_den, tol=1e-6)
        except Exception:
            pass
        if len(_strip_lz(ratio_num)) > 1 or len(_strip_lz(ratio_den)) > 1:
            raise ValueError("sd2dof: P210 must equal a constant times P21 in this version")
        kCoef = float(np.real(ratio_num[0]) / np.real(ratio_den[0]))

    # --- D22 = dtfm(P22,T,0,H)' (ζ-domain, matching _sdl2coef's own D22) ---
    # udelay enters exactly like _sdl2coef's own D22 computation (same
    # _dtfm call on t=-udelay); refdelay never reaches P22 (see docstring).
    A22, B22, C22, D22 = Minreal.ss(A, B_u, C_y2, D_y2_u)
    _n_z, _d_z = _dtfm(A22, B22, C22, D22, T, -udelay, H)
    # z→ζ conjugate via _z2zeta, NOT naive per-array [::-1] reversal:
    # for a strictly proper D22 (num shorter than den), the conjugate H(1/z)
    # gains a zero at the origin per unit of relative degree (MATLAB
    # @zpk/ctranspose.m lines 64-66: `zpow = ...; z{j}=[zj; zeros(zpow,1)]`).
    # Naive unpadded reversal silently drops that ζ-factor, corrupting
    # Delta = a*n + b*d downstream.
    n, d = _z2zeta(_n_z, _d_z)

    # --- A, B, E from _sdl2coef on the [z; y1] sub-plant ---
    C_sub = np.vstack([C[:o1, :], C_y1])
    D_sub = np.vstack([D[:o1, :], np.hstack([D_y1_d, D_y1_u])])
    sys2_ss = sig.StateSpace(A, B, C_sub, D_sub)
    A_tf, B_tf, E_tf, _, (A_zz, B_zz, _E_zz) = _sdl2coef(
        sys2_ss, T, H, n_meas=1, n_ctrl=1, return_zpk=True,
        udelay=udelay, refdelay=refdelay)

    # --- Rescale: A = minreal(A*dzpk*dzpk'), B = minreal(B*dzpk) ---
    # At the ROOT-LIST level: dzpk' (conj_dt) carries the padded-conjugate
    # origin poles exactly, and the snapped d-roots cancel exactly against
    # A/B's poles. The previous coefficient-level polymul + Minreal.tf
    # re-rooted a degree-9 product and smeared the ζ=1 integrator multiple
    # into {1, 1.00123} — sending polquad's Λ = sfactor(A) into a silent
    # degenerate constant (MATLAB's own sfactor ERRORS on that smeared A:
    # 'Exact Hermitian factorization is impossible' — verified via an
    # independent Octave re-derivation). This rescale is now exact.
    #
    # MATLAB sd2dofcoef.m:55-57 ("A = minreal(A*dzpk*dzpk'); B =
    # minreal(B*dzpk)") passes NO explicit tolerance -- unlike the several
    # OTHER minreal(1e-3) calls in this module that DO match an explicit-
    # tolerance MATLAB call. Using Zpk.minreal()'s default (sqrt(eps)) here
    # instead of a hardcoded 1e-3 for the same reason `_sdl2coef`'s B_zz
    # combination needed it: `_reducezp`'s cancellation test has an
    # ABSOLUTE floor of `tol` for roots near the origin, which a loose
    # 1e-3 can spuriously trigger against a genuinely small (but unrelated)
    # plant pole once delays are involved (found while validating sdl2's
    # preview control; the same class of bug, so fixed here proactively
    # rather than waiting to hit it).
    d_stripped = _strip_lz(np.asarray(d, float))
    d_z = Zpk.from_tf(d_stripped, np.array([1.0]))
    A_z2 = (A_zz * d_z * d_z.conj_dt()).minreal()
    B_z2 = (B_zz * d_z).minreal()
    A_num, A_den = A_z2.to_tf()
    B_num, B_den = B_z2.to_tf()

    return (A_num, A_den), (B_num, B_den), E_tf, kCoef, n, d


# ---------------------------------------------------------------------------
# Matrix-integral helpers (port of intaba.m and dtfm2.m)
# ---------------------------------------------------------------------------

def _intaba(A1, B_mat, A2, t):
    """
    Compute X = Phi1 * ∫_0^t Phi1^{-1}(v) * B * Phi2(v) dv
              = upper-right block of expm([[A1, B]; [0, A2]] * t).

    Port of intaba.m (K. Polyakov).
    """
    n1 = A1.shape[0]
    n2 = A2.shape[0]
    Z = np.block([[A1, B_mat], [np.zeros((n2, n1)), A2]])
    EZ = la.expm(Z * t)
    return EZ[:n1, n1:]


def dtfm2(F, T, H=None):
    """
    Discrete two-sided transform D{Gh~·F~·F·Gh}(T).

    Port of dtfm2.m (K. Polyakov).

    Computes the z-transform of the sampled two-sided autocorrelation of
    the hold-weighted plant F, evaluated at shift 0.

    Parameters
    ----------
    F : scipy.signal.StateSpace or (A, B, C, D) tuple
        Proper continuous-time plant.
    T : float
        Sampling period.
    H : scipy.signal.StateSpace or (A, B, C, D) tuple, optional
        Hold device (default: scalar ZOH).

    Returns
    -------
    num, den : np.ndarray
        Numerator and denominator of the discrete TF (z-domain, highest-first).
    """
    if isinstance(F, tuple) and len(F) == 4:
        A_m, B_m, C_m, D_m = [np.atleast_2d(np.array(x, float)) for x in F]
        F_ss = sig.StateSpace(A_m, B_m, C_m, D_m)
    else:
        F_ss = F if isinstance(F, sig.StateSpace) else F.to_ss()
    return _dtfm2(F_ss.A, F_ss.B, F_ss.C, F_ss.D, T, H)


def _dtfm(A, B, C, D, T, t, H=None):
    """
    Discrete transform D{F·H}(T, t) at an arbitrary time offset t — port of
    dtfm.m's BASIC ALGORITHM (lines 148-215), the modified Z-transform.
    ONE function for any t, matching dtfm.m's own single-function structure
    (unlike dtfm.m's `dtfm2.m` sibling, a genuinely different two-sided/
    bilateral transform — see `_dtfm2` — this is a single MATLAB function
    that was split into three Python pieces across separate sessions before
    being merged back here: `_dtfm_t0` (t=0 shortcut),
    `_dtfm_frac` (|t| within one period, state-space return), and
    `_dtfm_frac_wide` (any t, coefficient return). Verified numerically
    that all three were computing the exact same formula — the split only
    ever existed because an ADVANCE (t > 0, i.e. preview control) is
    inherently improper and has no causal state-space realization, forcing
    a coefficient return type for that case; every caller of the old
    state-space-returning variant immediately converted to (num, den)
    anyway via `_ss2tf_robust`, so that return type was never load-bearing).

    This is how MATLAB handles a continuous ioDelay τ EXACTLY: dtfm.m folds
    the delay into the time argument (t → t − τ), so a delayed plant at
    t=0 becomes t = −τ here — no Padé approximation, no fast poles. (Padé
    substitutes poles at ~±600 for τ=0.01, whose discretized ζ-images
    exp(±λT) span ~20 decades and destroy every coefficient-level
    polynomial downstream — the root cause of a real defect this port
    guards against.)

        tInt = floor(t/T);  m = t − tInt·T
        [Gammam, Phi1m, Phi2m] = intaba(A, B·K, L, m)
        [Gamma,  Phi]          = intaba(A, B·K, L, T)
        Ad = Phi;  Bd = Gamma·M;  Cd = C·Phi1m
        Dd = (C·Gammam + D·K·Phi2m)·M
        then multiply by z^tInt: tInt > 0 (an ADVANCE — preview control)
        appends tInt zeros to the NUMERATOR (dtfm.m: `add =
        zpk(zeros(1,zNum),[],1); D = D*add`), improper by construction;
        tInt < 0 (a DELAY) appends |tInt| zeros to the DENOMINATOR
        (dtfm.m: `D.ioDelay = zDen; delay2z(D)`, an extra z^zDen factor).
        At t=0 exactly, m=0 makes Gammam/Phi1m/Phi2m trivially 0/I/I —
        computed directly here rather than as a special case; numerically
        identical (verified) to running the general formula through.

    Returns (num, den) as descending-order coefficient arrays. num may be
    of higher degree than den for tInt > 0 (an intermediate coefficient
    feeding a downstream Diophantine/spectral-factorization stage, which
    doesn't require properness — this never needs to be a realizable
    causal system on its own).
    """
    n = A.shape[0]
    mi = B.shape[1]
    if H is None:
        L_h = np.zeros((1, 1))
        M_h = np.ones((1, mi))
        K_h = np.ones((1, 1))
    else:
        H_ss = H if isinstance(H, sig.StateSpace) else H.to_ss()
        L_h, M_h, K_h = H_ss.A, H_ss.B, H_ss.C

    tInt = int(np.floor(t / T))
    m = t - tInt * T

    Gammam = _intaba(A, B @ K_h, L_h, m)
    Phi1m = la.expm(A * m)
    Phi2m = la.expm(L_h * m)
    Gamma = _intaba(A, B @ K_h, L_h, T)
    Phi = la.expm(A * T)

    Ad = Phi
    Bd = Gamma @ M_h
    Cd = C @ Phi1m
    Dd = (C @ Gammam + (D @ K_h) @ Phi2m) @ M_h

    num, den = _ss2tf_robust(sig.StateSpace(Ad, Bd, Cd, Dd, dt=T))
    if tInt > 0:
        num = np.concatenate([num, np.zeros(tInt)])
    elif tInt < 0:
        den = np.concatenate([den, np.zeros(-tInt)])
    return num, den


def _dtfm2(A, B2, C1, D12, T, H=None):
    """
    Discrete two-sided transform D{G_h~ G_h}(T).

    Computes the z-transform of the sampled autocorrelation of the
    hold-weighted transfer matrix G_h = H * P12, evaluated at t=0.

    For ZOH (H(s) = 1/s):  G_h(s) = P12(s)/s

    Port of dtfm2.m (K. Polyakov).
    Returns (num, den) discrete TF as np.ndarray.
    """
    n = A.shape[0]
    m = B2.shape[1]   # number of control inputs
    o1 = C1.shape[0]  # number of performance outputs

    if H is None:
        # Default ZOH: state-space representation H = ss(0, 1, 1, 0)
        L_h = np.zeros((1, 1))
        M_h = np.ones((1, m))
        K_h = np.ones((1, 1))
    else:
        H_ss = H if isinstance(H, sig.StateSpace) else H.to_ss()
        L_h  = H_ss.A
        M_h  = H_ss.B
        K_h  = H_ss.C

    rH = K_h.shape[0]  # hold output dimension

    # Augmented system matrices
    Av = np.block([[A, B2 @ K_h],
                   [np.zeros((rH, n)), L_h]])
    CDK = np.hstack([C1, D12 @ K_h])  # o1 × (n + rH)

    # G = Phi_Av^{-1} * ∫ exp(-Av*v) * (CDK'*CDK) * exp(-Av'*v) dv * (?)
    # Using intaba trick: G = Phi' \ intaba(-Av', CDK'*CDK, Av, T)
    G_raw = _intaba(-Av.T, CDK.T @ CDK, Av, T)
    Phi_T = la.expm(Av * T)
    try:
        G = la.solve(la.expm(-Av.T * T), G_raw)
    except np.linalg.LinAlgError:
        G = G_raw

    g11 = G[:n, :n]
    g12 = G[:n, n:]
    g21 = G[n:, :n]
    g22 = G[n:, n:]

    # Gamma = expm(A*T) * int[0..T] exp(-A*v) * B2*K_h * exp(L_h*v) dv  (n × rH)
    Gamma = _intaba(A, B2 @ K_h, L_h, T)
    # Phi = expm(A*T) and Theta = inv(Phi') = expm(-A^T * T)  (n × n)
    Phi = la.expm(A * T)
    Theta = la.solve(Phi.T, np.eye(n))  # = expm(-A.T * T)

    Ad = np.block([[Theta,           Theta @ Theta @ g11],
                   [np.zeros((n, n)), Phi]])
    Bd = np.vstack([Theta @ Theta @ g12, Gamma]) @ M_h
    Cd = M_h.T @ np.hstack([-Gamma.T, g21 - Gamma.T @ Theta @ g11])
    Dd = M_h.T @ (g22 - Gamma.T @ Theta @ g12) @ M_h

    try:
        dt_ss = sig.StateSpace(Ad, Bd, Cd, Dd, dt=T)
        tf_out = dt_ss.to_tf()
        if hasattr(tf_out.num, '__len__') and hasattr(tf_out.num[0], '__len__'):
            return tf_out.num[0][0], tf_out.den[0][0]
        return tf_out.num, tf_out.den
    except Exception:
        # Fallback: return a scalar
        return np.array([float(np.real(Dd.flat[0]))]), np.array([1.0])


def _ztrm(A, B, C, D, T, t=0.0, snap=True):
    """
    One-sided modified Z-transform D{F}(T,t) at an arbitrary time offset
    0 <= t < T. Port of ztrm.m's BASIC ALGORITHM (lines 84-146). ONE
    function for any t, matching ztrm.m's own single-function structure
    (default t=0.0 covers A1/E's usual call, `ztrm(P21,T,0)` etc.).

    Merged from two Python pieces (`_ztrm_onesided`: t=0 only,
    `_ztrm_frac`: arbitrary t) that had accreted a real precision gap, not
    just a style split (unlike the analogous `_dtfm_t0`/`_dtfm_frac`/
    `_dtfm_frac_wide` merge the same day, which was pure duplication):
    ztrm.m ALWAYS applies the "Mandatory zero at z=0" correction below
    (lines 129-146), unconditionally for ANY t including t=0 -- but
    `_ztrm_onesided` never applied it at all. For a single, non-self-adjoint
    block (sdl2's A1=ztrm(P21,T,0), sdh2's instantaneous-variance Bs term)
    this is a real, if tiny, missing exactness fix: verified on the
    validated L2-preview example's SP11 block that the smallest-magnitude
    root was already ~1e-9 (not exactly 0) before the fix, silently masked
    until now by every test's tolerance being far looser than 1e-9.

    ``snap=False`` (used ONLY by `_ztrm_self_adjoint`, sdh2's
    A1=ztrm(P21*P21',T,0) and E=ztrm(SP11,T,0) SUMMED-over-rows case) skips
    this correction. Found the hard way (a real regression on
    the sdh2 preview test, 1-DOF cost 1.42 instead of 11.67): a SUM of
    several rows' self-adjoint terms is inherently PALINDROMIC (symmetric
    numerator coefficients, e.g. `[0.04, -0.122, 0.04]`), and blindly
    snapping the smallest-magnitude root to exactly 0 destroys that
    symmetry (`[0.04, -0.107, 0.0]`) rather than just cleaning up noise --
    MATLAB's own `sdh2coef.m:53` calls `symmetr(A1,'d')` right after `ztrm`
    to force EXACT palindrome symmetry regardless of what `ztrm` produced,
    but Python's `Zpk.symmetr` port isn't robust to the exact-zero root the
    snap introduces here (a separate, narrower fragility in `symmetr`
    itself, not evidence the snap is wrong for the non-summed callers --
    left as `snap=False` here rather than risk fixing `symmetr` under time
    pressure).

        Ad = expm(A*T);  Bd = Ad*B
        Cd = C*expm(A*t);  Dd = Cd*B + D
        then (if snap) force the numerator's SMALLEST-MAGNITUDE root to
        exactly z=0 -- an analytic property of the modified Z-transform
        (the extra one-step delay/hold structure guarantees at least one
        zero at the origin for a single structured P21/SP11-row/Bs input)
        that plain rootfinding only reproduces approximately; MATLAB
        overwrites it exactly rather than relying on that approximation,
        so this does the same.

    Returns (num, den) z-domain polynomial TF, highest-degree first.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    C = np.atleast_2d(np.asarray(C, float))
    D = np.atleast_2d(np.asarray(D, float))
    Phi = la.expm(A * T)
    Ad = Phi
    Bd = Phi @ B
    Cd = C @ la.expm(A * t)
    Dd = np.atleast_2d(Cd @ B + D)
    num, den = _ss2tf_robust(sig.StateSpace(Ad, Bd, Cd, Dd, dt=T))
    num = np.atleast_1d(np.asarray(num, float))
    if not snap:
        return np.real(num), den
    stripped = _strip_lz(num)
    if len(stripped) > 0 and np.max(np.abs(stripped)) > 1e-300:
        roots = np.roots(stripped)
        if len(roots) > 0:
            k0 = int(np.argmin(np.abs(roots)))
            roots[k0] = 0.0
            rebuilt = np.real_if_close(stripped[0] * np.poly(roots))
            pad = len(num) - len(rebuilt)
            num = np.concatenate([np.zeros(pad), rebuilt]) if pad > 0 else rebuilt
    return np.real(num), den


def _ztrm_self_adjoint(A, B, C, D, T):
    """
    ztrm(trace(P~·P))(T, 0) = ztrm(Σ_i P_i·P_i~)(T, 0), where P_i is the
    i-th output row of P = (A, B, C, D).

    Used for MATLAB sdh2coef.m's A1 = ztrm(minreal(P21*P21'), T, 0) (line
    51) and E = ztrm(trace(P11H*P11), T, 0) (line 115). Each row's
    self-adjoint product is realized as the SISO series u → P_i~ → P_i
    with the adjoint P_i~ = (-A', C_i', -B', D_i') in state space —
    dimension-safe for ANY number of inputs, unlike the previous
    _ztrm_autocorr, whose construction raised internally for plants with
    more than one disturbance input (i1 > 1, e.g. process + measurement
    noise in the filtering demos) and SILENTLY returned the constant 1 —
    handing the design a degenerate A/E with no signal anything was wrong.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    C = np.atleast_2d(np.asarray(C, float))
    D = np.atleast_2d(np.asarray(D, float))
    n = A.shape[0]

    # Realize the whole SUM Σ_i P_i~·P_i as ONE parallel state-space system
    # and take a single one-sided ztrm — the equivalent of MATLAB computing
    # trace(P~·P) first and discretizing once (sdh2coef.m line 115). The
    # previous per-row loop accumulated over a common denominator with
    # den_acc = den_i·den_acc, DOUBLING every shared pole's multiplicity;
    # the coefficient-level Minreal.tf(1e-6) then failed to cancel the
    # smeared re-rooted copies, so E reached _ynyd with e.g. a (ζ-1)⁴
    # cluster where the analytic function has (ζ-1)² (demo_autom97,
    # demo_doubint — MATLAB's own polquad.m rejects the exported result).
    # Minreal.ss dedups the parallel modes EXACTLY (state-space rank tests,
    # no polynomial rooting involved).
    blocks = []
    for i in range(C.shape[0]):
        Ci = C[i:i + 1, :]
        Di = D[i:i + 1, :]
        # series: u → P_i~ = (-A', Ci', -B', Di') → v (m-dim) → P_i
        # states [x_adj; x]:
        A_pp = np.block([[-A.T,     np.zeros((n, n))],
                         [-B @ B.T, A]])
        B_pp = np.vstack([Ci.T, B @ Di.T])
        C_pp = np.hstack([-Di @ B.T, Ci])
        D_pp = np.atleast_2d(Di @ Di.T)
        blocks.append((A_pp, B_pp, C_pp, D_pp))

    if len(blocks) == 1:
        A_s, B_s, C_s, D_s = blocks[0]
    else:
        from scipy.linalg import block_diag as _blkd
        A_s = _blkd(*[b[0] for b in blocks])
        B_s = np.vstack([b[1] for b in blocks])
        C_s = np.hstack([b[2] for b in blocks])
        D_s = np.atleast_2d(sum(float(np.atleast_2d(b[3])[0, 0]) for b in blocks))
        A_s, B_s, C_s, D_s = Minreal.ss(A_s, B_s, C_s, D_s)

    num_acc, den_acc = _ztrm(A_s, B_s, C_s, D_s, T, snap=False)
    return np.atleast_1d(num_acc), np.atleast_1d(den_acc)


def _ztrm_autocorr(A, B, C, D, T):
    """
    Bilateral Z-transform of sampled autocorrelation: D{P~ * P}(T).

    Same framework as _dtfm2 but with no hold augmentation. The denominator
    is palindromic (d * d_cr) so that polquad's d^2*d_cr^2 * A_num/A_den
    correctly reduces to a polynomial.
    """
    try:
        n = A.shape[0]
        m = B.shape[1]

        # Core bilateral integral (no hold → Av = A, CDK = C)
        G_raw = _intaba(-A.T, C.T @ C, A, T)
        try:
            G = la.solve(la.expm(-A.T * T), G_raw)   # n × n
        except np.linalg.LinAlgError:
            G = G_raw

        Phi   = la.expm(A * T)
        Theta = la.solve(Phi.T, np.eye(n))            # Phi^{-T}
        Gamma = _intaba(A, B, np.zeros((m, m)), T)[:n, :]  # ∫_0^T e^{At} B dt  (n×m)

        # 2n-dimensional bilateral state-space
        Ad = np.block([[Theta,            Theta @ Theta @ G],
                       [np.zeros((n, n)), Phi]])
        Bd = np.vstack([np.zeros((n, m)), Gamma])          # right (causal P) input
        Cd = np.hstack([-Gamma.T, -Gamma.T @ Theta @ G])  # left (anti-causal P~) output
        # D is (o1, i1); need Dd to be (i1, i1) to match Bd=(2n,i1) and Cd=(i1,2n).
        # D.T @ D = (i1,o1)@(o1,i1) = (i1,i1). D @ D.T would give (o1,o1) — wrong.
        Dd = np.atleast_2d(D.T @ D) if D.size > 0 else np.zeros((m, m))

        dt_ss = sig.StateSpace(Ad, Bd, Cd, Dd, dt=T)
        tf_out = dt_ss.to_tf()
        if hasattr(tf_out.num, '__len__') and hasattr(tf_out.num[0], '__len__'):
            return tf_out.num[0][0], tf_out.den[0][0]
        return tf_out.num, tf_out.den
    except Exception:
        # Previously this silently returned the constant 1 — a degenerate
        # A/E that made the designed filter do nothing with no signal
        # anything was wrong.
        raise


def _ztrm_cross(A, B1, B2, C1, C2, D, T, H, o1, i1):
    """
    Cross-term B(z) = D{P21 * G_h~}(T).

    Uses the same bilateral-Z state-space framework as _dtfm2 but with
    asymmetric left/right factors:
      left  (anti-causal): G_h = ZOH * P12  → CDK_Gh = [C1, D12@K_h]
      right (causal):      P21              → CDK_P21 = [C2, D21@K_h]

    Both are embedded in the augmented Av system so they share the same
    discrete-time eigenvalues, making the Z-transform sum geometric.
    """
    try:
        n = A.shape[0]
        D12 = D[:o1, i1:]
        D21 = D[o1:, :i1]

        if H is None:
            L_h = np.zeros((1, 1))
            M_h = np.ones((1, B2.shape[1]))
            K_h = np.ones((1, 1))
        else:
            H_ss = H if isinstance(H, sig.StateSpace) else H.to_ss()
            L_h  = H_ss.A;  M_h = H_ss.B;  K_h = H_ss.C

        rH = K_h.shape[0]

        # Augmented system (shared by both factors)
        Av = np.block([[A, B2 @ K_h],
                       [np.zeros((rH, n)), L_h]])
        CDK_Gh  = np.hstack([C1, D12 @ K_h])  # G_h  output (left/anti-causal)
        CDK_P21 = np.hstack([C2, D21 @ K_h])  # P21  output (right/causal)

        # Cross integral: G = expm([-Av', M; 0, Av]*T)[top-right]
        # Sum outer products over all performance/measurement channel pairs so
        # M is (nv,nv) regardless of how many perf (o1) or meas (n_meas) channels.
        # For o1==n_meas==1 this reduces to CDK_Gh.T @ CDK_P21 (original formula).
        _v = CDK_Gh.sum(axis=0)[:, None]   # (nv,1): sum over performance channels
        _u = CDK_P21.sum(axis=0)[None, :]  # (1,nv): sum over measurement channels
        G_raw = _intaba(-Av.T, _v @ _u, Av, T)
        try:
            G = la.solve(la.expm(-Av.T * T), G_raw)
        except np.linalg.LinAlgError:
            G = G_raw

        g11 = G[:n, :n]; g12 = G[:n, n:]
        g21 = G[n:, :n]; g22 = G[n:, n:]

        # Hold integrals for each factor
        Gamma_Gh  = _intaba(A, B2 @ K_h, L_h, T)      # G_h  causal integral  (n × rH)
        Gamma_P21 = _intaba(A, B1, np.zeros((i1, i1)), T)[:n, :]  # P21 B1 integral (n × i1)

        Phi   = la.expm(A * T)
        Theta = la.solve(Phi.T, np.eye(n))             # Phi^{-T}

        nv = n + rH
        Ad = np.block([[Theta,           Theta @ Theta @ g11],
                       [np.zeros((n, n)), Phi]])

        # Bd: right/causal (P21) factor uses B1-path (Gamma_P21, not M_h)
        Bd = np.vstack([Theta @ Theta @ g12 @ M_h, Gamma_P21])

        # Cd: left/anti-causal (G_h~) factor output through M_h.T
        Cd = M_h.T @ np.hstack([-Gamma_Gh.T, g21 - Gamma_Gh.T @ Theta @ g11])

        # Dd: feedthrough (asymmetric: Cd uses G_h side, Bd uses P21 side)
        Dd = M_h.T @ (g22 - Gamma_Gh.T @ Theta @ g12) @ np.ones((i1, i1))

        dt_ss = sig.StateSpace(Ad, Bd, Cd, Dd, dt=T)
        tf_out = dt_ss.to_tf()
        if hasattr(tf_out.num, '__len__') and hasattr(tf_out.num[0], '__len__'):
            return tf_out.num[0][0], tf_out.den[0][0]
        return tf_out.num, tf_out.den
    except Exception:
        # Previously this silently returned B = 0/1 — for any plant where
        # the construction failed (e.g. i1 > 1: D21@K_h dimension mismatch)
        # the designed "optimal" controller then did nothing, with no
        # signal anything was wrong.
        raise


def _ztrm_energy(A, B1, C1, D11, T):
    """
    Bilateral Z-transform of P11 energy: D{P11~ * P11}(T).

    Same formula as _ztrm_autocorr but with B=B1, C=C1, D=D11.
    Returns a 2n-dimensional bilateral TF with palindromic denominator.
    """
    return _ztrm_autocorr(A, B1, C1, D11, T)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_plant(plant) -> Tuple[sig.StateSpace, int, int]:
    """
    Convert any plant description to (StateSpace, n_meas, n_ctrl).

    GeneralizedPlant → use stored partition sizes directly.
    SISO plants      → auto-construct a standard H2-regulation generalized plant.
    2×2+ StateSpace  → use directly with n_meas=n_ctrl=1.
    """
    if isinstance(plant, GeneralizedPlant):
        return plant.to_statespace(), plant.n_meas, plant.n_ctrl

    if isinstance(plant, sig.StateSpace):
        nout, nin = plant.C.shape[0], plant.B.shape[1]
        if nout >= 2 and nin >= 2:
            return plant, 1, 1
        return _siso_to_gen_plant(plant), 1, 1

    if isinstance(plant, sig.lti):
        ss_obj = plant.to_ss()
        return _siso_to_gen_plant(ss_obj), 1, 1

    if isinstance(plant, tuple) and len(plant) == 2:
        num, den = plant
        ss_obj = sig.lti(num, den).to_ss()
        return _siso_to_gen_plant(ss_obj), 1, 1

    raise TypeError(f"Unsupported plant type {type(plant)}")


def _siso_to_gen_plant(plant_ss: sig.StateSpace) -> sig.StateSpace:
    """
    Construct a standard 2x2 H2 generalized plant from a SISO state-space model.

    Standard form (output-regulation / LQG):
        x'  = A x + B w + B u          (disturbance w and control u at plant input)
        z1  = C x                       (performance: plant output)
        z2  = u                         (performance: control effort — D12 = 1)
        y   = C x + w                   (measurement: plant output + noise)

    D11 = 0 (no direct noise → performance) ✓
    D12 = [[0],[1]] full column rank ✓
    D21 = [[1]] full row rank ✓
    D22 = [[0]] ✓

    Resulting plant has n_in=2 (w, u), n_out=3 (z1, z2, y),
    used with n_ctrl=1, n_meas=1.
    """
    A, B, C, D = plant_ss.A, plant_ss.B, plant_ss.C, plant_ss.D
    n = A.shape[0]
    i_c = B.shape[1]   # control dimension (1 for SISO)
    o_c = C.shape[0]   # output dimension (1 for SISO)

    # Two inputs: w (disturbance, same direction as u), u (control)
    B_gen = np.hstack([B, B])     # n × 2

    # Three outputs: z1 = C*x, z2 = 0 (D12 handles it), y = -C*x + w
    # Negative sign in measurement puts h2reg in negative-feedback convention:
    # u = K*y = K*(-C*x+w), so char poly = 1+K*D22_F (matches charpol convention).
    C_gen = np.vstack([C, np.zeros((i_c, n)), -C])  # (1+1+1) × n = 3 × n

    # D matrix (3 rows × 2 cols):
    # z1: D11=D*w (D=0 for strictly proper), D12=D*u (D=0 for strictly proper)
    # z2: D11=0, D12=1 (control regularization)
    # y:  D21=1 (noise to measurement), D22=0
    D11_z1 = D                                  # o_c × i_c  (≈0 for strictly proper)
    D12_z1 = D                                  # o_c × i_c
    D11_z2 = np.zeros((i_c, i_c))
    D12_z2 = np.eye(i_c)                        # i_c × i_c  (control cost)
    D21_y  = np.eye(i_c)                        # i_c × i_c  (noise through)
    D22_y  = np.zeros((i_c, i_c))

    D_gen = np.block([[D11_z1, D12_z1],
                      [D11_z2, D12_z2],
                      [D21_y,  D22_y ]])         # (3) × 2

    return sig.StateSpace(A, B_gen, C_gen, D_gen)


def _build_2dof_siso_plant(F_ss: sig.StateSpace) -> sig.StateSpace:
    """Build a standard 2-DOF generalised plant from a SISO StateSpace."""
    n_F = F_ss.A.shape[0]
    C_p = np.vstack([-F_ss.C, np.zeros((1, n_F)), -F_ss.C])
    D_p = np.array([[0.0,  float(F_ss.D.ravel()[0])],
                    [1.0,  0.0],
                    [0.0, -float(F_ss.D.ravel()[0])]])
    return sig.StateSpace(
        F_ss.A,
        np.hstack([np.zeros((n_F, 1)), F_ss.B]),
        C_p, D_p)


def _ss_to_tf_tuple(K_ss: sig.StateSpace):
    """Convert controller StateSpace → (num, den) tuple."""
    try:
        tf = K_ss.to_tf()
        if hasattr(tf.num, '__len__') and hasattr(tf.num[0], '__len__'):
            return (np.atleast_1d(tf.num[0][0]).ravel(),
                    np.atleast_1d(tf.den[0][0]).ravel())
        return (np.atleast_1d(tf.num).ravel(),
                np.atleast_1d(tf.den).ravel())
    except Exception:
        n = K_ss.A.shape[0]
        return np.array([1.0]), np.array([1.0])


def _ensure_tf(x):
    """Ensure x is a (num, den) tuple."""
    if isinstance(x, tuple) and len(x) == 2:
        return (np.atleast_1d(np.array(x[0], float)),
                np.atleast_1d(np.array(x[1], float)))
    if isinstance(x, Poln):
        return x.coef, np.array([1.0])
    if isinstance(x, (int, float)):
        return np.array([float(x)]), np.array([1.0])
    if isinstance(x, (sig.lti, sig.dlti)):
        return x.num, x.den
    return np.array([1.0]), np.array([1.0])


# ---------------------------------------------------------------------------
# sdahinf – AHinf-optimal sampled-data controller
# ---------------------------------------------------------------------------

def sdahinf(plant, T, H=None):
    """
    AHinf-optimal controller for a sampled-data system.

    Port of sdahinf.m (K. Polyakov). MATLAB's sdahinf.m has no state-space
    alternative -- it always solves via the polynomial pipeline
    (sdh2coef -> polhinf), regardless of what representation the plant was
    given in (zpk/tf/ss all normalize to one form first). This port matches
    that exactly: one function, one pipeline. (A separate state-space-lifting
    method used to exist here as `_sdahinf_ss`/method='ss' -- removed
    since MATLAB itself has no such alternative for this function.
    `sdhinfreg`, MATLAB's own dedicated state-space
    Hinf function, is the correct place for a state-space-only design.)

    Parameters
    ----------
    plant : scipy.signal.lti / StateSpace / (num, den) tuple
        Continuous-time plant (SISO) or generalized plant (StateSpace).
    T : float
        Sampling period.
    H : scipy.signal.lti, optional
        Generalized hold (default: ZOH).

    Returns
    -------
    K : (num_d, den_d) tuple
        Optimal discrete-time controller.
    err : float
        Optimal AHinf cost (= sqrt(T) * ||T_zw||_inf).
    """
    plant_ss, n_meas, n_ctrl = _parse_plant(plant)
    # MATLAB sdahinf.m line 26: sys(end,:) = -sys(end,:) — negative-feedback
    # transform before sdh2coef (see _sdh2_pol).
    plant_neg = _negate_meas_rows(plant_ss, n_meas)
    A_tf, B_tf, E_tf, D22_tf, _zt = _sdh2coef(plant_neg, T, None, H,
                                              n_meas=n_meas, n_ctrl=n_ctrl,
                                              return_zpk=True)
    # z-domain → ζ (root-list level: conj_dt; matches MATLAB sdh2coef.m's
    # internal z2zeta; matters for the non-palindromic cross term B).
    _Az, _Bz, _Ez = (_zt[0].conj_dt(), _zt[1].conj_dt(), _zt[2].conj_dt())
    A_tf, B_tf, E_tf = _Az.to_tf(), _Bz.to_tf(), _Ez.to_tf()
    # MATLAB sdahinf.m: PCancel = intpoles(sys) — the internal (shared)
    # plant poles must cancel analytically in L. Mapped CT → ζ here
    # (exp(-λT)) since the polynomial level carries no T. Without this the
    # zterm Z retains near-singular integrator pole pairs and the H∞
    # machinery converges to a wrong boundary optimum (found on
    # demo_h2hinf: lam 0.98 vs documented 1.2251, wrong K — MATLAB's own
    # polhinf run WITHOUT PCancel in an independent Octave re-derivation
    # reproduces the same wrong answer, proving the input, not the solver,
    # was at fault).
    _p_int = _intpoles(plant_neg, n_meas, n_ctrl)
    _pc_zeta = np.exp(-_p_int * T) if len(_p_int) else None
    K_tf, lam, _ = _polhinf(A_tf, B_tf, E_tf, D22_tf, P_cancel=_pc_zeta,
                            zpk_in=(_Az, _Bz, _Ez))
    # Non-generic case: _polhinf returns a list of two equally-optimal
    # controllers (see dhinf's nonGen handling). sdahinf's own MATLAB
    # source and callers assume a single K; take the first (arbitrary
    # choice among equals — MATLAB's own dhinf help note says both give
    # the identical documented cost) rather than propagating the pair
    # through every downstream caller (sdtrhinf, sdh2hinf, ...) with no
    # documented ground truth to validate that wider change against.
    if isinstance(K_tf, list):
        K_tf = K_tf[0]
    err = float(np.sqrt(T)) * float(lam)
    return K_tf, err


# ---------------------------------------------------------------------------
# sdahinorm – AHinf norm of a sampled-data system
# ---------------------------------------------------------------------------

def sdahinorm(plant, K, T, H=None):
    """
    AHinf norm of a sampled-data closed-loop system.

    Port of sdahinorm.m → sdhinferr.m (K. Polyakov): builds the H2
    coefficients A/B/E (sdh2coef) for the UN-negated plant, negates K
    instead (sdahinorm.m line 21), forms M = feedback(K~, D22) and

        X = A·M·M~ + E − B·M − (B·M)~   (real on the unit circle)

    and returns sqrt(T)·sqrt(sup X on the circle). For the AHinf-optimal
    controller X ≡ λ² (equalizer property), so this reproduces the design
    cost exactly. (Replaces a home-grown lifting/SVD sweep that did not
    match MATLAB's documented values — 2.12 vs 1.2251 on demo_h2hinf's
    Kinf.)
    """

    plant_ss, n_meas, n_ctrl = _parse_plant(plant)
    K_num, K_den = _ensure_tf(K)
    K_num = -np.asarray(K_num, float).ravel()       # K = -K
    K_den = np.asarray(K_den, float).ravel()

    _, _, _, D22_tf, _zt = _sdh2coef(plant_ss, T, None, H,
                                     n_meas=n_meas, n_ctrl=n_ctrl,
                                     return_zpk=True)
    A_z, B_z, E_z = (_zt[0].conj_dt(), _zt[1].conj_dt(), _zt[2].conj_dt())

    # M = feedback(K', D22) in ζ: K' via the padded conjugate (z→ζ)
    Kn_z, Kd_z = _z2zeta(K_num, K_den)
    D22n = np.atleast_1d(np.asarray(D22_tf[0], float))
    D22d = np.atleast_1d(np.asarray(D22_tf[1], float))
    M_num = np.polymul(Kn_z, D22d)
    M_den = np.polyadd(np.polymul(Kd_z, D22d), np.polymul(Kn_z, D22n))
    M_z = Zpk.from_tf(M_num, M_den).minreal(1e-6)
    Mc_z = M_z.conj_dt()

    # X = A·M·M~ + E − B·M − (B·M)~  (pole-aware sums; root-list products)
    BM = (B_z * M_z).minreal(1e-6)
    X_z = (A_z * M_z * Mc_z).minreal(1e-6).zsum(E_z).zsum(-BM).zsum(-BM.conj_dt())
    try:
        X_z = X_z.minreals(1e-3)
    except ValueError:
        X_z = X_z.minreal(1e-3)

    # sdhinferr.m: eliminate (removable) poles at |ζ| = 1 together with the
    # matching zeros before the frequency sweep
    _pX = list(X_z.p); _zX = list(X_z.z)
    for _q in [q for q in _pX if abs(abs(q) - 1.0) < 1e-6]:
        if not _zX:
            break
        _d = [abs(zz - _q) for zz in _zX]
        _j = int(np.argmin(_d))
        if _d[_j] < 1e-3:
            _zX.pop(_j)
            _pX.remove(_q)
    X_z = Zpk(np.array(_zX, complex), np.array(_pX, complex), X_z.k)

    # sup of X on the unit circle (X is real there); log+linear grid to
    # resolve near-ζ=1 peaks of near-integrating plants
    w = np.unique(np.concatenate([np.logspace(-6, 0, 2048) * np.pi,
                                  np.linspace(1e-4, np.pi, 4096)]))
    zc = np.exp(1j * w)
    num_v = X_z.k * np.prod(zc[:, None] - X_z.z[None, :], axis=1) \
        if len(X_z.z) else np.full(len(zc), X_z.k, complex)
    den_v = np.prod(zc[:, None] - X_z.p[None, :], axis=1) \
        if len(X_z.p) else np.ones(len(zc), complex)
    Xv = np.real(num_v / den_v)
    Xmax = float(np.nanmax(Xv))
    if not np.isfinite(Xmax) or Xmax < 0:
        Xmax = float(np.nanmax(np.abs(Xv)))
    return float(np.sqrt(T)) * float(np.sqrt(Xmax))


# ---------------------------------------------------------------------------
# sdh2hinf – mixed H2/AHinf sampled-data design
# ---------------------------------------------------------------------------

def sdh2hinf(plant, T, rho, o11=1, i11=1, H=None):
    """
    Mixed H2/AHinf-optimal sampled-data controller.

    Port of sdh2hinf.m (K. Polyakov).

    Parameters
    ----------
    plant : lti / StateSpace / (num, den) tuple
        Continuous-time generalized plant.
    T : float
        Sampling period.
    rho : float
        Blending weight in [0, 1].  rho=1 → pure H2, rho=0 → pure AHinf.
    o11 : int
        Number of performance outputs assigned to the H2 block (default 1).
    i11 : int
        Number of exogenous inputs assigned to the H2 block (default 1).
    H : optional
        Generalized hold (default: ZOH).

    Returns
    -------
    K : (num_d, den_d) tuple
        Optimal discrete-time controller.
    err : float
        Mixed cost at the optimum.
    """
    plant_ss, n_meas, n_ctrl = _parse_plant(plant)
    # MATLAB sdh2hinf.m line 28: sys(end,:) = -sys(end,:) — negative-feedback
    # transform on the FULL plant, before the H2/AHinf sub-plants are split
    # (the measurement row lands in both sub-plants). See _sdh2_pol.
    plant_ss = _negate_meas_rows(plant_ss, n_meas)

    A_ss = plant_ss.A
    B_ss = plant_ss.B
    C_ss = plant_ss.C
    D_ss = plant_ss.D
    n_out, n_in = C_ss.shape[0], B_ss.shape[1]

    i12 = n_in  - n_ctrl - i11
    o12 = n_out - n_meas - o11
    if i11 < 1 or o11 < 1:
        raise ValueError("i11 and o11 must be >= 1")
    if i12 < 0 or o12 < 0:
        raise ValueError("H2-block dimensions exceed plant dimensions")

    # Row/column index sets (Python 0-based); last row = measurement, last col = control
    indI1 = list(range(i11)) + [n_in - 1]
    indO1 = list(range(o11)) + [n_out - 1]
    indI2 = (list(range(i11, i11 + i12)) + [n_in - 1]) if i12 > 0 else indI1
    indO2 = (list(range(o11, o11 + o12)) + [n_out - 1]) if o12 > 0 else indO1

    def _subplant(row_idx, col_idx):
        return sig.StateSpace(
            A_ss,
            B_ss[:, col_idx],
            C_ss[row_idx, :],
            D_ss[np.ix_(row_idx, col_idx)],
        )

    sysH2_ss   = _subplant(indO1, indI1)
    sysHinf_ss = _subplant(indO2, indI2)

    # Port of sdh2hinf.m's actual algorithm: build H2 and AHinf coefficient
    # sets, solve the AHinf sub-problem via polhinf to get the spectral
    # weight Sigma, blend A/B/E = rho*(H2) + (1-rho)*(AHinf)*Sigma, then
    # solve ONE polquad for the final mixed-optimal controller. (Previously
    # this solved H2/AHinf as separate state-space sub-problems and searched
    # over ad-hoc controller blends/gain-scalings — not what MATLAB does at
    # all; replaced to match MATLAB's analytical formula exactly.)
    A2, B2, E2, D22_tf, _zt2 = _sdh2coef(sysH2_ss, T, None, H,
                                         n_meas=n_meas, n_ctrl=n_ctrl,
                                         return_zpk=True)
    Ainf, Binf, Einf, _, _zti = _sdh2coef(sysHinf_ss, T, None, H,
                                          n_meas=n_meas, n_ctrl=n_ctrl,
                                          return_zpk=True)
    # z-domain → ζ-domain at the root-list level (conj_dt), as MATLAB's
    # sdh2coef.m does internally via z2zeta.
    A2_z, B2_z, E2_z = (_zt2[0].conj_dt(), _zt2[1].conj_dt(), _zt2[2].conj_dt())
    Ai_z, Bi_z, Ei_z = (_zti[0].conj_dt(), _zti[1].conj_dt(), _zti[2].conj_dt())

    _, _, Sigma = _polhinf(Ai_z.to_tf(), Bi_z.to_tf(), Ei_z.to_tf(), D22_tf,
                           zpk_in=(Ai_z, Bi_z, Ei_z))
    Sigma_z = Zpk.from_tf(np.atleast_1d(Sigma[0]), np.atleast_1d(Sigma[1]))

    def _blend(z_h2, z_hinf):
        # rho*A2 + (1-rho)*Ainf*Sigma — MATLAB sumzpk (pole-aware sum),
        # here at the root-list level: shared poles merge exactly, the
        # rho endpoints reduce to the single surviving term.
        if rho >= 1.0 - 1e-14:
            return rho * z_h2
        term = (z_hinf * Sigma_z).minreal(1e-3)
        if rho <= 1e-14:
            return (1.0 - rho) * term
        return (rho * z_h2).zsum((1.0 - rho) * term)

    A_z = _blend(A2_z, Ai_z)
    B_z = _blend(B2_z, Bi_z)
    E_z = _blend(E2_z, Ei_z)

    K_tf, err = _polquad(A_z.to_tf(), B_z.to_tf(), E_z.to_tf(), D22_tf,
                         zpk_in=(A_z, B_z, E_z))
    err = float(np.sqrt(max(err, 0.0)))
    return K_tf, err


# ---------------------------------------------------------------------------
# sdtrhinf / sdtrhinferr – AHinf tracking design and norm
# ---------------------------------------------------------------------------

def sdtrhinf(plant, T, H=None):
    """
    AHinf-optimal controller for sampled-data tracking systems.

    Port of sdtrhinf.m (K. Polyakov).

    Uses the L2 cost structure (sdl2coef) instead of the H2 structure used
    by sdahinf, making it appropriate for tracking rather than regulation.

    Parameters
    ----------
    plant : lti / StateSpace / (num, den) tuple
        Continuous-time plant.
    T : float
        Sampling period.
    H : optional
        Generalized hold (default: ZOH).

    Returns
    -------
    K : (num_d, den_d) tuple
        AHinf-optimal discrete-time controller.
    err : float
        Optimal AHinf cost (scaled by sqrt(T)).
    """
    plant_ss, n_meas, n_ctrl = _parse_plant(plant)
    # Port of MATLAB sdtrhinf.m: sys(end,:) = -sys(end,:)
    # sdtrhinferr does NOT apply this negation (uses original sys).
    plant_neg = _negate_meas_rows(plant_ss, n_meas)
    A_tf, B_tf, E_tf, D22_tf, _zt = _sdl2coef(plant_neg, T, H, n_meas,
                                              n_ctrl, return_zpk=True)
    K_tf, lam, _ = _polhinf(A_tf, B_tf, E_tf, D22_tf, zpk_in=_zt)
    # Non-generic case: see sdahinf's identical carve-out — take the first
    # of the two equally-optimal controllers rather than propagating the
    # pair (no documented ground truth here to validate a wider change).
    if isinstance(K_tf, list):
        K_tf = K_tf[0]
    err = float(np.sqrt(T)) * float(lam)
    return K_tf, err


def sdtrhinferr(plant, K, T, H=None):
    """
    AHinf norm for a sampled-data tracking system.

    Port of sdtrhinferr.m (K. Polyakov).

    Evaluates the AHinf cost on the unit circle using the L2 coefficient
    structure (sdl2coef), matching the cost minimised by sdtrhinf.

    Parameters
    ----------
    plant : lti / StateSpace / (num, den) tuple
        Continuous-time plant.
    K : (num_d, den_d) tuple or dlti
        Discrete-time controller (negative feedback convention).
    T : float
        Sampling period.
    H : optional
        Generalized hold (default: ZOH).

    Returns
    -------
    err : float
        AHinf tracking norm (scaled by sqrt(T)).
    """
    plant_ss, n_meas, n_ctrl = _parse_plant(plant)
    K_num, K_den = _ensure_tf(K)
    A_tf, B_tf, E_tf, D22_tf = _sdl2coef(plant_ss, T, H, n_meas, n_ctrl)

    def _r(coef):
        return np.real(np.asarray(coef)).astype(float).ravel()

    A_num, A_den   = _r(A_tf[0]),   _r(A_tf[1])
    B_num, B_den   = _r(B_tf[0]),   _r(B_tf[1])
    E_num, E_den   = _r(E_tf[0]),   _r(E_tf[1])
    D22_num, D22_den = _r(D22_tf[0]), _r(D22_tf[1])
    K_n,  K_d     = _r(K_num),      _r(K_den)

    # Start at pi/N to avoid w=0 where integrating K (pole at z=1) gives
    # K_f→∞, making M_f = ∞/∞ = nan.  np.nanmax handles any remaining nans.
    # Note: MATLAB's sdhinferr cancels unit-circle poles from X symbolically
    # before computing the sup-norm; this frequency sweep cannot do that, so
    # results may differ from MATLAB when A or E have poles at z=1.
    N = 2048
    w = np.linspace(np.pi / N, np.pi, N)
    z = np.exp(1j * w)

    def _evtf(num, den):
        return np.polyval(num, z) / np.polyval(den, z)

    A_f   = np.real(_evtf(A_num, A_den))
    B_f   = _evtf(B_num, B_den)
    E_f   = np.real(_evtf(E_num, E_den))
    D22_f = _evtf(D22_num, D22_den)
    K_f   = _evtf(K_n, K_d)

    # M = feedback((-K)', D22) — negate K for positive→negative feedback
    K_neg_cr = -np.conj(K_f)
    M_f = K_neg_cr / (1.0 + K_neg_cr * D22_f)

    X_f = A_f * np.abs(M_f)**2 + E_f - 2.0 * np.real(B_f * M_f)
    X_f = np.maximum(X_f, 0.0)

    return float(np.sqrt(T)) * float(np.sqrt(float(np.nanmax(X_f))))


# ---------------------------------------------------------------------------
# modsdh2 / modsdl2 – reduced-order modal optimal controllers
# ---------------------------------------------------------------------------

def _par2cp(rho, alpha_T=0.0, beta=np.inf):
    """
    Map rho ∈ [0,1]^ord_K → closed-loop characteristic polynomial Delta(z).

    Pairs of entries parameterise complex-conjugate pole pairs; single
    trailing entries parameterise real poles.  Port of par2cp.m (Polyakov).
    """
    rho = np.asarray(rho, dtype=float).ravel()
    n_pairs = len(rho) // 2
    n_real  = len(rho) % 2

    alpha_T  = float(alpha_T)
    beta_val = float(beta)
    shifted  = beta_val < 0.0
    beta_abs = abs(beta_val)

    Ea = np.exp(-alpha_T)
    Eb = np.exp(-np.pi / beta_abs) if not np.isinf(beta_abs) else 0.0

    if not shifted:
        E0     = min(Ea, Eb) if not np.isinf(beta_abs) else Ea
        r2_min = -Ea * E0
    else:
        E0     = Ea * Eb
        r2_min = -(Ea ** 2) * Eb

    r2_max  = Ea ** 2
    r2_crit = E0 ** 2

    k = 0
    Delta = np.array([1.0])

    for _ in range(n_pairs):
        r2   = r2_min + np.clip(rho[k], 0.0, 1.0) * (r2_max - r2_min)
        r1_lo = -r2 / max(Ea, 1e-12) - Ea

        if r2 <= r2_crit or E0 < 1e-12:
            r1_hi = r2 / max(E0, 1e-12) + E0
        else:
            sqr2 = np.sqrt(max(r2, 0.0))
            if shifted:
                sqr2 = sqr2 / max(Ea, 1e-12)
            if np.isinf(beta_abs) or sqr2 < 1e-12:
                r1_hi = 2.0 * np.sqrt(max(r2, 0.0))
            else:
                phi_max = beta_abs * np.log(1.0 / sqr2)
                r1_hi   = -2.0 * sqr2 * np.cos(phi_max)

        r1 = r1_lo + np.clip(rho[k + 1], 0.0, 1.0) * (r1_hi - r1_lo)
        Delta = np.polymul(Delta, np.array([1.0, r1, r2]))
        k += 2

    r0_lo = -Ea
    r0_hi = E0
    for _ in range(n_real):
        r0 = r0_lo + np.clip(rho[k], 0.0, 1.0) * (r0_hi - r0_lo)
        if abs(r0) < 1e-3:
            r0 = 0.0
        Delta = np.polymul(Delta, np.array([1.0, r0]))
        k += 1

    return Delta


def _modal_par2k(Delta_z, D22_num, D22_den):
    """
    Build minimal-degree stabilising K via Bezout identity in z-domain.

    Solves X·D22_num + Y·D22_den = Delta_z → K = X/Y.
    Simplified port of go_par2k.m (ksi = 0, no polopth2).
    """
    n = np.asarray(D22_num, float).ravel()
    d = np.asarray(D22_den, float).ravel()
    c = np.asarray(Delta_z, float).ravel()

    try:
        X_p, Y_p, _err, _ = dioph(n, d, c)
    except Exception:
        return None, None

    X_coef = np.real(np.asarray(
        X_p.coef if hasattr(X_p, 'coef') else X_p, float)).ravel()
    Y_coef = np.real(np.asarray(
        Y_p.coef if hasattr(Y_p, 'coef') else Y_p, float)).ravel()

    X_coef = striplz(X_coef) if len(X_coef) > 1 else X_coef
    Y_coef = striplz(Y_coef) if len(Y_coef) > 1 else Y_coef
    if len(X_coef) == 0:
        X_coef = np.array([0.0])
    if len(Y_coef) == 0:
        Y_coef = np.array([1.0])

    return X_coef, Y_coef


def _modal_plant_tf(plant):
    """Extract (num, den) arrays from any SISO plant description."""
    if isinstance(plant, sig.StateSpace):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tf = plant.to_tf()
        if hasattr(tf.num, '__len__') and hasattr(tf.num[0], '__len__'):
            return (np.asarray(tf.num[0][0], float).ravel(),
                    np.asarray(tf.den[0][0], float).ravel())
        return np.asarray(tf.num, float).ravel(), np.asarray(tf.den, float).ravel()
    if isinstance(plant, sig.lti):
        return np.asarray(plant.num, float).ravel(), np.asarray(plant.den, float).ravel()
    if isinstance(plant, tuple) and len(plant) == 2:
        return np.asarray(plant[0], float).ravel(), np.asarray(plant[1], float).ravel()
    raise TypeError(f"modsdh2/modsdl2 require a SISO plant; got {type(plant)}")


def _extract_p22_ss(plant):
    """
    Extract the P22 channel (last-row, last-col) from a generalized plant.

    GeneralizedPlant → use the stored partition via the .P22 property.
    MIMO StateSpace  → extract lower-right 1×1 corner.
    SISO inputs      → returned as-is (converted to StateSpace).
    Matches MATLAB modalopt.m: P22 = sys(end,end).
    """
    if isinstance(plant, GeneralizedPlant):
        return Minreal.ss(plant.P22)
    if isinstance(plant, sig.StateSpace):
        nout, nin = plant.C.shape[0], plant.B.shape[1]
        if nout >= 2 and nin >= 2:
            p22 = sig.StateSpace(plant.A, plant.B[:, -1:],
                                 plant.C[-1:, :], plant.D[-1:, -1:])
            return Minreal.ss(p22)
        return plant
    if isinstance(plant, sig.lti):
        return plant.to_ss()
    if isinstance(plant, tuple) and len(plant) == 2:
        return sig.lti(*plant).to_ss()
    raise TypeError(f"Cannot extract P22 from {type(plant)}")


def _modalopt(plant, T, ord_K, alpha, beta, use_l2, method, n_iter, H=None):
    """
    Global optimisation over reduced-order controller pole parameters.

    Port of modalopt.m (K. Polyakov).  Parameterises the closed-loop
    characteristic polynomial via _par2cp, builds K via _modal_par2k for
    each candidate, and minimises the H2 (or L2) cost.
    """
    from directsd.polynomial.transforms import dtfm
    from directsd.glopt.optimize import randsearch, crandsearch, dual_annealing

    # For MIMO generalized plants: extract P22 = sys(-1,-1) for parameterisation
    # but retain the full plant for cost evaluation.
    # MATLAB modalopt.m: P22 = sys(end,end); Pz = -dtfm(P22, T, 0)
    # The negation is part of the feedback-convention matching in the toolbox.
    _p22_ss  = _extract_p22_ss(plant)
    _is_mimo = (isinstance(plant, GeneralizedPlant) or
                (isinstance(plant, sig.StateSpace) and
                 plant.C.shape[0] >= 2 and plant.B.shape[1] >= 2))
    plant_num, plant_den = _modal_plant_tf(_p22_ss)

    n_plant = len(plant_den) - 1
    if ord_K is None:
        ord_K = max(1, n_plant)
    ord_K = max(1, int(ord_K))

    alpha_T = float(alpha) * float(T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D22_num_raw, D22_den_raw = dtfm((plant_num, plant_den), T)
    D22_num = np.real(np.asarray(D22_num_raw, float)).ravel()
    D22_den = np.real(np.asarray(D22_den_raw, float)).ravel()
    # MATLAB: Pz = -dtfm(P22, T, 0). For MIMO sys the last-row convention
    # means P22 = sys(-1,-1) is already sign-negated relative to the physical
    # plant, so we apply the same negation to produce the correct Bezout basis.
    if _is_mimo:
        D22_num = -D22_num

    if use_l2:
        from directsd.analysis.errors import sdl2err as _cost_fn
    else:
        from directsd.analysis.norms import sdh2norm as _cost_fn

    _COST_PENALTY = 1e8  # large finite fallback; scipy dual_annealing rejects inf/nan

    def _cost(rho):
        try:
            Delta_z = _par2cp(rho, alpha_T, beta)
            K_num, K_den = _modal_par2k(Delta_z, D22_num, D22_den)
            if K_num is None:
                return _COST_PENALTY
            val = float(_cost_fn(plant, (K_num, K_den), T))
            return val if np.isfinite(val) else _COST_PENALTY
        except Exception:
            return _COST_PENALTY

    # MATLAB modalopt uses dim = deg(Pz) + ord_K parameters, not just ord_K.
    # With only ord_K params, Delta has degree ord_K < n_plant+ord_K, causing
    # the Bezout solution to have a spurious unstable pole.
    total_dim = n_plant + ord_K
    bounds = [(0.0, 1.0)] * total_dim
    x0     = np.full(total_dim, 0.5)
    meth   = str(method).lower()

    if meth in ('crandsearch', 'crs'):
        x_best, f_best, _ = crandsearch(_cost, bounds, max_feval=n_iter)
    elif meth in ('dual_annealing', 'annealing', 'da'):
        x_best, f_best, _ = dual_annealing(_cost, bounds, maxiter=n_iter)
    else:
        x_best, f_best, _ = randsearch(_cost, x0, bounds=bounds, n_iter=n_iter)

    rho_best         = np.clip(x_best, 0.0, 1.0)
    Delta_z          = _par2cp(rho_best, alpha_T, beta)
    K_num, K_den     = _modal_par2k(Delta_z, D22_num, D22_den)

    if K_num is None:
        K_num, K_den = np.array([1.0]), np.array([1.0])
        f_best       = float('inf')

    return (K_num, K_den), float(f_best)


def modsdh2(plant, T, ord_K=None, alpha=0.0, beta=np.inf,
            method='randsearch', n_iter=500, H=None):
    """
    Reduced-order H2-optimal sampled-data controller (modal parameterisation).

    Minimises the H2 norm over the family of stabilising controllers of order
    ``ord_K`` whose closed-loop poles lie in the stability sector
    (alpha, beta).  Uses global random search in [0,1]^ord_K.

    Port of modsdh2.m (K. Polyakov).

    Parameters
    ----------
    plant : scipy.signal.StateSpace or (num, den) tuple or scipy lti
        Continuous-time generalised plant (preferred, same as passed to sdh2)
        or SISO plant.  For a generalised plant the P22 channel is extracted
        internally for pole parameterisation while the full plant is used for
        cost evaluation, matching MATLAB's modalopt.m behaviour.
    T : float
        Sampling period.
    ord_K : int, optional
        Controller order (default: plant order).
    alpha : float
        Stability margin; closed-loop poles satisfy |z| <= exp(-alpha*T).
    beta : float
        Stability sector angle (rad/s); np.inf disables the sector constraint.
    method : str
        'randsearch' (default), 'crandsearch', or 'dual_annealing'.
    n_iter : int
        Optimiser budget (number of cost-function evaluations).
    H : optional
        Generalised hold (reserved; ZOH is used).

    Returns
    -------
    K : (num_d, den_d) tuple
        Optimal discrete-time controller.
    err : float
        Achieved H2 cost.
    """
    return _modalopt(plant, T, ord_K, alpha, beta,
                     use_l2=False, method=method, n_iter=n_iter, H=H)


def modsdl2(plant, T, ord_K=None, alpha=0.0, beta=np.inf,
            method='randsearch', n_iter=500, H=None):
    """
    Reduced-order L2-optimal sampled-data controller (modal parameterisation).

    Identical to modsdh2 but minimises the L2 (tracking) norm.

    Port of modsdl2.m (K. Polyakov).

    Parameters
    ----------
    plant : scipy.signal.StateSpace or (num, den) tuple or scipy lti
        Continuous-time generalised plant (preferred, same as passed to sdl2)
        or SISO plant.  For a generalised plant the P22 channel is extracted
        internally for pole parameterisation while the full plant is used for
        cost evaluation, matching MATLAB's modalopt.m behaviour.
    T : float
        Sampling period.
    ord_K : int, optional
        Controller order (default: plant order).
    alpha : float
        Stability margin; closed-loop poles satisfy |z| <= exp(-alpha*T).
    beta : float
        Stability sector angle (rad/s); np.inf disables the sector constraint.
    method : str
        'randsearch' (default), 'crandsearch', or 'dual_annealing'.
    n_iter : int
        Optimiser budget (number of cost-function evaluations).
    H : optional
        Generalised hold (reserved; ZOH is used).

    Returns
    -------
    K : (num_d, den_d) tuple
        Optimal discrete-time controller.
    err : float
        Achieved L2 cost.
    """
    return _modalopt(plant, T, ord_K, alpha, beta,
                     use_l2=True, method=method, n_iter=n_iter, H=H)


# ---------------------------------------------------------------------------
# polhinf – polynomial H∞ optimisation
# ---------------------------------------------------------------------------

def _np_coprime(A, B):
    """coprime for raw numpy arrays: returns (A/g, B/g, g) as float arrays."""
    A = np.real(np.asarray(A, complex)).astype(float).ravel()
    B = np.real(np.asarray(B, complex)).astype(float).ravel()
    _one = np.array([1.0])
    if len(A) <= 1 or len(B) <= 1:
        return A.copy(), B.copy(), _one
    try:
        pA, pB, pG = coprime(A, B)
        return (np.real(pA.coef).astype(float),
                np.real(pB.coef).astype(float),
                np.real(pG.coef).astype(float))
    except Exception:
        return A.copy(), B.copy(), _one


def _np_triple(A, B, C):
    """triple for raw numpy arrays: returns (A/g, B/g, C/g) as float arrays."""
    A = np.real(np.asarray(A, complex)).astype(float).ravel()
    B = np.real(np.asarray(B, complex)).astype(float).ravel()
    C = np.real(np.asarray(C, complex)).astype(float).ravel()
    try:
        pA, pB, pC, _ = triple(A, B, C)
        return (np.real(pA.coef).astype(float),
                np.real(pB.coef).astype(float),
                np.real(pC.coef).astype(float))
    except Exception:
        return A.copy(), B.copy(), C.copy()



def _tfsub(n1, d1, n2, d2):
    """Rational subtraction n1/d1 - n2/d2, raw numpy coefficient arrays."""
    t1 = np.polymul(n1, d2); t2 = np.polymul(n2, d1)
    ml = max(len(t1), len(t2))
    t1p = np.concatenate([np.zeros(ml - len(t1)), t1])
    t2p = np.concatenate([np.zeros(ml - len(t2)), t2])
    return np.real(t1p - t2p).astype(float), np.real(np.polymul(d1, d2)).astype(float)


def _tfpadd(a, b, sign=1.0):
    """Polynomial add/subtract with length alignment (leading-zero padding)."""
    ml = max(len(a), len(b))
    ap = np.concatenate([np.zeros(ml - len(a)), a])
    bp = np.concatenate([np.zeros(ml - len(b)), b])
    return np.real(ap + sign * bp).astype(float)


def _pdeg(p):
    """Polynomial degree via leading-zero trimming, for raw numpy coefficient
    arrays. Shared by _hinfred_np/_hinffiter_np/_hinfbisec_np (previously
    duplicated identically in each)."""
    t = np.trim_zeros(np.asarray(p, float), 'f')
    return len(t) - 1 if len(t) > 0 else 0


def _xi0_init(aa, gg, n_pts=1024):
    """Initial guess xi0 = 1/sqrt(zinfnorm(gg/aa)) for the Hinf bisection solvers.

    Port of MATLAB hinfred.m/hinffiter.m/hinfbisec.m's shared
    ``xi0 = 1/sqrt(zinfnorm(gg/aa))`` step: approximates the discrete Linf-norm
    of gg/aa on the unit circle by frequency-gridding rather than
    ``zinfnorm.m``'s exact state-space Hinf-norm algorithm (a fine
    approximation since xi0 here only seeds the Newton-Raphson / bisection
    iteration). Shared by all three solvers — previously duplicated in each.
    """
    ww = np.linspace(1e-4, np.pi, n_pts)
    zz = np.exp(1j * ww)
    ratio = np.abs(np.polyval(gg, zz)) / (np.abs(np.polyval(aa, zz)) + 1e-300)
    return 1.0 / np.sqrt(np.max(ratio) + 1e-300)


def _ynyd(A_tf, B_tf, D22_tf, P_cancel=None, zpk_in=None):
    """
    Shared port of MATLAB ``private/ynyd.m`` — dispatcher.

    Tries the ROOT-LIST (zpk) implementation first (`_ynyd_zpk` —
    cancellations by exact root matching, mirroring MATLAB's zpk
    arithmetic; the coefficient-array implementation `_ynyd_coef` remains
    as a fallback). ``zpk_in`` optionally carries (A_z, B_z) as
    zpk.zpk.Zpk objects straight from the coefficient builders.
    """
    try:
        if zpk_in is not None:
            A_z, B_z = zpk_in
        else:
            A_z = Zpk.from_tf(A_tf[0], A_tf[1])
            B_z = Zpk.from_tf(B_tf[0], B_tf[1])
        return _ynyd_zpk(A_z, B_z, D22_tf, P_cancel)
    except Exception as _exc:
        warnings.warn(f"_ynyd: root-list path failed ({_exc}); "
                      f"using coefficient fallback")
        return _ynyd_coef(A_tf, B_tf, D22_tf, P_cancel)


def _ynyd_zpk(A_z, B_z, D22_tf, P_cancel=None):
    """
    ROOT-LIST port of ynyd.m — every rational operation runs on
    (zeros, poles, gain) lists with exact-match cancellation, mirroring
    MATLAB's zpk pipeline statement by statement:

        dzpk = zpk(d, 1, T)
        Av   = minreal(dzpk*dzpk*dzpk'*dzpk'*A);  Lam = sfactor(Av)
        [a0,b0] = dioph(n, d, 1)
        L1 = minreal(dzpk'*dzpk'*B'/Lam');  L2 = minreal(a0*Lam/dzpk)
        L  = minreal(sumzpk(L1, -L2));  setpoles(L, [L1.p; others(L2.p,L1.p)])
        [dLp,dLm,dL0] = factor(dL);  dLp = dLp*dL0
        dLLam = minreal(zpk(dLp,1,T)*Lam)
        na/da = tf2nd(minreal(dzpk/dLLam));  nb/db = tf2nd(minreal(n/dLLam))
        R1 = minreal(zpk(dL,1,T)*L1)
        YN = mrdivide(tf2nd(minreal(na*R1)))
        YD = mrdivide(tf2nd(minreal(sumzpk(dLm*db/dzpk, -R1*nb), 1e-4)))

    Returns the same coefficient-array dict as `_ynyd_coef` (denominators
    monic with the gain in the numerator — MATLAB's tf2nd convention).
    """

    def _r(x):
        return np.real(np.asarray(x, float)).ravel()

    D22_num, D22_den = _r(D22_tf[0]), _r(D22_tf[1])
    d = _r(_strip_lz(D22_den))
    n = _r(_strip_lz(D22_num))

    d_z = Zpk.from_tf(d, np.array([1.0]))
    dc = d_z.conj_dt()

    # Av = minreal(dzpk²·dzpk'²·A);  Lam = sfactor(Av)  — RATIONAL factor
    Av_z = (d_z * d_z * dc * dc * A_z).minreal()
    Lam_z = Av_z.sfactor('d')
    Lam, Lam_den = Lam_z.to_tf()

    # a0·n + b0·d = 1
    n_pol = Poln(n, 'z'); d_pol = Poln(d, 'z'); one = Poln([1.0], 'z')
    try:
        a0_p, b0_p, _, _ = dioph(n_pol, d_pol, one)
        a0 = _r(a0_p.coef); b0 = _r(b0_p.coef)
    except Exception:
        a0 = np.array([1.0]); b0 = np.array([0.0])

    # L1 = minreal(dzpk'·dzpk'·B'/Lam');  L2 = a0·Lam/dzpk (or 0)
    L1_z = (dc * dc * B_z.conj_dt() / Lam_z.conj_dt()).minreal()
    if np.linalg.norm(a0) > np.finfo(float).eps:
        L2_z = (Zpk.from_tf(a0, np.array([1.0])) * Lam_z / d_z).minreal()
    else:
        L2_z = Zpk([], [], 0.0)

    # L = minreal(sumzpk(L1, -L2)) with the zero-gain check
    L_z = L1_z.zsum(-L2_z)
    if abs(L_z.k) < 1e-10 * (abs(L1_z.k) + 1e-300):
        L_z = Zpk([], [], 0.0)

    # Correct poles of L (ynyd.m lines 57-70):
    #   pL = [L1.p; others(L2.p, L1.p)];
    #   pL = others(pL, pCancel);  dLCancel = poln(pCancel,'rz')  (if given)
    #   L  = setpoles(L, pL)
    # P_cancel here is the ζ-domain root list exp(-PCancel·T) — the caller
    # (sdahinf/sdh2 via intpoles) does the CT→ζ mapping since this level
    # has no T.
    pL2x, _ = _others2(L2_z.p, L1_z.p)
    pL = np.concatenate([L1_z.p, np.atleast_1d(pL2x)]) if len(L1_z.p) or len(np.atleast_1d(pL2x)) \
        else np.zeros(0, complex)
    dLCancel = np.array([1.0])
    if P_cancel is not None and len(np.atleast_1d(P_cancel)):
        _pc = np.atleast_1d(np.asarray(P_cancel, complex)).ravel()
        pL, _ = _others2(pL, _pc, tol=1e-4)
        dLCancel = np.real(np.poly(_pc)).astype(float)
    try:
        L_z = L_z.setpoles(pL)
    except ValueError:
        # setpoles could not cancel an extra pole against any zero — fall
        # back to value-snapping only (the pre-PCancel behaviour)
        if len(L_z.p) and len(pL):
            newp = L_z.p.copy()
            avail = list(pL)
            for i, rt in enumerate(newp):
                if not avail:
                    break
                dd = [abs(rt - q) for q in avail]
                j = int(np.argmin(dd))
                if dd[j] < 1e-3 * (1.0 + abs(rt)):
                    newp[i] = avail.pop(j)
            L_z = Zpk(L_z.z, newp, L_z.k)

    nL_arr, L_d = L_z.to_tf()
    nL = _r(_strip_lz(nL_arr))
    if len(nL) == 0 or np.all(np.abs(nL) < 1e-30):
        nL = b0 if np.linalg.norm(b0) > 1e-10 else np.array([1.0])
    L1_n, L1_d = L1_z.to_tf()
    L2_n, L2_d = L2_z.to_tf()

    # [dLp, dLm, dL0] = factor(dL); dLp = dLp*dL0 — classification directly
    # on L's POLE LIST (no re-rooting), with the unit-circle snap and
    # near-real forcing kept from earlier numerical-robustness fixes.
    rts_dL = L_z.p.copy()
    _tol_n = 5e-4
    _im_tol = 1e-6
    _mask_re = np.abs(np.imag(rts_dL)) < _im_tol * (1.0 + np.abs(rts_dL))
    rts_dL = np.where(_mask_re, np.real(rts_dL), rts_dL)
    dLp_rts = rts_dL[np.abs(rts_dL) > 1.0 - _tol_n]
    dLm_rts = rts_dL[np.abs(rts_dL) <= 1.0 - _tol_n]
    dL0_rts = rts_dL[(np.abs(rts_dL) >= 1.0 - _tol_n) & (np.abs(rts_dL) <= 1.0 + _tol_n)]
    if dL0_rts.size > 0:
        dL0_rts = dL0_rts / np.abs(dL0_rts)
        for _r0 in dL0_rts:
            _dists = np.abs(dLp_rts - _r0)
            if _dists.size > 0 and _dists.min() < 10 * _tol_n:
                dLp_rts[np.argmin(_dists)] = _r0
    dLp = np.real(np.poly(dLp_rts)).astype(float) if len(dLp_rts) else np.array([1.0])
    dLm = np.real(np.poly(dLm_rts)).astype(float) if len(dLm_rts) else np.array([1.0])
    dL0 = np.real(np.poly(dL0_rts)).astype(float) if len(dL0_rts) else np.array([1.0])

    # dLLam = minreal(zpk(dLp,1,T)*Lam); na/da = d/dLLam; nb/db = n/dLLam
    dLp_z = Zpk(np.concatenate([dLp_rts]) if len(dLp_rts) else [], [], 1.0)
    dLLam_z = (dLp_z * Lam_z).minreal()
    na, da = (d_z / dLLam_z).minreal().to_tf()
    nb, db = (Zpk.from_tf(n, np.array([1.0])) / dLLam_z).minreal().to_tf()
    na = _r(_strip_lz(na)); da = _r(_strip_lz(da))
    nb = _r(_strip_lz(nb)); db = _r(_strip_lz(db))

    # R1 = minreal(zpk(dL,1,T)*L1)
    dL_z = Zpk(L_z.p, [], 1.0)
    R1_z = (dL_z * L1_z).minreal()

    # YN = na*R1 → polynomial quotient
    YN_z = (Zpk.from_tf(na, np.array([1.0])) * R1_z).minreal()
    nYN, dYN = YN_z.to_tf()
    if len(dYN) <= 1:
        _sc = float(dYN[0]) if len(dYN) == 1 and abs(float(dYN[0])) > 1e-30 else 1.0
        YN = _r(_strip_lz(nYN / _sc))
    else:
        YN = _r(_strip_lz(np.polydiv(nYN, dYN)[0]))
    if len(YN) == 0:
        YN = np.array([0.0])

    # YD = sumzpk(dLm*db/dzpk, -R1*nb) → minreal(·,1e-4) → quotient
    YD1_z = (Zpk.from_tf(np.polymul(dLm, db), np.array([1.0])) / d_z).minreal()
    YD2_z = (R1_z * Zpk.from_tf(nb, np.array([1.0]))).minreal()
    YD_z = YD1_z.zsum(-YD2_z).minreal(1e-4)
    nYD, dYD = YD_z.to_tf()
    # ynyd.m: if norm(nYD) < 1e-3*(norm(YD1)+norm(YD2)): nYD = 0
    _n1 = np.linalg.norm(YD1_z.to_tf()[0]); _n2 = np.linalg.norm(YD2_z.to_tf()[0])
    if np.linalg.norm(nYD) < 1e-3 * (_n1 + _n2):
        nYD = np.array([0.0])
    if len(dYD) <= 1:
        _sc = float(dYD[0]) if len(dYD) == 1 and abs(float(dYD[0])) > 1e-30 else 1.0
        YD = _r(_strip_lz(nYD / _sc))
    else:
        YD = _r(_strip_lz(np.polydiv(nYD, dYD)[0]))
    if len(YD) == 0:
        YD = np.array([0.0])

    return {
        'n': n, 'd': d, 'a0': a0, 'b0': b0, 'Lam': Lam, 'Lam_den': Lam_den,
        'na': na, 'da': da, 'nb': nb, 'db': db,
        'dLp': dLp, 'dLm': dLm, 'dL0': dL0, 'nL': nL,
        'YN': YN, 'YD': YD,
        'L1_n': L1_n, 'L1_d': L1_d, 'L2_n': L2_n, 'L2_d': L2_d,
        'L_n': nL_arr, 'L_d': L_d,
        'dLCancel': dLCancel,
    }


def _ynyd_coef(A_tf, B_tf, D22_tf, P_cancel=None):
    """
    Shared port of MATLAB ``private/ynyd.m`` (K. Polyakov) — auxiliary
    polynomials for polynomial sampled-data design, used by both
    :func:`_polquad` and :func:`_polhinf`.

    Note: MATLAB's PCancel/dLCancel pole-cancellation bookkeeping (ynyd.m
    lines 60-69) is not implemented; ``P_cancel`` is accepted for API
    symmetry with the MATLAB signature but is currently ignored, matching
    both call sites' prior behaviour.

    Returns
    -------
    dict with keys (all plain numpy coefficient arrays, ζ-domain, highest
    power first):
      n, d            - polynomials forming D22 (numerator, denominator)
      a0, b0          - basic controller: a0*n + b0*d = 1
      Lam, Lam_den    - RATIONAL spectral factor Lam/Lam_den of
                        Av = d^2 * d~^2 * A (Lam_den = stable half of the
                        part of A's denominator that does not cancel)
      na, da, nb, db  - coprime pairs: na/da = d/dLLam, nb/db = n/dLLam
      dLp, dLm, dL0   - stable(+neutral), antistable, neutral factors of dL
                        (ynyd.m: dLp = dLp*dL0 — neutral is merged into dLp)
      nL              - numerator of L = L1 - L2
      YN, YD          - polynomials for the two Diophantine equations
      L1_n, L1_d      - antistable part of L (numerator, denominator)
      L2_n, L2_d      - stable part of L (numerator, denominator)
      L_n, L_d        - L = L1 - L2 (numerator, denominator)
    """
    def _r(x):
        return np.real(np.asarray(x, float)).ravel()

    # ── Factorization (ynyd.m lines 31-35) ──────────────────────────────────
    D22_num, D22_den = _r(D22_tf[0]), _r(D22_tf[1])
    A_num, A_den = _r(A_tf[0]), _r(A_tf[1])
    B_num, B_den = _r(B_tf[0]), _r(B_tf[1])

    d = _r(_strip_lz(D22_den))
    n = _r(_strip_lz(D22_num))
    k_d = len(d) - 1     # deg(d)
    # rev(d) — the polynomial part of dzpk' = rev(d)/ζ^k (@zpk/ctranspose.m:
    # conjugating a k-degree polynomial adds k origin POLES; those ζ-factors
    # are tracked explicitly below wherever dzpk' enters a rational).
    d_cr = d[::-1]

    d_sq    = np.polymul(d, d)
    d_cr_sq = np.polymul(d_cr, d_cr)
    d2d2    = np.polymul(d_sq, d_cr_sq)
    Av_num  = np.polymul(d2d2, A_num)
    # Cancel A_den from Av_num: try EXACT polynomial division first (coefficient-level,
    # no rootfinding involved), falling back to root-matching only if that fails.
    #
    # Why division-first matters: A_den's roots are often *structurally guaranteed* to
    # equal roots of d/d~ (e.g. a common plant pole reused in both A and D22) — but if
    # we cancel via np.roots(Av_num) + np.roots(A_den) + tolerance-match + np.poly
    # rebuild, each of those two independent np.roots() calls re-derives the "same"
    # mathematical root from scratch and can land at very slightly different floats
    # (e.g. 2.0 vs 2.004). That gap is small enough to not matter here, but it silently
    # propagates into `Lam` (via sfactor below) and then into na/da/nb/db, where it's
    # exactly what prevents downstream _tfmin calls (in polquad/polhinf) from fully
    # cancelling a pole-zero pair that MATLAB's exact zpk-based root tracking always
    # cancels cleanly — producing a higher-order, sometimes-unstable K where MATLAB
    # gets a lower-order/stable one. Plain polynomial long
    # division never re-derives A_den's roots at all, so it can't introduce this drift.
    _Avq, _Avr = np.polydiv(Av_num, A_den) if len(A_den) > 1 else (Av_num, np.array([0.0]))
    if np.linalg.norm(_Avr) < 1e-6 * (np.linalg.norm(Av_num) + 1e-30):
        Av = np.real(_strip_lz(_Avq)).astype(float)
        Av_den = np.array([float(np.real(A_den[0]))]) if len(A_den) > 0 else np.array([1.0])
    else:
        # Division left a non-negligible remainder: A_den isn't an exact factor of
        # Av_num (or is only approximately one) — fall back to root-matching.
        # Unmatched A_den roots STAY in the denominator (Av — and Lam below —
        # are genuinely RATIONAL then, matching MATLAB's minreal(dzpk²·dzpk'²·A)
        # + zpk-level sfactor); previously they were silently dropped, the same
        # defect the open-loop fix removed for the D22=0 branch (see _polquad).
        _rts_avn = np.roots(Av_num).tolist() if len(Av_num) > 1 else []
        _rts_ade = np.roots(A_den).tolist() if len(A_den) > 1 else []
        _unused  = list(_rts_ade)
        _reduced = []
        for _rv in _rts_avn:
            _hit = False
            for _j, _rd in enumerate(_unused):
                if abs(_rv - _rd) < 1e-3 * (1.0 + abs(_rv)):
                    _unused.pop(_j); _hit = True; break
            if not _hit:
                _reduced.append(_rv)
        _n_sc = float(np.real(Av_num[0]))
        _d_sc = float(np.real(A_den[0])) if len(A_den) > 0 else 1.0
        Av = (np.real(np.poly(_reduced)).astype(float) * _n_sc
              if _reduced else np.array([_n_sc]))
        Av_den = (np.real(np.poly(_unused)).astype(float) * _d_sc
                  if _unused else np.array([_d_sc]))
    # sfactor (root-based spectral factorization) can raise "unpaired roots
    # remain" — most commonly for plants with high-multiplicity poles/zeros
    # exactly on (or very near) the unit circle, e.g. a ZOH-discretized
    # double integrator's D22 has a double pole at z=1, and Av = d^2*d~^2*A
    # inherits an even higher multiplicity (an 8th-order near-(z-1)^8 factor
    # for that case). np.roots on such a polynomial scatters the repeated
    # root into several numerically-distinct nearby roots, which breaks
    # sfactor's tolerance-based reciprocal-pairing (_extrpair). sfactfft
    # (FFT-cepstrum method) doesn't depend on rootfinding and handles this
    # case much better (verified: ~0.1% relative reconstruction error vs.
    # sfactor's total failure, for the double-integrator case) — same
    # fallback already used for dz_poly in _polhinf's zterm block. Some
    # imprecision here doesn't necessarily doom the final result — the real
    # validity gate is the K-stability check at the end of _polquad/
    # _polhinf, which catches the cases where the imprecision actually
    # produced an invalid controller; don't over-reject here based on Lam's
    # own reconstruction accuracy alone.
    # Lam = sfactor(Av) with MATLAB sfactor.m's discrete DEFAULT type 'd',
    # computed at the zpk level on the full RATIONAL Av including the 2k
    # origin poles that dzpk'² contributes. The joint factorization
    # (a) needs those origin poles for sfactor.m's zero/pole balance check
    # (`length(zs)+z0 == length(ps)+p0`), (b) computes the gain from the
    # stable halves of zeros and poles together (per-polynomial
    # factorization can flip its sign — the same open-loop lesson learned
    # in _polquad), and
    # (c) returns a RATIONAL Lam = Lam/Lam_den whose denominator is the
    # stable half of whatever part of A_den did not cancel above.
    # (An earlier attempt flipped only the sfactor type to 'd' while
    # keeping the rest of the chain's compensating unpadded-'z' conventions
    # — a net regression; this version converts the whole chain at once.)
    from directsd.polynomial.spectral import (
        _sfactor_lti_scipy as _sfactor_rat, sfactfft as _sfactfft_lam,
    )
    Lam_den = np.array([1.0])
    try:
        _zs_av = np.roots(Av) if len(Av) > 1 else np.array([], dtype=complex)
        _ps_av = np.concatenate([
            np.roots(Av_den) if len(Av_den) > 1 else np.array([], dtype=complex),
            np.zeros(2 * k_d, dtype=complex),      # origin poles from dzpk'²
        ])
        _kg_av = float(np.real(Av[0])) / float(np.real(Av_den[0]))
        _, _fs0 = _sfactor_rat(sig.ZerosPolesGain(_zs_av, _ps_av, _kg_av), 'd')
        Lam = np.real(np.poly(_fs0.zeros)).astype(float) * float(np.real(_fs0.gain))
        Lam_den = np.real(np.poly(_fs0.poles)).astype(float)
    except Exception:
        try:
            # FFT-cepstrum fallback (no rootfinding — handles the
            # high-multiplicity unit-circle case); polynomial-only, so
            # pair it with a separately-factored stable half of Av_den when
            # one survived the cancellation above.
            _res = _sfactfft_lam(np.real(Av).astype(float), ftype='d')
            Lam = np.real(_res[0] if isinstance(_res, tuple) else _res).astype(float)
            if len(Av_den) > 1:
                _, _fs0d = _sfactor_rat(sig.ZerosPolesGain(
                    np.roots(Av_den), np.array([], dtype=complex),
                    float(np.real(Av_den[0]))), 'd')
                Lam_den = (np.real(np.poly(_fs0d.zeros)).astype(float)
                           * float(np.real(_fs0d.gain)))
        except Exception:
            # Both methods failed outright (not just imprecise) — previously
            # silently substituted Lam=[sqrt(Av[0])] here, a trivial constant
            # with the wrong degree and no relation to the actual spectral
            # content, corrupting na/da/nb/db/YN/YD downstream with no
            # signal anything had gone wrong. Fall back to that placeholder
            # only as a last resort now; the K-stability check downstream
            # (_polquad/_polhinf) is what actually catches invalid results.
            Lam = np.array([float(np.sqrt(max(abs(float(Av[0])), 1e-30)))])
            Lam_den = np.array([1.0])

    # Lam is the spectral factor of Av = d^2*d_cr^2*(A_num/A_den) — built
    # entirely from the *exactly known* polynomials d and A_den. Lam is
    # therefore guaranteed (by construction) to share roots with d and
    # A_den exactly, but sfactor's rootfinding-based algorithm re-derives
    # those roots independently of d's/A_den's own roots and can land them
    # at slightly different floats (e.g. 2.004 instead of the true 2.0).
    # That drift is exactly what prevents downstream _tfmin cancellations
    # (na/da, nb/db, and the final K reconstruction in _polhinf) from fully
    # collapsing a pole-zero pair that MATLAB's exact zpk-based root
    # tracking always cancels cleanly. Snap any
    # Lam root that's close to a known-exact root of d or A_den back onto
    # that exact value before rebuilding Lam.
    if len(Lam) > 1:
        _lam_roots = np.roots(Lam)
        _known_roots = []
        if len(d) > 1:
            _known_roots.extend(np.roots(d))
        if len(A_den) > 1:
            _known_roots.extend(np.roots(A_den))
        if _known_roots:
            _snapped = []
            for _rt in _lam_roots:
                _best, _best_dist = None, 1e-3 * (1.0 + abs(_rt))
                for _kt in _known_roots:
                    _dist = abs(_rt - _kt)
                    if _dist < _best_dist:
                        _best, _best_dist = _kt, _dist
                _snapped.append(_best if _best is not None else _rt)
            _lead = float(np.real(Lam[0]))
            Lam = np.real(np.poly(_snapped)).astype(float) * _lead

    # Same snap for Lam_den (its roots are the stable half of A_den's).
    if len(Lam_den) > 1 and len(A_den) > 1:
        _ld_roots = np.roots(Lam_den)
        _ade_roots = np.roots(A_den)
        _snapped_d = []
        for _rt in _ld_roots:
            _best, _best_dist = None, 1e-3 * (1.0 + abs(_rt))
            for _kt in _ade_roots:
                _dist = abs(_rt - _kt)
                if _dist < _best_dist:
                    _best, _best_dist = _kt, _dist
            _snapped_d.append(_best if _best is not None else _rt)
        Lam_den = (np.real(np.poly(_snapped_d)).astype(float)
                   * float(np.real(Lam_den[0])))

    # ── Basic controller a0·n + b0·d = 1 (ynyd.m line 39) ───────────────────
    n_pol = Poln(n, 'z'); d_pol = Poln(d, 'z'); one = Poln([1.0], 'z')
    try:
        a0_p, b0_p, _, _ = dioph(n_pol, d_pol, one)
        a0 = _r(a0_p.coef); b0 = _r(b0_p.coef)
    except Exception:
        a0 = np.array([1.0]); b0 = np.array([0.0])

    # ── L1, L2, L  (ynyd.m lines 42-70) ──────────────────────────────────────
    # L1 = dzpk'²·B'/Lam' with every conjugate PADDED (@zpk/ctranspose.m
    # adds origin ζ-factors for the relative-degree difference):
    # dzpk'² = rev(d)²/ζ^{2k}, B' = _z2zeta(B), Lam' = _z2zeta(Lam).
    # Lam is improper by exactly 2k (deg Lam − deg Lam_den = 2k when A is
    # balanced), so the ζ^{2k} that _z2zeta(Lam) puts into Lam''s
    # denominator cancels the explicit ζ^{2k} from dzpk'² — but only if both
    # are actually tracked; dropping them asymmetrically was an
    # unpadded-conjugate root cause of the same class of bug throughout.
    B_cr_n, B_cr_d = _z2zeta(B_num, B_den)
    LamC_n, LamC_d = _z2zeta(Lam, Lam_den)
    _zeta_2k = np.concatenate([[1.0], np.zeros(2 * k_d)])   # ζ^{2k}
    L1_n_raw = np.polymul(np.polymul(d_cr_sq, B_cr_n), LamC_d)
    L1_d_raw = np.polymul(np.polymul(_zeta_2k, B_cr_d), LamC_n)
    L1_n_raw, L1_d_raw = _cancel_origin_tf(L1_n_raw, L1_d_raw)
    L1_n, L1_d = Minreal.tf(L1_n_raw, L1_d_raw)

    # L2 = a0·Lam/dzpk  (ynyd.m lines 44-46: zero when a0 == 0)
    if np.linalg.norm(a0) > np.finfo(float).eps:
        L2_n, L2_d = Minreal.tf(np.polymul(a0, Lam), np.polymul(d, Lam_den))
    else:
        L2_n, L2_d = np.array([0.0]), np.array([1.0])

    # Snap any near-unit-circle root of L1_d/L2_d onto |z|=1 exactly:
    # for integrator plants, d has an exact root at |z|=1, and d_cr_sq/Lam
    # inherit imprecise near-1 images of it here (~1e-4 off) well before the
    # later dLp/dLm/dL0 classification — snapping only there (as done below)
    # doesn't reach L1_d/L2_d, which R1/YN/YD use directly, silently
    # reintroducing the same imprecision downstream.
    def _snap_unit_roots(poly, tol=5e-4):
        if len(poly) <= 1:
            return poly
        rts = np.roots(poly)
        near = np.abs(np.abs(rts) - 1.0) < tol
        if not np.any(near):
            return poly
        rts[near] = rts[near] / np.abs(rts[near])
        return np.real(np.poly(rts)).astype(float) * float(np.real(poly[0]))

    L1_d = _snap_unit_roots(L1_d)
    L2_d = _snap_unit_roots(L2_d)

    L_num_r, L_den_r = _tfsub(L1_n, L1_d, L2_n, L2_d)
    L_n, L_d_min = Minreal.tf(L_num_r, L_den_r)

    # Restore poles of L that may have cancelled numerically
    _pL1 = list(np.roots(L1_d)) if len(L1_d) > 1 else []
    _pL2 = list(np.roots(L2_d)) if len(L2_d) > 1 else []
    _pL2x = []
    _used = [False] * len(_pL1)
    for _p in _pL2:
        _hit = False
        for _j, _q in enumerate(_pL1):
            if not _used[_j] and abs(_p - _q) < 1e-4 * (1.0 + abs(_p)):
                _used[_j] = True; _hit = True; break
        if not _hit:
            _pL2x.append(_p)
    _pL = _pL1 + _pL2x
    if _pL:
        L_d = np.real(np.poly(_pL)).astype(float) * float(np.real(L_d_min[0]))
        _min_rts = list(np.roots(L_d_min)) if len(L_d_min) > 1 else []
        _umask   = [False] * len(_min_rts)
        _extra   = []
        for _p in _pL:
            _hit = False
            for _j, _q in enumerate(_min_rts):
                if not _umask[_j] and abs(_p - _q) < 1e-4 * (1.0 + abs(_p)):
                    _umask[_j] = True; _hit = True; break
            if not _hit:
                _extra.append(_p)
        if _extra:
            L_n = np.real(np.polymul(L_n, np.poly(_extra))).astype(float)
    else:
        L_d = L_d_min

    nL = _r(_strip_lz(L_n))
    if len(nL) == 0 or np.all(np.abs(nL) < 1e-30):
        nL = b0 if np.linalg.norm(b0) > 1e-10 else np.array([1.0])

    # ── dL → dLp (|r|≥1, stable+neutral) and dLm (|r|<1) (ynyd.m 76-77) ─────
    # MATLAB: dLp = stable*dL0 (neutral merged into dLp); dLm = unstable only.
    #
    # _tol_n widened from 1e-4 to 5e-4: for plants with an
    # integrator (e.g. F=1/(5s^2+s) in the L2-tracking example), D22's own
    # denominator has an *exact* root at zeta=1, but accumulated floating-point
    # error through Av/Lam/L construction can push the corresponding root of
    # dL to ~1e-4 away from the unit circle by the time factor() classifies
    # it — just past the old 1e-4 tolerance, so it landed in dLm (antistable)
    # instead of dL0 (neutral). This silently corrupted the AN/CN/AD/CD
    # construction feeding diophsys downstream, producing a solvability-
    # condition violation. Also snap any root
    # classified as neutral back onto the unit circle exactly (preserving
    # its phase) — the misclassification tolerance fix alone still left a
    # marginally-unstable K in one tested case; snapping removes the residual
    # ~1e-5 error that caused that.
    rts_dL   = np.roots(L_d) if len(L_d) > 1 else np.array([], dtype=complex)
    _tol_n   = 5e-4
    # Also force near-real roots to be exactly real: the magnitude-only
    # snap below fixes a root like
    # 1.0000004 (off the unit circle but genuinely real), but a genuinely
    # *repeated* real root can instead get numerically smeared into a
    # near-conjugate pair like 1.0+3.8e-8j / 1.0-3.8e-8j — already at |z|≈1
    # (so the magnitude snap is a no-op) but not exactly real, which then
    # corrupts downstream root-matching (coprime/diophsys) expecting a
    # clean repeated real root, not a spurious near-real complex pair.
    # Found via _polquad's open-loop branch (sd2dof).
    _im_tol_real = 1e-6
    _real_mask = np.abs(np.imag(rts_dL)) < _im_tol_real * (1.0 + np.abs(rts_dL))
    rts_dL = np.where(_real_mask, np.real(rts_dL), rts_dL)
    dLp_rts  = rts_dL[np.abs(rts_dL) > 1.0 - _tol_n]   # stable + neutral (|d|≥1)
    dLm_rts  = rts_dL[np.abs(rts_dL) <= 1.0 - _tol_n]   # unstable only (|d|<1)
    dL0_rts  = rts_dL[(np.abs(rts_dL) >= 1.0 - _tol_n) & (np.abs(rts_dL) <= 1.0 + _tol_n)]  # neutral (|d|≈1)
    if dL0_rts.size > 0:
        dL0_rts = dL0_rts / np.abs(dL0_rts)   # snap onto |z|=1 exactly, keep phase
        # dLp includes dL0's roots too (MATLAB merges dL0 into dLp) — snap
        # those specific entries in dLp_rts to match, by nearest-value replace.
        for _r0 in dL0_rts:
            _dists = np.abs(dLp_rts - _r0)
            if _dists.size > 0 and _dists.min() < 10 * _tol_n:
                dLp_rts[np.argmin(_dists)] = _r0
    scl      = float(np.real(L_d[0]))
    dLp      = np.real(np.poly(dLp_rts)).astype(float) * scl if len(dLp_rts) > 0 else np.array([scl])
    dLm      = np.real(np.poly(dLm_rts)).astype(float) if len(dLm_rts) > 0 else np.array([1.0])
    dL0      = np.real(np.poly(dL0_rts)).astype(float) if len(dL0_rts) > 0 else np.array([1.0])

    # Rebuild L_d itself from the snapped root sets:
    # the dL0 snap above only fixed dLp/dLLam's precision, but R1/YN/YD below
    # use L_d and L1_d directly (not via dLLam) — they were still built from
    # the *unsnapped* L_d, silently reintroducing the same ~1e-4 imprecision
    # this fix is meant to remove. Rebuilding L_d from the corrected roots
    # keeps every downstream use consistent with the same precision.
    if len(rts_dL) > 0:
        L_d = np.real(np.poly(np.concatenate([dLp_rts, dLm_rts]))).astype(float) * scl

    # ── na = d/dLLam, nb = n/dLLam  (ynyd.m lines 80-81) ─────────────────────
    # dLLam = dLp·Lam is RATIONAL (Lam = Lam/Lam_den), so
    # na/da = d·Lam_den/(dLp·Lam) and nb/db = n·Lam_den/(dLp·Lam).
    dLLam_n = np.polymul(dLp, Lam)
    na, da  = Minreal.tf(np.polymul(d, Lam_den), dLLam_n)
    nb, db  = Minreal.tf(np.polymul(n, Lam_den), dLLam_n)

    # ── R1 = dL·L1; YN = na·R1; YD = dLm·db/d − R1·nb  (ynyd.m 84-101) ──────
    R1_n, R1_d = Minreal.tf(np.polymul(L_d, L1_n), L1_d)

    YN_n, YN_d = Minreal.tf(np.polymul(na, R1_n), R1_d)
    if len(YN_d) <= 1:
        sc = float(YN_d[0]) if abs(float(YN_d[0])) > 1e-30 else 1.0
        YN = _r(_strip_lz(YN_n / sc))
    else:
        YN = _r(_strip_lz(np.polydiv(YN_n, YN_d)[0]))
    if len(YN) == 0:
        YN = np.array([0.0])

    _num_yd = np.polymul(np.polymul(dLm, db), R1_d)
    _den_yd = np.polymul(np.polymul(R1_n, nb), d)
    nYD_r, dYD_r = Minreal.tf(_tfpadd(_num_yd, _den_yd, sign=-1.0), np.polymul(d, R1_d))
    if len(dYD_r) <= 1:
        YD = nYD_r / float(dYD_r[0]) if len(dYD_r) == 1 else nYD_r
    else:
        YD = _r(np.polydiv(nYD_r, dYD_r)[0])
    YD = _r(_strip_lz(YD))
    if len(YD) == 0:
        YD = np.array([0.0])

    return {
        'n': n, 'd': d, 'a0': a0, 'b0': b0, 'Lam': Lam, 'Lam_den': Lam_den,
        'na': na, 'da': da, 'nb': nb, 'db': db,
        'dLp': dLp, 'dLm': dLm, 'dL0': dL0, 'nL': nL,
        'YN': YN, 'YD': YD,
        'L1_n': L1_n, 'L1_d': L1_d, 'L2_n': L2_n, 'L2_d': L2_d,
        'L_n': L_n, 'L_d': L_d,
        'dLCancel': np.array([1.0]),
    }


def polhinf(A, B, E, D22, P_cancel=None):
    """
    Polynomial H∞ optimisation for a sampled-data system.

    Solves the Kwakernaak-Saeki equations:
        sigma * sigma~ = lam^2 * dz2 - phi2
        b0 * Q + P * sigma * q0 = recip(P) * c0

    Parameters
    ----------
    A, B, E : (num, den) tuples or Poln
        Quadratic cost coefficients (z-domain rational functions).
    D22 : (num, den) tuple
        Discrete plant model.
    P_cancel : array-like, optional
        Continuous-time poles that must cancel in the controller.

    Returns
    -------
    K : (num_d, den_d) tuple, or a list of two such tuples
        In the non-generic case (MATLAB polhinf.m: nonGen = iscell(P)),
        two equally-optimal controllers exist; K is then `[K0, K1]`
        instead of a single (num,den) tuple — mirrors MATLAB's own
        `K = {K, K1}` cell return (see dhinf's identical convention).
    lam : float
        Optimal H∞ cost.
    """
    K_tf, lam, _ = _polhinf(
        _ensure_tf(A), _ensure_tf(B), _ensure_tf(E), _ensure_tf(D22),
        P_cancel,
    )
    return K_tf, lam


def _polhinf(A_tf, B_tf, E_tf, D22_tf, P_cancel=None, zpk_in=None):
    """
    Internal polhinf — works with (num, den) rational-function tuples.

    Port of polhinf.m + hinfbisec.m / hinffiter.m / hinfred.m (K. Polyakov).

    Algorithm:
      1. ynyd:  d, n from D22; Lam = sfactor(d^2*d~^2*A); basic ctrl (a0,b0);
                L1, L2, L; factor dL → dLp, dLm; na, da, nb, db; YN, YD.
      2. zterm: Z = E - B*B~/A; dz = sfactor(dZ); phi2 (normalised numerator).
      3. FY = sfactor(Y*Y') → (chi, eta).
      4. Bisection (hinfbisec) → lam, P, Q, sigma.
      5. Extract controller: K = Q/(P*...).

    zpk_in : optional (A_z, B_z, E_z) triple of zpk.zpk.Zpk objects —
    root-list forms of A/B/E carried exactly from the coefficient builder
    (dhinf). When absent they are derived from the coefficient tuples.
    """
    # Cast all inputs to real (matrix-exponential helpers can produce near-complex outputs)
    def _to_real(arr):
        return np.real(np.asarray(arr)).astype(float).ravel()

    D22_num, D22_den = _to_real(D22_tf[0]), _to_real(D22_tf[1])
    A_tf = (_to_real(A_tf[0]), _to_real(A_tf[1]))
    B_tf = (_to_real(B_tf[0]), _to_real(B_tf[1]))
    E_tf = (_to_real(E_tf[0]), _to_real(E_tf[1]))

    # n, d, Lam, a0, b0 are computed by the shared _ynyd() call below
    # (Step 4b) — the old inline Step 1/Step 2 duplication has been removed.

    # ------------------------------------------------------------------
    # Step 3 — zterm.m ported at the ROOT-LIST (zpk) level:
    #   BBA = minreals(B*B'/A);  Z = sumzpk(E, -BBA);  Z = symmetr(Z);
    #   Z = minreals(Z, 1e-3);  remove z=1 pole/zero pairs;
    #   dz = sfactor(±dZ);  phi2 = delzero(K*nZ);  phi2/=dz(1)^2; dz/=dz(1).
    # Cancellation happens by EXACT root matching on root lists — the
    # coefficient-array versions (both the one-shot cross-multiplication
    # and the staged tf_symmetric reduction) fail to cancel Z down to its
    # true minimal degree because np.roots smears the re-derived roots of
    # the high-degree products; the resulting inflated dz2/phi2 sent the
    # solvers to a spurious boundary optimum (lam = sqrt(max Z)) via the
    # non-generic exit. Proven root cause by running the original
    # zterm.m/polhinf.m in an independent Octave re-derivation
    # (Reports/octave_harness/).
    # ------------------------------------------------------------------
    from directsd.polynomial.spectral import sfactfft as _sfactfft

    if zpk_in is not None:
        A_z, B_z, E_z = zpk_in
    else:
        A_z = Zpk.from_tf(A_tf[0], A_tf[1])
        B_z = Zpk.from_tf(B_tf[0], B_tf[1])
        E_z = Zpk.from_tf(E_tf[0], E_tf[1])

    _BBA_raw = B_z * B_z.conj_dt() / A_z
    try:
        BBA_z = _BBA_raw.minreals()
    except ValueError:
        # not exactly reciprocal-symmetric (smeared upstream roots) —
        # degrade to plain pair cancellation rather than failing
        BBA_z = _BBA_raw.minreal(1e-3)
    Z_z = E_z.zsum(-BBA_z)
    if abs(Z_z.k) < np.finfo(float).eps or abs(Z_z.k) < 1e-8 * abs(E_z.k):
        Z_z = Zpk([], [], 0.0)
    try:
        Z_z = Z_z.symmetr('z')
    except ValueError:
        pass
    try:
        Z_z = Z_z.minreals(1e-3)
    except ValueError:
        Z_z = Z_z.minreal(1e-3)

    # ------------------------------------------------------------------
    # zterm.m lines 36-73 — special construction with cancellation.
    # When PCancel poles (ζ-roots) survive in the direct Z, the E and
    # B·B~/A terms each carry SINGULAR parts at those poles which are
    # equal analytically but differ numerically — subtracting them the
    # direct way leaves near-singular garbage. MATLAB separates the
    # cancellable partial fractions (separss) and eliminates them by a
    # polynomial Diophantine identity instead:
    #   dCancel = gcd(dE, dLCancel·dLCancel~)
    #   [E0,E1]   = separss(E,  dCancel, 'infu')
    #   DD = dA/(dB·dB~) → delzero → [DD0,DD1] = separss(DD, dCancel,'infu')
    #   NN = nB·nB~/nA   → delzero → origin bookkeeping (zerosND)
    #   m0·dN + m1·dD0 = nN·nD0;  X1 = m1/dN
    #   Z0 = symm(X1 + NN·DD1);   Z = E1 - Z0
    # ------------------------------------------------------------------
    if P_cancel is not None and len(np.atleast_1d(P_cancel)):
        _pc = np.atleast_1d(np.asarray(P_cancel, complex)).ravel()
        _pc_nz = _pc[np.abs(_pc) > 1e-12]
        _targets = np.concatenate([_pc, 1.0 / _pc_nz])
        # dCancel = multiset intersection of E's poles with pc ∪ 1/pc
        _avail = list(_targets)
        _dCancel = []
        for _q in E_z.p:
            for _j, _t in enumerate(_avail):
                if abs(_q - _t) < 1e-4 * (1.0 + abs(_q)):
                    _dCancel.append(_q)
                    _avail.pop(_j)
                    break
        _zCancel = (_others2(Z_z.p, np.array(_dCancel, complex), tol=1e-4)[1]
                    if _dCancel else np.zeros(0))
        if len(np.atleast_1d(_zCancel)):
            from directsd.polynomial.poln import Poln as _Poln
            _E0, _E1 = _pf_split(E_z, _dCancel)
            # DD = minreal(zpk(dA, dB·dB~)) at the root-list level
            _recip = lambda r: 1.0 / r[np.abs(r) > 1e-12]
            _dB_lead2 = float(np.real(np.prod(-B_z.p))) if len(B_z.p) else 1.0
            DD_z = Zpk(A_z.p.copy(),
                       np.concatenate([B_z.p, _recip(B_z.p)]),
                       1.0 / _dB_lead2).minreal()
            # delzero on num/den root lists (origin-root bookkeeping)
            def _delzero_z(q):
                zo = int(np.sum(np.abs(q.z) < 1e-12))
                po = int(np.sum(np.abs(q.p) < 1e-12))
                return (Zpk(q.z[np.abs(q.z) >= 1e-12],
                            q.p[np.abs(q.p) >= 1e-12], q.k), zo, po)
            DD_z, _znD, _zdD = _delzero_z(DD_z)
            _DD0, _DD1 = _pf_split(DD_z, _dCancel)
            # NN = minreal(zpk(nB·nB~, nA)) at the root-list level
            _nB_lead_conj = B_z.k * (float(np.real(np.prod(-np.conj(B_z.z[np.abs(B_z.z) > 1e-12]))))
                                     if np.any(np.abs(B_z.z) > 1e-12) else 1.0)
            NN_z = Zpk(np.concatenate([B_z.z, _recip(B_z.z)]),
                       A_z.z.copy(),
                       B_z.k * _nB_lead_conj / A_z.k).minreal()
            NN_z, _znN, _zdN = _delzero_z(NN_z)
            _zerosND = _znD + _znN - _zdD - _zdN
            _nN = np.real(NN_z.k * np.poly(NN_z.z)) if len(NN_z.z) else np.array([NN_z.k])
            _dN = np.real(np.poly(NN_z.p)) if len(NN_z.p) else np.array([1.0])
            if _zerosND > 0:
                _nN = np.concatenate([_nN, np.zeros(_zerosND)])
            elif _zerosND < 0:
                _dN = np.concatenate([_dN, np.zeros(-_zerosND)])
            _nD0 = (np.real(_DD0.k * np.poly(_DD0.z)) if len(_DD0.z)
                    else np.array([_DD0.k]))
            _dD0 = np.real(np.poly(_DD0.p)) if len(_DD0.p) else np.array([1.0])
            # m0·dN + m1·dD0 = nN·nD0  →  X1 = m1/dN
            _rhs = np.polymul(_nN, _nD0)
            try:
                _m0p, _m1p, _, _ = dioph(_Poln(_dN, 'z'), _Poln(_dD0, 'z'),
                                         _Poln(_rhs, 'z'))
                _m1 = np.real(np.asarray(_m1p.coef, complex)).astype(float)
                X1_z = Zpk.from_tf(_m1, np.array([1.0]))
                _pX1 = (np.concatenate([NN_z.p, np.zeros(-_zerosND, complex)])
                        if _zerosND < 0 else NN_z.p.copy())
                X1_z = Zpk(X1_z.z, _pX1, X1_z.k)
                Z0_z = X1_z.zsum(NN_z * _DD1).minreal(1e-3)
                Z0_z = Z0_z.zsum(Z0_z.conj_dt()) * 0.5   # symmetrize
                Z_z = _E1.zsum(-Z0_z)
            except Exception as _zx:
                warnings.warn(f"_polhinf: zterm cancellation branch failed "
                              f"({_zx}); keeping direct Z")

    # zterm.m lines 76-88: extract poles at z=1 (cancel matching pairs)
    _tol1 = np.sqrt(np.finfo(float).eps)
    _pZ = list(Z_z.p); _zZ = list(Z_z.z)
    _n_at_1 = sum(1 for r in _pZ if abs(r - 1.0) < _tol1)
    for _ in range(_n_at_1):
        for lst in (_pZ, _zZ):
            if lst:
                _d1 = [abs(r - 1.0) for r in lst]
                _j = int(np.argmin(_d1))
                if _d1[_j] < 1e-4:
                    lst.pop(_j)
    Z_z = Zpk(np.array(_zZ, complex), np.array(_pZ, complex), Z_z.k)

    # [nZ, dZ] = tf2nd(Z); delzero(dZ); cut0 symmetrization of nZ
    nZ, dZ_full = Z_z.to_tf()
    dZs = np.trim_zeros(dZ_full, 'b')
    if len(dZs) == 0:
        dZs = np.array([1.0])
    _zeros_D = len(dZ_full) - len(dZs)
    _ell = (len(dZs) - 1) // 2
    _mid = _zeros_D + _ell
    _phi_raw = _strip_lz(nZ).copy()
    _cut0 = 2 * _mid - (len(_phi_raw) - 1)
    if _cut0 > 0 and len(_phi_raw) > _cut0:
        _phi_raw[-_cut0:] = 0.0

    # dz = sfactor(dZ) with the ± retry (zterm.m lines 100-101); sfactfft
    # kept as a last-resort fallback for high-multiplicity circle roots.
    dz_poly = np.array([1.0])
    _dz_sign = 1.0
    try:
        _sf_dz = sfactor(Poln(np.real(dZs).astype(float), 'z'), 'd')
        _dz_p = _sf_dz[0] if isinstance(_sf_dz, tuple) else _sf_dz
        dz_poly = np.real(_dz_p.coef).astype(float)
    except Exception:
        try:
            _sf_dz = sfactor(Poln(-np.real(dZs).astype(float), 'z'), 'd')
            _dz_p = _sf_dz[0] if isinstance(_sf_dz, tuple) else _sf_dz
            dz_poly = np.real(_dz_p.coef).astype(float)
            _dz_sign = -1.0
        except Exception:
            try:
                _res = _sfactfft(np.real(dZs).astype(float), ftype='d')
                dz_poly = np.real(_res[0] if isinstance(_res, tuple) else _res).astype(float)
            except Exception:
                dz_poly = np.array([1.0])

    # phi2 = delzero(K*nZ) (strip trailing origin zeros), then normalize so
    # dz is MONIC: phi2 /= dz(1)^2, dz /= dz(1)  (zterm.m lines 105-112 —
    # the monic normalization was missing from the coefficient-based port).
    phi2 = _dz_sign * _phi_raw
    phi2 = np.trim_zeros(np.trim_zeros(phi2, 'b'), 'f')
    if len(phi2) == 0:
        phi2 = np.array([0.0])
    if abs(dz_poly[0]) > 1e-300:
        phi2 = phi2 / dz_poly[0] ** 2
        dz_poly = dz_poly / dz_poly[0]

    # MATLAB polhinf.m line 32: dz2 = dz*dz' — from the returned spectral
    # factor (non-negative on the circle by construction).
    dz2_bisect_src = np.real(np.polymul(dz_poly, dz_poly[::-1])).astype(float)

    # ------------------------------------------------------------------
    # Step 4b — ynyd algorithm: L = L1 - L2, factor dL, na/da/nb/db, YN, YD.
    # Delegates to the shared _ynyd() port (also used by _polquad) so both
    # callers use the same dLp/dLm/dL0 convention (MATLAB ynyd.m: neutral
    # roots merge into dLp, not dLm).
    # ------------------------------------------------------------------
    _yd = _ynyd(A_tf, B_tf, D22_tf, P_cancel,
                zpk_in=(A_z, B_z))
    a0, b0 = _yd['a0'], _yd['b0']
    L1_n, L1_d = _yd['L1_n'], _yd['L1_d']
    L_d = _yd['L_d']
    nL = _yd['nL']
    dLp, dLm, dL0 = _yd['dLp'], _yd['dLm'], _yd['dL0']
    na, da, nb, db = _yd['na'], _yd['da'], _yd['nb'], _yd['db']
    YN, YD = _yd['YN'], _yd['YD']

    # chi/eta — faithful port of polhinf.m lines 50-52:
    #   Y  = zpk(dz, delzero(dLm), T)
    #   FY = minreal(sfactor(Y*Y'), 1e-4)
    #   [chi, eta] = tf2nd(FY)
    # (Replaces a hand-rolled reciprocal-root-matching construction that
    # produced chi = eta = 1 where the true values are e.g. chi = 27.179
    # and eta = ζ-2.5 for dhinf help-Example 2 — verified against the
    # Octave-harness ground truth.)
    _dLm_nz_rts = [r for r in (np.roots(dLm) if len(dLm) > 1 else [])
                   if abs(r) > 1e-8]        # delzero: drop z=0 roots
    _dLm_nz = (np.real(np.poly(_dLm_nz_rts)).astype(float) * float(dLm[0])
               if _dLm_nz_rts else np.array([float(dLm[0])]))
    try:
        _Y_z = Zpk.from_tf(dz_poly, _dLm_nz)
        _FY = (_Y_z * _Y_z.conj_dt()).sfactor('d').minreal(1e-4)
        chi = (np.real(_FY.k * np.poly(_FY.z)).astype(float)
               if len(_FY.z) else np.array([float(_FY.k)]))
        eta = (np.real(np.poly(_FY.p)).astype(float)
               if len(_FY.p) else np.array([1.0]))
    except Exception:
        chi = np.array([1.0])
        eta = np.array([1.0])

    # na, da, nb, db, YN, YD already computed by the shared _ynyd() call above.

    # ------------------------------------------------------------------
    # Step 5 — bisection inputs: b=dLm, q=eta*dLp, c=chi*nL, r=1
    # Extract common factors first (MATLAB polhinf.m lines 56-59):
    #   [b0,q0,c0] = triple(b0,q0,c0); [r0,q0,c0] = triple(r0,q0,c0);
    #   [q0,c0,gQ] = coprime(q0,c0);
    # ------------------------------------------------------------------
    b_coef = dLm
    q_coef = np.convolve(eta, dLp)
    c_coef = np.convolve(chi, nL)
    r_coef = np.array([1.0])
    b_coef, q_coef, c_coef = _np_triple(b_coef, q_coef, c_coef)
    r_coef, q_coef, c_coef = _np_triple(r_coef, q_coef, c_coef)
    q_coef, c_coef, gQ = _np_coprime(q_coef, c_coef)
    deg_P = len(b_coef) - 1

    # ------------------------------------------------------------------
    # Step 6 — run bisection, fall through to F-iteration on failure
    # ------------------------------------------------------------------
    def _r(x):
        return np.real(np.asarray(x)).astype(float).ravel()

    dz2_bisect = _r(dz2_bisect_src); phi2_bisect = _r(phi2)
    b_coef = _r(b_coef); q_coef = _r(q_coef)
    c_coef = _r(c_coef); r_coef = _r(r_coef)

    lam = np.inf
    P = np.array([1.0])
    Q = nL.copy()
    sigma = np.array([1.0])

    # Match MATLAB polhinf.m call order: hinfred → hinffiter → hinfbisec
    try:
        lam, P, Q, sigma, _ = _hinfred_np(
            dz2_bisect, phi2_bisect, b_coef, q_coef, c_coef, r_coef, deg_P,
        )
    except Exception as exc:
        warnings.warn(f"hinfred failed ({exc}); trying hinffiter")

    if np.isinf(lam) or np.isnan(lam):
        try:
            lam, P, Q, sigma, _ = _hinffiter_np(
                dz2_bisect, phi2_bisect, b_coef, q_coef, c_coef, r_coef, deg_P,
            )
        except Exception as exc:
            warnings.warn(f"hinffiter failed ({exc}); trying hinfbisec")

    if np.isinf(lam) or np.isnan(lam):
        try:
            lam, P, Q, sigma, _ = _hinfbisec_np(
                dz2_bisect, phi2_bisect, b_coef, q_coef, c_coef, r_coef, deg_P,
            )
        except Exception as exc:
            warnings.warn(f"hinfbisec failed ({exc}); returning basic controller")

    # Restore Q: MATLAB polhinf.m line 100 (Q = Q * gQ)
    Q = np.polymul(np.real(np.asarray(Q, float)).ravel(), gQ)

    if np.isinf(lam) or np.isnan(lam):
        K_num = a0 if np.linalg.norm(a0) > 1e-10 else np.array([1.0])
        K_den = b0 if np.linalg.norm(b0) > 1e-10 else np.array([1.0])
        sigma_tf = (np.array([1.0]), np.array([1.0]))
        return (striplz(np.real(K_num)), striplz(np.real(K_den))), float('nan'), sigma_tf

    # Non-generic case (MATLAB polhinf.m line 83: nonGen = iscell(P)) —
    # _hinfred_np/_hinfbisec_np return P as a (P0, P1) tuple instead of a
    # plain array when the non-generic branch fires; unpack it the same
    # way MATLAB's `P1 = P{2}; P = P{1};` does. K1 (from P1) is built below,
    # after K (from P), by lines 194-210 of polhinf.m.
    nonGen = isinstance(P, tuple)
    if nonGen:
        P, P1 = P

    # ------------------------------------------------------------------
    # K reconstruction (port of polhinf.m lines 165-189):
    # Simplify AN/BN/CN via triple+coprime, compute N=N0/CN, D=D0/CD,
    # then K = db*gN*N / (da*gD*D)
    # ------------------------------------------------------------------
    P_rl   = np.real(np.asarray(P, float)).ravel()
    P_recip = P_rl[::-1]                            # recip(P) = P reversed
    sig_rl  = np.real(np.asarray(sigma, float)).ravel()
    sig_dLp = np.polymul(sig_rl, np.polymul(eta, dLp))
    BN = np.real(_strip_lz(-np.polymul(chi, YN))).astype(float)
    if len(BN) == 0: BN = np.array([0.0])
    AN = np.polymul(sig_dLp, na)
    BD = np.real(_strip_lz(-np.polymul(chi, YD))).astype(float)
    if len(BD) == 0: BD = np.array([0.0])
    AD = -np.polymul(sig_dLp, nb)

    # MATLAB polhinf.m lines 165-180:
    #   CN = dLm*g; CD = CN;
    #   [AN,BN,CN] = triple(AN,BN,CN); [AD,BD,CD] = triple(AD,BD,CD);
    #   [AN,BN,gN] = coprime(AN,BN);   [AD,BD,gD] = coprime(AD,BD);
    #   N0 = -recip(P)*BN - P*AN;  N = mrdivide(N0, CN)
    #   D0 = -recip(P)*BD - P*AD;  D = mrdivide(D0, CD)
    #   K = zpk(db*gN*N, da*gD*D, T)' ← the ' (K') inverts roots: ζ → z
    # g=1 in the generic case (polhinf.m line 39: "g=[]; % better numerically!")
    CN = np.real(_strip_lz(dLm.copy())).astype(float)
    CD = np.real(_strip_lz(dLm.copy())).astype(float)
    AN, BN, CN = _np_triple(AN, BN, CN)
    AD, BD, CD = _np_triple(AD, BD, CD)
    AN, BN, gN = _np_coprime(AN, BN)
    AD, BD, gD = _np_coprime(AD, BD)

    # N0, D0 from P (with reduced AN/BN/AD/BD after triple+coprime)
    N0 = _tfpadd(-np.polymul(P_recip, BN), np.polymul(P_rl, AN), sign=-1.0)
    D0 = _tfpadd(-np.polymul(P_recip, BD), np.polymul(P_rl, AD), sign=-1.0)

    def _poly_div_quotient(num, den):
        if len(den) <= 1:
            sc = float(den[0]) if len(den) == 1 and abs(den[0]) > 1e-30 else 1.0
            return np.real(_strip_lz(num / sc)).astype(float)
        q, _r = np.polydiv(num, den)
        return np.real(_strip_lz(q)).astype(float)

    # Divide N0/CN and D0/CD — removes dLm roots, crucial for controller stability
    N = _poly_div_quotient(N0, CN)
    D = _poly_div_quotient(D0, CD)
    if len(N) == 0: N = np.array([1.0])
    if len(D) == 0: D = np.array([1.0])

    # Standard K in ζ-domain: K_ζ = db*gN*N / (da*gD*D)
    K_z_n, K_z_d = Minreal.tf(np.polymul(np.polymul(db, gN), N),
                           np.polymul(np.polymul(da, gD), D))

    K_num, K_den = _z2zeta(K_z_n, K_z_d)
    K_num = np.real(_strip_lz(K_num)).astype(float)
    K_den = np.real(_strip_lz(K_den)).astype(float)

    # ------------------------------------------------------------------
    # Second non-generic controller K1 (port of polhinf.m lines 192-210):
    # same AN/BN/CN/AD/BD/CD/gN/gD as K above, but AN and AD NEGATED, and
    # P1 instead of P — everything else (division by the same CN/CD,
    # z-domain conversion) is identical to K's construction.
    # ------------------------------------------------------------------
    K1_tf = None
    if nonGen:
        P1_rl = np.real(np.asarray(P1, float)).ravel()
        P1_recip = P1_rl[::-1]
        AN1 = -AN
        AD1 = -AD
        N0_1 = _tfpadd(-np.polymul(P1_recip, BN), np.polymul(P1_rl, AN1), sign=-1.0)
        D0_1 = _tfpadd(-np.polymul(P1_recip, BD), np.polymul(P1_rl, AD1), sign=-1.0)
        N1 = _poly_div_quotient(N0_1, CN)
        D1 = _poly_div_quotient(D0_1, CD)
        if len(N1) == 0: N1 = np.array([1.0])
        if len(D1) == 0: D1 = np.array([1.0])
        K1_z_n, K1_z_d = Minreal.tf(np.polymul(np.polymul(db, gN), N1),
                                    np.polymul(np.polymul(da, gD), D1))
        K1_num, K1_den = _z2zeta(K1_z_n, K1_z_d)
        K1_num = np.real(_strip_lz(K1_num)).astype(float)
        K1_den = np.real(_strip_lz(K1_den)).astype(float)
        K1_tf = (striplz(K1_num), striplz(K1_den))

    # ------------------------------------------------------------------
    # Sigma — polhinf.m lines 108-111:
    #   Fsigma = zpk(recip(P)*chi, sigma*eta, T);  Sigma = Fsigma*Fsigma'
    # The GAINS of chi/sigma/P carry through — an earlier version assumed
    # chi = eta = 1 and normalized P/sigma to monic, which scaled Sigma by
    # an arbitrary constant; sdh2hinf's blend rho*A2 + (1-rho)*Ainf*Sigma
    # then weighted the H∞ term wrongly (Sigma's roots matched MATLAB but
    # its gain was off ×30 on demo_h2hinf).
    P_r = np.real(np.asarray(P, dtype=float)).ravel()
    s_r = np.real(np.asarray(sigma, dtype=float)).ravel()
    if np.linalg.norm(s_r) < 1e-10 or abs(lam) < 1e-8:
        # Degenerate spectral factor → Sigma = 1 (no weighting)
        sigma_tf = (np.array([1.0]), np.array([1.0]))
    else:
        _Fs = Zpk.from_tf(np.polymul(P_r[::-1], chi), np.polymul(s_r, eta))
        sigma_tf = (_Fs * _Fs.conj_dt()).to_tf()

    K_tf = (striplz(K_num), striplz(K_den))
    # Non-generic case: MATLAB polhinf.m line 209 returns K = {K, K1} (a
    # cell of two equally-optimal controllers). Mirrored here as a plain
    # `list` of two (num,den) tuples — distinguishable by type from the
    # ordinary single-controller (num,den) tuple return, matching MATLAB's
    # own iscell(K) duck-typing.
    if nonGen:
        K_tf = [K_tf, K1_tf]
    return K_tf, float(abs(lam)), sigma_tf


# ---------------------------------------------------------------------------
# Core Hinf bisection — port of hinfbisec.m (K. Polyakov)
# ---------------------------------------------------------------------------

def _hinfbisec_np(aa, gg, b0, q0, c0, r0, deg_P, tol=1e-6):
    """
    Bisection for polynomial Hinf equations (numpy coefficient arrays).

    Solve:  sigma*sigma~ = lam^2*aa - gg
            b0*r0*Q + P*sigma*q0 = recip(P)*c0
    All arrays in descending order (highest power first).

    Port of hinfbisec.m (K. Polyakov).

    Returns
    -------
    lam, P, Q, sigma : np.ndarray (descending)
    err : float
    """
    from directsd.linalg.matrices import toep
    from directsd.polynomial.spectral import sfactfft

    aa = np.asarray(aa, dtype=float).ravel()
    gg = np.asarray(gg, dtype=float).ravel()
    b = np.convolve(np.asarray(b0, float).ravel(),
                    np.asarray(r0, float).ravel())
    q = np.asarray(q0, dtype=float).ravel()
    c = np.asarray(c0, dtype=float).ravel()

    deg_aa = _pdeg(aa); deg_gg = _pdeg(gg)
    deg_b0 = _pdeg(np.asarray(b0, float))
    deg_q = _pdeg(q); deg_c = _pdeg(c)
    deg_si = max(deg_aa, deg_gg) // 2
    deg_all = deg_P + max(deg_si + deg_q, deg_c)
    deg_Q = max(deg_all - _pdeg(b), 0)
    mP = deg_P + 1
    mQ = max(deg_Q + 1, 1)
    n2 = deg_all + 1

    xi0 = _xi0_init(aa, gg)

    # Check non-generic case at xi0 (MATLAB hinfbisec.m lines 76-88 — same
    # {P0,P1} pair as hinfred.m's non-generic branch).
    ag2_0 = _sumpol2_c(aa, -(xi0 ** 2) * gg)
    sigma0 = _sfact_coef(ag2_0)
    if not (np.any(np.isnan(sigma0)) or np.any(np.isinf(sigma0))):
        P0, Q0 = _dioph2_np(np.convolve(sigma0, q), -(xi0 * c), b, deg_P)
        rts0 = np.roots(P0) if len(P0) > 1 else np.array([])
        if len(rts0) == 0 or np.all(np.abs(rts0) < 1.0):
            P1, _Q1 = _dioph2_np(np.convolve(-sigma0, q), -(xi0 * c), b, deg_P)
            lam = 1.0 / xi0
            return lam, (P0, P1), lam * Q0, lam * sigma0, 0.0
    else:
        sigma0 = _sfact_coef(aa)

    # Bisection main loop
    xi_min, xi_max = 0.0, xi0
    P = P0.copy(); Q_out = Q0.copy(); sigma = sigma0.copy()
    xi = xi0 * 0.5

    for _ in range(100):
        if xi_max - xi_min < tol:
            break
        xi = (xi_min + xi_max) / 2.0

        ag2 = _sumpol2_c(aa, -(xi ** 2) * gg)
        sigma = _sfact_coef(ag2)

        sigma_q = np.convolve(sigma, q)
        JQ = toep(b, n2, mQ)
        JP_full = (toep(sigma_q, n2, mP)
                   - np.fliplr(toep(xi * c, n2, mP)))

        A_mat = np.real(np.hstack([JQ, JP_full[:, :-1]]))
        rhs = np.real(-JP_full[:, -1])

        try:
            X, _, _, _ = la.lstsq(A_mat, rhs, cond=None)
        except Exception:
            xi_max = xi
            continue

        Q_asc = X[:mQ]
        P_inner = X[mQ:]
        P = np.flipud(np.concatenate([P_inner, [1.0]]))
        Q_out = np.flipud(Q_asc)

        rts_P = np.roots(P) if len(P) > 1 else np.array([])
        if len(rts_P) > 0 and np.any(np.abs(rts_P) > 1.0):
            xi_max = xi
        else:
            xi_min = xi

    if xi < 1e-12:
        return np.inf, P, Q_out, sigma, float('nan')
    lam = 1.0 / xi

    # Port of MATLAB hinfbisec.m "remove" step:
    # Find the root of P closest to ±1, remove it (unit-circle common root
    # with Q), then rebuild P and Q without that root.  The degree of P drops
    # by 1 to match what hinfred/hinffiter return.
    rts_P = np.roots(P) if len(P) > 1 else np.array([])
    if len(rts_P) > 0:
        e1 = np.min(np.abs(rts_P - 1.0))
        e2 = np.min(np.abs(rts_P + 1.0))
        errP = min(e1, e2)
        rts0 = 1.0 if e1 <= e2 else -1.0

        rts_Q = np.roots(Q_out * lam) if len(Q_out) > 1 else np.array([])
        errQ = np.min(np.abs(rts_Q - rts0)) if len(rts_Q) > 0 else 0.0
        if max(errP, errQ) > 0.1:
            return np.inf, P, Q_out * lam, sigma * lam, float('nan')

        # Remove the unit-circle root from P
        idx_P = int(np.argmin(np.abs(rts_P - rts0)))
        rts_P_red = np.delete(rts_P, idx_P)
        P = (np.real(float(P[0]) * np.poly(rts_P_red)).astype(float)
             if len(rts_P_red) > 0 else np.array([float(P[0])]))

        # Remove the corresponding root from Q
        if len(rts_Q) > 0:
            idx_Q = int(np.argmin(np.abs(rts_Q - rts0)))
            rts_Q_red = np.delete(rts_Q, idx_Q)
            Q_lam_lead = float((Q_out * lam)[0])
            Q_out = (np.real(Q_lam_lead * np.poly(rts_Q_red)).astype(float) / lam
                     if len(rts_Q_red) > 0 else np.array([Q_lam_lead / lam]))

        # Check strict antistability of reduced P
        rts_P_fin = np.roots(P) if len(P) > 1 else np.array([])
        if len(rts_P_fin) > 0 and np.any(np.abs(rts_P_fin) > 1.0):
            return np.inf, P, Q_out * lam, sigma * lam, float('nan')

    return abs(lam), P, Q_out * lam, sigma * lam, 0.0


# ---------------------------------------------------------------------------
# Core Hinf Newton-Raphson solver — port of hinfred.m (K. Polyakov)
# ---------------------------------------------------------------------------

def _hinfred_np(aa, gg, b0, q0, c0, r0, deg_P, tol=1e-6):
    """
    Newton-Raphson solver for polynomial Hinf equations (port of hinfred.m).

    Solve:
        sigma*sigma~ = lam^2*aa - gg
        b0*r0*Q + P*sigma*q0 = recip(P)*c0
    All arrays in descending order (highest power first).
    Returns (lam, P, Q, sigma, err).  P has degree deg_P-1 (unit-circle root
    already absorbed into the algebraic structure).
    """
    from directsd.linalg.matrices import toep
    import scipy.linalg as la

    aa = np.asarray(aa, dtype=float).ravel()
    gg = np.asarray(gg, dtype=float).ravel()
    b0_arr = np.asarray(b0, dtype=float).ravel()
    r0_arr = np.asarray(r0, dtype=float).ravel()
    b = np.convolve(b0_arr, r0_arr)
    q = np.asarray(q0, dtype=float).ravel()
    c = np.asarray(c0, dtype=float).ravel()

    deg_aa = _pdeg(aa); deg_gg = _pdeg(gg)
    deg_b0 = _pdeg(b0_arr)
    deg_q  = _pdeg(q);  deg_c  = _pdeg(c)
    deg_r  = _pdeg(r0_arr)

    delA  = max(deg_aa, deg_gg) // 2
    delB  = deg_b0                        # = deg_P
    delC  = max(delA + deg_q, deg_c)
    n1    = delA + 1
    n2    = deg_b0 + delC
    mSigma = delA + 1                     # number of sigma coefficients
    mP    = delB                          # P has mP coefficients (degree mP-1)
    mQ    = max(delC - deg_r, 1)          # number of Q coefficients
    mQP   = mQ + mP

    if mP == 0:
        # dLm is scalar (constant 1): P is also a constant.  Do NOT return early — fall
        # through to the non-generic check which calls _dioph2_np(..., deg_X=0), giving
        # P=[1.] (monic scalar).  This is the correct solution when L denominator poles
        # are entirely anti-stable (MATLAB ynyd.m / polhinf.m convention).
        pass

    # Ca2, Cgam2 — positive-power half of aa/gg (padded to n1)
    def _pos_half(poly, n):
        half = len(poly) // 2 + 1
        raw  = poly[:half] if len(poly) >= half else np.concatenate([np.zeros(half - len(poly)), poly])
        return np.concatenate([np.zeros(max(0, n - len(raw))), raw[:n]])

    Ca2   = _pos_half(aa, n1)    # descending, length n1
    Cgam2 = _pos_half(gg, n1)

    # Initial sigma = sfactfft(aa, 'd')
    sigma = _sfact_coef(aa)
    _vs = sigma[::-1]            # ascending
    vecSigma = np.zeros(mSigma)
    vecSigma[:min(len(_vs), mSigma)] = _vs[:min(len(_vs), mSigma)]

    xi0 = _xi0_init(aa, gg)

    # Non-generic check (MATLAB hinfred.m lines 76-84).
    # When xi0 is very large (gg≈0 case), MATLAB gets xi0=Inf → NaN propagation →
    # non-generic check fails naturally. Python avoids NaN by clamping, so we
    # must explicitly skip the check when the ratio (gg/aa) is essentially zero —
    # equivalent to xi0 being huge, since xi0 = 1/sqrt(max(ratio)+1e-300).
    _xi0_degenerate = xi0 > 1e8
    if not _xi0_degenerate:
        ag2_0  = _sumpol2_c(aa, -(xi0**2) * gg)
        sigma0 = _sfact_coef(ag2_0)
        # Guard against a numerically-degenerate non-generic candidate: when xi0 sits
        # extremely close to the true critical value, ag2_0 = aa - xi0^2*gg can be
        # essentially all floating-point noise (its intended value is exactly zero
        # there), and _sfact_coef then returns a finite-but-meaningless near-zero
        # sigma0 rather than raising. Trusting that garbage sigma0 picks a bogus P0
        # and can send the "non-generic" shortcut down the wrong branch. Require
        # sigma0's magnitude to be non-negligible relative to ag2_0 before trusting it;
        # otherwise skip the shortcut and let the full Newton-Raphson iteration below
        # (which doesn't depend on this single sigma0 value) find the real solution.
        _sigma0_ok = (
            not (np.any(np.isnan(sigma0)) or np.any(np.isinf(sigma0)))
            and np.linalg.norm(sigma0) > 1e-6 * np.sqrt(np.linalg.norm(ag2_0) + 1e-300)
        )
        if _sigma0_ok:
            P0, Q0 = _dioph2_np(np.convolve(sigma0, q), -(xi0 * c), b, mP)
            rts0 = np.roots(P0) if len(P0) > 1 else np.array([])
            if len(rts0) == 0 or np.all(np.abs(rts0) < 1.0):
                # Non-generic case (MATLAB hinfred.m lines 97-110): a second
                # candidate P1 solves the SAME Diophantine equation with
                # sigma0 negated (b0*r0*Q + P*(-sigma0)*q0 = recip(P)*c0);
                # both P0 and P1 give equally-optimal controllers once
                # threaded through _polhinf's nonGen branch (P returned as
                # a (P0, P1) tuple; the caller detects tuple-vs-array to
                # tell nonGen from the ordinary single-P case — MATLAB's
                # own iscell(P) check).
                # An earlier over-triggering concern (this branch firing for
                # demo_dhinf's documented *generic* Example 2) was traced to
                # an upstream root-list bug, since fixed — no longer
                # observed; Example 2 takes the main Newton-Raphson path
                # below as expected.
                P1, _Q1 = _dioph2_np(np.convolve(-sigma0, q), -(xi0 * c), b, mP)
                lam = 1.0 / xi0
                return lam, (P0, P1), lam * Q0, lam * sigma0, 0.0

    # Saeki's generalized eigenvalue initial guess
    sigma_q = np.convolve(sigma, q)
    TP  = toep(sigma_q, n2, mP)
    TQ  = toep(b,       n2, mQ)
    TwP = np.fliplr(toep(c, n2, mP))   # = MATLAB hank(c, n2, mP)

    TA = np.real(np.hstack([TQ, TP]))
    TB = np.real(np.hstack([np.zeros((n2, mQ)), TwP]))

    # Row-compress via QR (port of MATLAB compress(...,'row'))
    try:
        Q_mat, _ = la.qr(TA, mode='economic')
        TA_c = Q_mat.T @ TA
        TB_c = Q_mat.T @ TB
    except Exception:
        TA_c = TA; TB_c = TB
    TA_sq = TA_c[:mQP, :]
    TB_sq = TB_c[:mQP, :]

    # Generalized eigenvalue → initial xi
    xi = 0.95 * xi0
    try:
        eigvals = la.eig(TA_sq, TB_sq)[0]
        fin     = eigvals[np.isfinite(eigvals)]
        if len(fin) > 0:
            xi = float(np.min(np.abs(fin)))
    except Exception:
        pass
    if xi > xi0 or xi < 1e-15:
        xi = 0.95 * xi0

    # Solve linear system for initial vecQP (monic P: last coeff = 1)
    A_init = TA_sq - xi * TB_sq
    try:
        vecQP, _, _, _ = la.lstsq(A_init[:, :mQP-1], -A_init[:, mQP-1], cond=None)
    except Exception:
        vecQP = np.zeros(mQP - 1)

    indQ_sl = slice(0,    mQ)
    indP_sl = slice(mQ,   mQ + mP - 1)   # mP-1 inner P coefficients

    # DeStabilize P: map roots outside unit circle to inside (so all |r|<1)
    P_asc = np.concatenate([vecQP[indP_sl], [1.0]])  # ascending, mP elements
    P_cur = P_asc[::-1]                               # descending, monic
    rtsP  = np.roots(P_cur) if len(P_cur) > 1 else np.array([])
    if len(rtsP) > 0:
        outside = np.abs(rtsP) > 1.0
        if np.any(outside):
            rtsP[outside] = 1.0 / rtsP[outside]
            P_cur = np.real(np.poly(rtsP)).astype(float)
            P_asc = P_cur[::-1]
            vecQP[indP_sl] = P_asc[:-1]

    # Build initial state vector X = [xi, vecSigma, vecQP]
    X = np.concatenate([[xi], vecSigma, vecQP])
    i_xi    = 0
    i_sig   = slice(1,          1 + mSigma)
    i_q     = slice(1 + mSigma, 1 + mSigma + mQ)
    i_p     = slice(1 + mSigma + mQ, 1 + mSigma + mQ + mP - 1)

    # Newton-Raphson main loop (up to 200 iterations)
    old_xi = np.inf
    reset  = False

    for _it in range(200):
        xi_v    = float(X[i_xi])
        vs_v    = X[i_sig]              # ascending sigma coefficients
        vq_v    = X[i_q]               # ascending Q coefficients
        vp_v    = np.concatenate([X[i_p], [1.0]])   # ascending P, mP elements

        sigma_v = vs_v[::-1]           # descending
        P_v     = vp_v[::-1]           # descending, monic

        if abs(old_xi - xi_v) <= tol * (abs(xi_v) + 1e-30):
            break
        old_xi = xi_v

        # F1 = fliplr(toep(sigma, n1, n1)) @ vecSigma - Ca2 + xi^2*Cgam2
        T_sig = toep(sigma_v, n1, n1)
        F1 = np.fliplr(T_sig) @ vs_v - Ca2 + xi_v**2 * Cgam2

        # F2 = toep(b,n2,mQ)@vecQ + toep(sigma*q,n2,mP)@vecP - fliplr(toep(xi*c,n2,mP))@vecP
        sq_v = np.convolve(sigma_v, q)
        T_b    = toep(b,      n2, mQ)
        T_sq   = toep(sq_v,   n2, mP)
        T_xc   = np.fliplr(toep(xi_v * c, n2, mP))
        F2 = T_b @ vq_v + T_sq @ vp_v - T_xc @ vp_v

        F = np.concatenate([F1, F2])

        # Jacobian
        Jsigma1 = toep(sigma_v[::-1], n1, n1) + np.fliplr(toep(sigma_v, n1, n1))
        Jxi1    = 2.0 * xi_v * Cgam2

        Jsigma2 = toep(np.convolve(P_v, q), n2, mSigma)
        Jxi2    = -np.fliplr(toep(c, n2, mP)) @ vp_v

        JQ = T_b
        JP = (T_sq - np.fliplr(toep(xi_v * c, n2, mP)))[:, :-1]  # drop last col

        n_total = 1 + mSigma + mQ + mP - 1
        J = np.zeros((n1 + n2, n_total))
        J[:n1,  0]              = Jxi1
        J[:n1,  1:1+mSigma]    = Jsigma1
        J[n1:,  0]              = Jxi2
        J[n1:,  1:1+mSigma]    = Jsigma2
        J[n1:,  1+mSigma:1+mSigma+mQ]   = JQ
        J[n1:,  1+mSigma+mQ:]  = JP

        try:
            dX, _, _, _ = la.lstsq(J, F, cond=None)
            # Iterative refinement (mirrors MATLAB linsys 'svd','refine')
            _e = F - J @ dX
            for _ in range(20):
                try:
                    _dx2, _, _, _ = la.lstsq(J, _e, cond=None)
                except Exception:
                    break
                _dX2 = dX + _dx2
                _e2 = F - J @ _dX2
                if np.linalg.norm(_e2) >= np.linalg.norm(_e):
                    break
                dX, _e = _dX2, _e2
        except Exception:
            break

        X    = X - dX
        xi_v = float(X[i_xi])

        # Reset if P ceases to be strictly antistable (once)
        P_chk = np.concatenate([X[i_p], [1.0]])[::-1]
        rts_chk = np.roots(P_chk) if len(P_chk) > 1 else np.array([])
        if len(rts_chk) > 0 and np.any(np.abs(rts_chk) > 1.0) and not reset:
            xi_r  = 0.01 * xi0
            A_r   = TA_sq - xi_r * TB_sq
            try:
                vqp_r, _, _, _ = la.lstsq(A_r[:, :mQP-1], -A_r[:, mQP-1], cond=None)
            except Exception:
                vqp_r = np.zeros(mQP - 1)
            sig_r = _sfact_coef(_sumpol2_c(aa, -(xi_r**2) * gg))
            vs_r  = sig_r[::-1]
            _vs_r = np.zeros(mSigma)
            _vs_r[:min(len(vs_r), mSigma)] = vs_r[:min(len(vs_r), mSigma)]
            X     = np.concatenate([[xi_r], _vs_r, vqp_r])
            reset = True

    # Extract final solution. NOTE: the Newton-Raphson may legitimately
    # converge to a NEGATIVE xi (MATLAB's loop does exactly this for dhinf
    # help-Example 2: 0.476 → -0.584 → … → -0.47071) — hinfred.m handles it
    # with a final `if lam < 0, lam = -lam` and has no positivity guard.
    # The previous check `xi_fin < 1e-15` rejected every negative-xi
    # solution as "degenerate" (found by diffing iteration
    # trajectories against an independent Octave re-derivation).
    xi_fin  = float(X[i_xi])
    if abs(xi_fin) < 1e-15 or not np.isfinite(xi_fin):
        return np.inf, np.array([1.0]), np.array([1.0]), np.array([1.0]), float('nan')

    lam     = 1.0 / xi_fin
    sig_fin = X[i_sig][::-1]                         # descending
    Q_fin   = X[i_q][::-1]                           # descending
    P_fin   = np.concatenate([X[i_p], [1.0]])[::-1]  # descending, monic

    Q_fin   = lam * Q_fin
    sig_fin = lam * sig_fin
    if lam < 0:
        lam = -lam

    # Reject if the Newton-Raphson diverged (NaN/Inf sigma)
    if np.any(np.isnan(sig_fin)) or np.any(np.isinf(sig_fin)):
        return np.inf, P_fin, Q_fin, sig_fin, float('nan')

    # Full-order solution check (hinfred.m lines 270-291): re-solve a FRESH
    # Diophantine equation using the *converged* sigma but the *original,
    # unscaled* c (not xi*c as used inside the main loop) —
    # dioph2(sigma_fin*q, -c, b, mP) -> (Px, Qx) — and require Px/Qx to share
    # a root at the unit circle (+1 or -1). This is a distinct
    # self-consistency/optimality condition from the Kwakernaak/Saeki theory,
    # not a check on the main loop's own P. Previously this used P_fin's own
    # roots here by mistake (checking an entirely different, wrong
    # condition).
    try:
        Px, Qx = _dioph2_np(np.convolve(sig_fin, q), -c, b, mP)
        rts_Px = np.roots(Px) if len(Px) > 1 else np.array([])
        rts_Qx = np.roots(Qx) if len(Qx) > 1 else np.array([])
        if len(rts_Px) > 0:
            e1 = np.min(np.abs(rts_Px - 1.0))
            e2 = np.min(np.abs(rts_Px + 1.0))
            errP = min(e1, e2)
            rts0 = 1.0 if e1 < e2 else (-1.0 if e2 < e1 else 0.0)
            errQ = np.min(np.abs(rts_Qx - rts0)) if len(rts_Qx) > 0 else np.inf
            if max(errP, errQ) > 0.1:
                return np.inf, P_fin, Q_fin, sig_fin, float('nan')
    except Exception:
        pass

    # Strict instability check (hinfred.m lines 293-298): the main loop's own
    # P must not have any root strictly outside the unit circle.
    rts_P = np.roots(P_fin) if len(P_fin) > 1 else np.array([])
    if len(rts_P) > 0 and np.any(np.abs(rts_P) > 1.0):
        return np.inf, P_fin, Q_fin, sig_fin, float('nan')

    return abs(lam), P_fin, Q_fin, sig_fin, 0.0


# ---------------------------------------------------------------------------
# Core Hinf F-iteration — port of hinffiter.m (K. Polyakov)
# ---------------------------------------------------------------------------

def _hinffiter_np(aa, gg, b0, q0, c0, r0, deg_P, tol=1e-6):
    """
    F-iteration for polynomial Hinf equations.

    Port of hinffiter.m (K. Polyakov).  Falls back when bisection returns Inf.
    """
    from directsd.linalg.matrices import toep
    from directsd.polynomial.spectral import sfactfft

    aa = np.asarray(aa, dtype=float).ravel()
    gg = np.asarray(gg, dtype=float).ravel()
    b = np.convolve(np.asarray(b0, float).ravel(),
                    np.asarray(r0, float).ravel())
    q = np.asarray(q0, dtype=float).ravel()
    c = np.asarray(c0, dtype=float).ravel()

    deg_b0_val = _pdeg(np.asarray(b0, float))
    deg_q = _pdeg(q); deg_c = _pdeg(c)
    deg_si = max(_pdeg(aa), _pdeg(gg)) // 2
    # MATLAB hinffiter.m: n2=delB+delC, mF=delB, mN=delC-deg_r (square system).
    # Python had +1 in both n2 and mN, solving a different (wrong) linear system.
    deg_all = deg_b0_val + max(deg_si + deg_q, deg_c)  # = delB + delC
    deg_F = deg_b0_val                                  # = delB = mF
    deg_N = max(deg_all - deg_b0_val - _pdeg(np.asarray(r0, float)), 0)  # = delC - deg_r
    mF = deg_F; mN = deg_N      # no +1 (MATLAB: mN = delC - deg_r, not +1)
    n2 = deg_all                 # no +1 (MATLAB: n2 = delB + delC, not +1)

    xi0 = _xi0_init(aa, gg)

    # Non-generic check
    ag2_0 = _sumpol2_c(aa, -(xi0 ** 2) * gg)
    sigma0 = _sfact_coef(ag2_0)
    if not (np.any(np.isnan(sigma0)) or np.any(np.isinf(sigma0))):
        F0, N0 = _dioph2_np(np.convolve(sigma0, q), -(xi0 * c), b, mF)
        if len(F0) > 1 and np.all(np.abs(np.roots(F0)) < 1.0):
            lam = 1.0 / xi0
            return lam, F0, lam * N0, lam * sigma0, 0.0
    else:
        sigma0 = _sfact_coef(aa)

    lam = 1e8
    Right = np.convolve(c, np.array([tol]))
    F = np.array([1.0])
    N = np.array([1.0])
    sigma = sigma0.copy()

    for _outer in range(200):
        sigma = _sfact_coef(_sumpol2_c(aa, -(gg / (lam ** 2))))
        lam_old = lam

        for _inner in range(20):
            lam_x = lam
            JN = toep(b, n2, mN)
            JF = toep(np.convolve(sigma, q), n2, mF)
            B_rhs = toep(Right, n2, 1)
            A_mat = np.real(np.hstack([JN, JF]))
            rhs_vec = np.real(B_rhs[:, 0])
            try:
                X, _, _, _ = la.lstsq(A_mat, rhs_vec, cond=None)
            except Exception:
                break
            N = np.flipud(X[:mN])
            Flam = np.flipud(X[mN:mN + mF])
            if len(Flam) == 0:
                break
            lam = abs(Flam[0]) if abs(Flam[0]) > 1e-30 else lam
            F = Flam / lam
            Fs = F[::-1]
            Right = np.convolve(Fs, c)
            if abs(lam_x - lam) < tol * abs(lam):
                break

        if abs(lam_old - lam) < tol * abs(lam):
            break

    # After convergence, recompute sigma at the final lam.
    # sfactfft can return NaN when aa - gg/lam² is not PSD (lam > lam_opt).
    sigma_final = _sfact_coef(_sumpol2_c(aa, -(gg / (lam ** 2))))
    if np.any(np.isnan(sigma_final)) or np.any(np.isinf(sigma_final)):
        return np.inf, F, N, np.array([float('nan')]), float('nan')
    sigma = lam * sigma_final

    # Validity checks
    rts_F = np.roots(F) if len(F) > 1 else np.array([])
    if len(rts_F) > 0:
        e1 = np.min(np.abs(rts_F - 1.0))
        e2 = np.min(np.abs(rts_F + 1.0))
        if min(e1, e2) > 0.1 or np.any(np.abs(rts_F) > 1.0):
            return np.inf, F, N, sigma, float('nan')

    return abs(lam), F, N, sigma, 0.0


# ---------------------------------------------------------------------------
# Small helpers for the Hinf solvers
# ---------------------------------------------------------------------------

def _sumpol2_c(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two symmetric (centered) quasipolynomials — port of sumpol2.m:
    the arrays are CENTER-aligned (`c = [0.. a ..0] + [0.. b ..0]` with
    (la-lb)/2 zeros on each side), not end-aligned. End-alignment (the
    previous behaviour) silently produced an ASYMMETRIC result whenever the
    operands had different degrees — e.g. hinfred's ag2 = aa - xi0²·gg with
    deg(aa)=4, deg(gg)=2 for dhinf help-Example 2 (all earlier call sites
    happened to pass equal-degree operands).
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    la, lb = len(a), len(b)
    if la % 2 == 1 and lb % 2 == 1:
        n = (la - lb) // 2
        if n >= 0:
            out = a.copy()
            out[n:la - n] += b
        else:
            out = b.copy()
            out[-n:lb + n] += a
        return out
    # Non-quasipolynomial (even-length) operand: MATLAB sumpol2 errors here;
    # keep the legacy end-alignment for any remaining plain-polynomial use.
    if la >= lb:
        out = a.copy()
        out[la - lb:] += b
    else:
        out = b.copy()
        out[lb - la:] += a
    return out


def _sfact_coef(coef: np.ndarray) -> np.ndarray:
    """
    Spectral factorization of palindromic polynomial via sfactfft.

    Matches MATLAB hinfred.m exactly: use sfactfft('d') as-is (raw sign/scale
    preserved).  sfactfft's FFT-cepstrum may displace near-unit-circle roots
    slightly inward; that displacement is intentional — it places the Diophantine
    on the correct solution branch (P has an outside root → generic case → stable K).
    Exception: when coef(1)==0 (zero at z=1), sfactfft is unreliable so we fall
    back to polynomial sfactor (MATLAB hinfred.m line 80).
    """
    from directsd.polynomial.spectral import sfactfft
    coef = np.real(np.asarray(coef)).ravel().astype(float)
    if len(coef) < 2:
        return coef.copy()

    # MATLAB hinfred.m line 80: if sum(ag2) ≈ 0, use polynomial sfactor instead
    if abs(np.sum(coef)) < 1e-8 * np.linalg.norm(coef):
        try:
            sf_res = sfactor(Poln(coef, 'z'), 'd')
            sf_res = sf_res[0] if isinstance(sf_res, tuple) else sf_res
            return np.real(np.asarray(
                sf_res.coef if hasattr(sf_res, 'coef') else sf_res
            )).astype(float)
        except Exception:
            pass  # fall through to sfactfft

    try:
        result = sfactfft(coef, ftype='d')
    except Exception:
        return np.array([float(np.sqrt(abs(coef[0])))])
    if isinstance(result, tuple):
        result = result[0]
    fp = np.real(np.asarray(result.coef if hasattr(result, 'coef') else result)).astype(float)
    if len(fp) < 2 or np.any(np.isnan(fp)) or np.any(np.isinf(fp)):
        return np.array([float(np.sqrt(abs(coef[0])))])
    return fp


def _dioph2_np(a_coef, b_coef, c_coef, deg_X):
    """
    Solve X·a + X~·b + Y·c = 0 (deg(X)=deg_X) via the Poln dioph2 solver.

    Returns (X_coef, Y_coef) as descending real numpy arrays.
    """
    a_r = np.real(np.asarray(a_coef)).astype(float)
    b_r = np.real(np.asarray(b_coef)).astype(float)
    c_r = np.real(np.asarray(c_coef)).astype(float)
    a_p = Poln(_strip_lz(a_r), 'z')
    b_p = Poln(_strip_lz(b_r), 'z')
    c_p = Poln(_strip_lz(c_r), 'z')
    X, Y, _ = dioph2(a_p, b_p, c_p, int(deg_X))
    return np.real(np.asarray(X.coef)).astype(float), np.real(np.asarray(Y.coef)).astype(float)


def _fit_sym_poly(values: np.ndarray, w: np.ndarray, half_deg: int) -> np.ndarray:
    """
    Fit a symmetric Laurent polynomial to real values on [0, pi].

    Returns coefficients of the positive half (degree 0 to half_deg),
    assembled into a full symmetric polynomial of degree 2*half_deg
    (descending order).
    """
    coefs = np.zeros(half_deg + 1)
    for k in range(half_deg + 1):
        integrand = values * np.cos(k * w)
        _trap = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        coefs[k] = float(_trap(integrand, w)) / np.pi
    coefs[0] /= 2.0

    # Build full symmetric polynomial: c0 + c1*(z^1+z^-1) + ... → degree 2*half_deg
    n = 2 * half_deg
    out = np.zeros(n + 1)
    out[half_deg] += coefs[0]
    for k in range(1, half_deg + 1):
        out[half_deg - k] += coefs[k]
        out[half_deg + k] += coefs[k]
    return _strip_lz(out)
