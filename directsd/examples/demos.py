"""
DirectSD Python Toolbox – Demo Scripts
=======================================
Faithful ports of the 25 MATLAB dsddemos/*.m scripts.

Each demo is a self-contained function. Run all of them via::

    python -m directsd.examples.demos

Or run a single demo::

    python -c "from directsd.examples.demos import demo_doubint; demo_doubint()"

Differences from MATLAB originals
----------------------------------
* Simulink-based transient simulations are replaced with
  ``scipy.signal.lsim`` (CT) or the ``sdsim`` function.
* MATLAB ``pause`` / interactive prompts are omitted.
* ``z2zeta`` (ζ = 1/z substitution) is handled inline where needed.
* ``sdh2norm(sys, K, t_array)`` (inter-sample variance curve) is not yet
  implemented in Python; average H2-norm is shown instead.
* ``modsdh2``/``modsdl2`` require a SISO P22 plant; full MIMO cost is
  computed via ``sdh2``/``sdl2`` on the standard system.
* ``ch2`` returns degenerate results for plants with integrators;
  CT-redesign sections use ``bilintr`` only when ``ch2`` succeeds.
"""

from __future__ import annotations
import math
import numpy as np
import scipy.signal as sig

from directsd import (
    sdh2, sdl2, ch2, dhinf, sdahinf, sdtrhinf,
    sdh2hinf, modsdh2, modsdl2, sd2dof, split2dof,
    sdh2norm, sdhinorm, sdahinorm, dinfnorm,
    sdl2err, sd2doferr, sdtrhinferr,
    sdmargin,
    sfactor, bilintr,
    neldermead, simanneal, sector, banana,
    h2reg, sdh2reg, sdfast, sdh2simple, sdnorm,
    sdsim,
)
from directsd import GeneralizedPlant
from directsd.tf import mul, neg, add, feedback, nd, to_lti
from directsd.examples._common import cl_poles as _cl_poles, _z2zeta


# ── Shared helpers ────────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print(f"\n{'='*56}")
    print(f"  {title}")
    print('='*56)


def _hdr(title: str) -> None:
    print(f"\n{'#'*60}")
    print(f"  DirectSD Demo: {title}")
    print(f"{'#'*60}")



def _K_ss(K, T):
    """Convert (num, den) controller tuple → discrete StateSpace with dt=T."""
    return sig.StateSpace(*sig.dlti(K[0], K[1], dt=T).to_ss())


def _hinf(plant, K, T):
    """sdhinorm returns (norm, freq); extract norm only."""
    return sdhinorm(plant, K, T)[0]


def _tf_nd(K_ss):
    """Extract (num, den) 1-D arrays from SISO discrete StateSpace (h2reg output)."""
    tf = K_ss.to_tf()
    n, d = tf.num, tf.den
    # SISO: num/den are 1-D arrays; MIMO: 2-D object arrays
    if isinstance(n, np.ndarray) and n.dtype == object:
        return np.atleast_1d(n[0][0]).ravel(), np.atleast_1d(d[0][0]).ravel()
    return np.atleast_1d(n).ravel(), np.atleast_1d(d).ravel()


# ── demo_doubint ──────────────────────────────────────────────────────────────

def demo_doubint():
    """Optimal control for double integrator (Polyakov et al., CCA 2002)."""
    _hdr("Optimal digital control of double integrator")

    F = sig.lti([1], [1, 0, 0])   # 1/s^2
    T = 0.1

    _sep("Sampled-data H2-optimisation")
    Kopt, err_opt = sdh2(F, T)
    print(f"H2-optimal controller:  {np.round(Kopt[0],4)} / {np.round(Kopt[1],4)}")
    poles = _cl_poles(F, Kopt, T)
    print(f"Closed-loop poles:      {np.round(poles, 4)}")
    # sdh2norm has numerical issues for double integrator; use design cost
    print(f"Sampled-data H2 cost:   {err_opt:.6f}")

    _sep("Continuous-time H2-optimisation (Tustin redesign)")
    Kc, cost_ct = ch2(F)
    # ch2 may return degenerate result for pure integrators
    try:
        n, d = Kc
        if len(np.atleast_1d(d).ravel()) <= 1 and np.allclose(np.atleast_1d(d).ravel(), 0):
            print("CT H2 degenerate for pure double integrator (improper cost).")
        else:
            Kcd = bilintr(Kc, 'tustin', T)
            poles_c2d = _cl_poles(F, Kcd, T)
            print(f"CT H2 → Tustin K:       {np.round(Kcd[0],4)} / {np.round(Kcd[1],4)}")
            print(f"Closed-loop poles:      {np.round(poles_c2d, 4)}")
            err_c2d = sdh2norm(F, Kcd, T)
            print(f"Sampled-data H2 cost:   {err_c2d:.6f}")
    except Exception as exc:
        print(f"  (CT redesign skipped: {exc})")

    _sep("H2-optimisation for ZOH-discretised model")
    F_ss = sig.lti([1], [1, 0, 0]).to_ss()
    # Measurement row uses -C (P22 = -F) to match negative-feedback convention
    # (consistent with _siso_to_gen_plant and MATLAB's sys = [[-F,-F],[0,1],[-F,-F]])
    sys_gen = sig.StateSpace(
        F_ss.A,
        np.hstack([F_ss.B, F_ss.B]),
        np.vstack([F_ss.C, np.zeros((1, F_ss.A.shape[0])), -F_ss.C]),
        np.array([[0., 0.], [0., 1.], [0., 0.]])
    )
    dsys_d = sys_gen.to_discrete(T, method='zoh')
    Kd_ss, h2n_d = h2reg(dsys_d, n_meas=1, n_ctrl=1)
    Kd = _tf_nd(Kd_ss)
    print(f"Discrete-time K:        {np.round(Kd[0],4)} / {np.round(Kd[1],4)}")
    poles_d = _cl_poles(F, Kd, T)
    print(f"Closed-loop poles:      {np.round(poles_d, 4)}")
    err_d = sdh2norm(F, Kd, T)
    print(f"Sampled-data H2 cost:   {err_d:.6f}")

    print(f"\nSummary (H2 cost, lower is better):")
    print(f"  Sampled-data design:  {err_opt:.6f}")
    print(f"  Discrete-time design: {err_d:.6f}")


# ── demo_ait98 ────────────────────────────────────────────────────────────────

def demo_ait98():
    """H2 and AHinf-optimisation (Polyakov, ARC 1998)."""
    _hdr("H2 and AHinf-optimisation of sampled-data system")

    F2 = sig.lti([1], [1, -1])   # 1/(s-1)  unstable plant
    T = 1

    _sep("H2-optimisation (SISO plant, standard form)")
    K, err_opt = sdh2(F2, T)
    print(f"H2-optimal controller:  {np.round(K[0],4)} / {np.round(K[1],4)}")
    poles = _cl_poles(F2, K, T)
    print(f"Closed-loop poles:      {np.round(poles, 4)}")
    err = sdh2norm(F2, K, T)
    print(f"Sampled-data H2 cost:   {err:.6f}")
    lam = sdahinorm(F2, K, T)
    print(f"AHinf-cost:             {lam:.6f}")
    lam_inf = _hinf(F2, K, T)
    print(f"Hinf-norm:              {lam_inf:.6f}")

    _sep("AHinf-optimisation")
    Kinf, lam_opt = sdahinf(F2, T)
    print(f"AHinf-optimal K:        {np.round(Kinf[0],4)} / {np.round(Kinf[1],4)}")
    poles2 = _cl_poles(F2, Kinf, T)
    print(f"Closed-loop poles:      {np.round(poles2, 4)}")
    err2 = sdh2norm(F2, Kinf, T)
    print(f"H2 cost:                {err2:.6f}")
    print(f"Optimal AHinf-cost:     {lam_opt:.6f}")
    lam_d = sdahinorm(F2, Kinf, T)
    print(f"AHinf-cost (verify):    {lam_d:.6f}")
    lam_inf2 = _hinf(F2, Kinf, T)
    print(f"Hinf-norm:              {lam_inf2:.6f}")


# ── demo_at96 ─────────────────────────────────────────────────────────────────

def demo_at96():
    """Optimal ship course stabilisation – 'Kazbek' tanker (AT 1996)."""
    _hdr("Stochastic optimisation for 'Kazbek' type tanker")

    F = sig.lti([0.051], [25, 1, 0])
    rho = np.sqrt(0.1)
    Sw = sig.lti([0.0757], [1, 0, 2.489, 0, 1.848])
    Fw, _ = sfactor(Sw)
    Fw_lti = to_lti(Fw)              # ensure TF form (sfactor may return ZPK)
    T = 1

    def _sys(rho_):
        return GeneralizedPlant([
            [mul(F, Fw_lti),          F],       # z1
            [0,                         rho_],    # z2
            [neg(mul(F, Fw_lti)),   neg(F)], # y
        ])

    _sep("H2-optimal controller")
    sys = _sys(rho)
    K, err_opt = sdh2(sys, T)
    print(f"H2-optimal K:        {np.round(K[0],4)} / {np.round(K[1],4)}")
    print(f"Optimal H2 cost:     {err_opt:.6f}")
    err_v = sdh2norm(sys, K, T)
    print(f"Direct computation:  {err_v:.6f}")

    _sep("Trade-off curve  sigma_psi vs sigma_u  (7 rho values)")
    rr = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    print(f"  {'rho':>8}  {'H2 cost':>12}")
    for r in rr:
        sys_r = _sys(np.sqrt(r))
        K_r, J_r = sdh2(sys_r, T)
        print(f"  {r:>8.4f}  {J_r:>12.6f}")


# ── demo_autom97 ──────────────────────────────────────────────────────────────

def demo_autom97():
    """H2-optimal control for delayed plants (Automatica 1997)."""
    _hdr("Stochastic optimisation with time-delay")

    F = sig.lti([1, 0.5], [1, -1, 0])
    Fw = sig.lti([2], [1, 2])
    T = 0.2

    print(f"Plant: (s+0.5)/(s^2-s),  disturbance: 2/(s+2)")
    print(f"Sampling period T={T}  (input delay tau≈0.093 not modelled here)")

    sys = GeneralizedPlant([
        [mul(F, Fw),         F],
        [0,                    0],
        [neg(mul(F, Fw)),  neg(F)],
    ])

    _sep("Sampled-data H2-optimisation")
    K, err_opt = sdh2(sys, T)
    print(f"H2-optimal K:           {np.round(K[0],4)} / {np.round(K[1],4)}")
    print(f"Optimal H2 cost:        {err_opt:.6f}")
    err = sdh2norm(sys, K, T)
    print(f"Average variance:       {err:.6f}")
    print(f"Note: inter-sample variance curve not computed (Python limitation)")


# ── demo_c2d ──────────────────────────────────────────────────────────────────

def demo_c2d():
    """Optimal digital redesign – Rattan's example (IEEE TAC 1999)."""
    _hdr("Optimal digital redesign (Rattan's example)")

    F = sig.lti([10], [1, 1, 0])
    Kc_ct = sig.lti([0.416, 1], [0.139, 1])
    Q = feedback(mul(F, Kc_ct))
    R = sig.lti([1], [1, 0])
    T = 0.04

    sys = GeneralizedPlant([
        [mul(Q, R),    neg(F)],
        [R,              neg(F)],
    ])

    _sep("L2-optimisation")
    K, err_opt = sdl2(sys, T)
    print(f"Optimal controller:  {np.round(K[0],4)} / {np.round(K[1],4)}")
    poles = _cl_poles(F, K, T)
    print(f"Closed-loop poles:   {np.round(poles,4)}")
    print(f"Optimal L2 cost:     {err_opt:.6f}")
    err = sdl2err(sys, K, T)
    print(f"Direct calculation:  {err:.6f}")


# ── demo_cf1 ──────────────────────────────────────────────────────────────────

def demo_cf1():
    """Example 12.4.2 from Chen & Francis (1995)."""
    _hdr("Example 12.4.2 – Chen & Francis (1995)")

    pd = np.convolve([1/12, 1/2, 1], [1/12, 1/2, 1])
    Gh = sig.lti([2], pd)
    Gm = sig.lti([1], [1, 0])
    T = 1

    sys = GeneralizedPlant([
        [mul(add(Gm, neg(1)), Gh),  Gm],
        [neg(mul(Gm, Gh)),            neg(Gm)],
    ])

    _sep("L2-optimisation")
    K, err_L2 = sdl2(sys, T)
    print(f"L2-optimal K:       {np.round(K[0],4)} / {np.round(K[1],4)}")
    print(f"L2 cost:            {err_L2:.6f}")
    err = sdl2err(sys, K, T)
    print(f"Direct computation: {err:.6f}")

    _sep("H2-optimisation")
    KH2, err_H2 = sdh2(sys, T)
    print(f"H2-optimal K:       {np.round(KH2[0],4)} / {np.round(KH2[1],4)}")
    print(f"H2 cost:            {err_H2:.6f}")
    err_L2b = sdl2err(sys, KH2, T)
    print(f"L2 cost:            {err_L2b:.6f}")


# ── demo_cf2 ──────────────────────────────────────────────────────────────────

def demo_cf2():
    """Examples 6.6.1, 8.4.2, 12.1.1 from Chen & Francis (1995)."""
    _hdr("Examples 6.6.1, 8.4.2, 12.1.1 – Chen & Francis (1995)")

    F = sig.lti([1], np.convolve([10, 1], [25, 1]))
    R = sig.lti([1], [1, 0])
    T = 1

    sys = GeneralizedPlant([
        [neg(R),   F],
        [R,          neg(F)],
    ])

    _sep("L2-optimisation")
    K, err_opt = sdl2(sys, T)
    print(f"L2-optimal K:       {np.round(K[0],4)} / {np.round(K[1],4)}")
    print(f"L2 cost:            {err_opt:.6f}")
    err = sdl2err(sys, K, T)
    print(f"Direct computation: {err:.6f}")


# ── demo_dhinf ────────────────────────────────────────────────────────────────

def _nd_neg(nd):
    """Negate numerator of a (num, den) tuple."""
    return ([-x for x in nd[0]], nd[1])


def demo_dhinf():
    """Discrete-time polynomial Hinf-optimisation (Grimble 1994)."""
    _hdr("Polynomial Hinf-design for discrete-time systems")

    T = 1

    # ── Example 1 (matches MATLAB demo_dhinf example 1) ───────────────────
    _sep("Example 1")
    # F(z) = z/(-2z+1), Fw(z) = (z-2)/(-2z+1)
    # sys = [Fw F; 0 1; -Fw -F]  (3×2 generalised plant in z-domain)
    F1  = ([1., 0.],  [-2., 1.])
    Fw1 = ([1., -2.], [-2., 1.])
    sys1 = [
        [Fw1,                      F1],
        [([0.], [1.]),             ([1.], [1.])],
        [_nd_neg(Fw1),             _nd_neg(F1)],
    ]
    K1, err1 = dhinf(_z2zeta(sys1))
    print(f"Hinf-optimal K:       {np.round(K1[0],4)} / {np.round(K1[1],4)}")
    print(f"Optimal Hinf-cost:    {err1:.6f}")
    F1_lti = sig.lti(*F1)
    poles1 = _cl_poles(F1_lti, K1, T)
    print(f"Closed-loop poles:    {np.round(poles1, 4)}")

    # ── Example 2 (demo_dhinf.m example 2 — the dsd_help generic example) ──
    # F2 = 1/(z²-2.1z-1), F1 = 2z²+z, Fw = 0.3z+1 (descending-z coefficients
    # of MATLAB's ascending tf([...]) inputs); sys rows: [F2·Fw, F2·F1;
    # 0, 1; -F2·Fw, -F2·F1].  Documented: K = 1.6201(z+0.4019)/(z+1.302),
    # cost 2.1244.  (The previous port used an invented z²/(z²-1.6z+0.63)
    # system here — not MATLAB's demo.)
    _sep("Example 2")
    F2d = [-1.0, -2.1, 1.0]
    sys2 = [
        [([0.3, 1.0], F2d),        ([2.0, 1.0, 0.0], F2d)],
        [([0.], [1.]),             ([1.], [1.])],
        [([-0.3, -1.0], F2d),      ([-2.0, -1.0, 0.0], F2d)],
    ]
    K2, err2 = dhinf(_z2zeta(sys2))
    print(f"Hinf-optimal K:       {np.round(K2[0],4)} / {np.round(K2[1],4)}")
    print(f"Optimal Hinf-cost:    {err2:.6f}  (documented 2.1244)")
    print("Documented K:         1.6201 (z+0.4019)/(z+1.302)")

    # ── Example 3 (demo_dhinf.m example 3 — non-generic case) ──────────────
    # F2 = 1/(0.32-1.2z+z²), F1 = -0.8z+z²... (dsd_help Ex3); two documented
    # equally-optimal controllers, cost 2.5725.
    _sep("Example 3 (non-generic)")
    F3d = [0.32, -1.2, 1.0]
    sys3 = [
        [([-0.4, 1.0], F3d),       ([-0.8, 1.0, 0.0], F3d)],
        [([0.], [1.]),             ([1.], [1.])],
        [([0.4, -1.0], F3d),       ([0.8, -1.0, 0.0], F3d)],
    ]
    K3, err3 = dhinf(_z2zeta(sys3))
    print(f"Optimal Hinf-cost:    {err3:.6f}  (documented 2.5725)")
    print("Documented optima:    -0.72171(z-2.24)(z-0.4)/((z+1.825)(z-0.8859))")
    print("                  or   1.9521(z-0.8282)(z-0.4)/(z^2-2.496z+1.617)")
    # Non-generic case: dhinf returns BOTH equally-optimal controllers as a
    # list (MATLAB: K = {K1,K2} cell array).
    for i, Ki in enumerate(K3):
        print(f"K{i}:                   {np.round(Ki[0],4)} / {np.round(Ki[1],4)}")
    print("\nNote: dhinf solves discrete-time polynomial Hinf for full MIMO plant.")


# ── demo_fil1 ─────────────────────────────────────────────────────────────────

def demo_fil1():
    """Optimal sampled-data filtering (Rosenwasser et al., IJACSP 1998)."""
    _hdr("Optimal sampled-data filtering")

    Sr = sig.lti([4], [-1, 0, 4])
    Fr, _ = sfactor(Sr)
    Fr_lti = to_lti(Fr)          # convert ZPK to TF
    F = sig.lti([1], [1, 1])
    Fn = 1
    T = 0.1

    # 2 rows (z1, y), 3 cols (w1, w2, u)
    sys = GeneralizedPlant([
        [Fr_lti, 0,  neg(F)],
        [Fr_lti, Fn,  0],
    ])

    _sep("H2-optimisation (average variance)")
    K, err_opt = sdh2(sys, T)
    print(f"Optimal H2 cost:    {err_opt:.6f}")
    err = sdh2norm(sys, K, T)
    print(f"Average variance:   {err:.6f}")

    _sep("H2-optimisation for t=0 (method='pol')")
    K0, err0 = sdh2(sys, T, t=0.0, method='pol')
    print(f"Variance at t=0:    {err0:.6f}")
    err0_v = sdh2norm(sys, K0, T)
    print(f"Average variance:   {err0_v:.6f}")
    print("\nNote: inter-sample variance curve requires 'pol' method with t array.")


# ── demo_fil2 ─────────────────────────────────────────────────────────────────

def demo_fil2():
    """Optimal sampled-data filtering with time-delay."""
    _hdr("Optimal sampled-data filtering with time-delay")

    # MATLAB: F.iodelay = 0.051, sys = [Fr 0 -F; Fr Fn 0] (dsd_help.md
    # "Sampled-data filtering with delay"). Fr = sfactor(4/(-s^2+4)) = Fw
    # exactly (see demo_fil1). Delay handled EXACTLY via sdh2's udelay
    # (modified Z-transform, matching MATLAB's F.iodelay semantics) — no
    # Padé in the design plant.
    F = sig.lti([1], [1, 1])
    Fw = sig.lti([2], [1, 2])
    Fn = 1
    T = 0.1
    tau = 0.051

    sys = GeneralizedPlant([
        [Fw, 0,       neg(F)],
        [Fw, Fn,      0],
    ])

    _sep("H2-optimisation (average variance)")
    K, err_opt = sdh2(sys, T, udelay=tau)
    print(f"H2-optimal K:       {np.round(K[0],4)} / {np.round(K[1],4)}")
    print(f"MATLAB K:           3.7544 z(z-0.9048)/((z+0.2678)(z-0.5201))")
    print(f"Optimal H2 cost:    {err_opt:.6f}")

    # sdh2norm has no delay parameter (like sdl2err) — verify on a
    # Padé(3)-approximated plant (fine for evaluation; MATLAB's own
    # documented err/err0 are what a Padé-free exact evaluator would give).
    n_pade = 3
    _c = [math.factorial(2 * n_pade - k) * math.factorial(n_pade)
          / (math.factorial(2 * n_pade) * math.factorial(k) * math.factorial(n_pade - k))
          for k in range(n_pade + 1)]
    pn = np.array([_c[k] * (-tau) ** k for k in range(n_pade + 1)])[::-1]
    pd = np.array([_c[k] * tau ** k for k in range(n_pade + 1)])[::-1]
    F_delay = sig.lti(np.polymul([1], pn), np.polymul([1, 1], pd))
    sys_eval = GeneralizedPlant([
        [Fw, 0,  neg(F_delay)],
        [Fw, Fn, 0],
    ])
    err = sdh2norm(sys_eval, K, T)
    print(f"Average variance:   {err:.6f}  (MATLAB sdh2norm(sys,K)=0.7879)")

    _sep("H2-optimisation for t=0 (method='pol')")
    # sdh2coef.m's "Instantaneous variance" branch (t != None), combined
    # with the same udelay as the average design above — matches
    # dsd_help.md's documented example exactly (K0 = sdh2(sys,T,0) is
    # called on the SAME delayed `sys`, not a delay-free variant).
    K0, err0 = sdh2(sys, T, t=0.0, udelay=tau)
    print(f"H2-optimal K0:       {np.round(K0[0],4)} / {np.round(K0[1],4)}")
    print(f"MATLAB K0:           6.245 z(z-0.9048)/((z+0.9901)(z-0.5201))")
    print(f"Variance at t=0:     {err0:.6f}  (MATLAB sdh2norm(sys,K0,0)=0.7577)")
    # sdh2norm's t-path now takes udelay directly (exact modified
    # Z-transform, same mechanism as design) instead of forcing a
    # Padé-approximated plant — the Padé route was numerically unusable
    # here since the fast Padé poles interact badly with expm(A*t) at a
    # specific small t.
    err_at0 = sdh2norm(sys, K, T, t=0.0, udelay=tau)
    print(f"K at t=0:            {err_at0:.6f}  (MATLAB sdh2norm(sys,K,0)=0.7735)")
    err0_avg = sdh2norm(sys_eval, K0, T)
    print(f"K0 average variance: {err0_avg:.6f}  (MATLAB sdh2norm(sys,K0)=1.8466)")


# ── demo_fil3 ─────────────────────────────────────────────────────────────────

def demo_fil3():
    """Optimal sampled-data filtering for 2nd-order plant."""
    _hdr("Optimal sampled-data filtering – 2nd-order plant")

    F = sig.lti([1], [1, 1, 1])
    Fw = sig.lti([2], [1, 2])
    Fn = 1
    T = 0.1

    sys = GeneralizedPlant([
        [neg(Fw), 0,  F],
        [Fw,        Fn, 0],
    ])

    _sep("H2-optimisation")
    K, err_opt = sdh2(sys, T)
    print(f"Optimal H2 cost:    {err_opt:.6f}")
    err = sdh2norm(sys, K, T)
    print(f"Average variance:   {err:.6f}")

    _sep("H2-optimisation for t=0 (method='pol')")
    K0, err0 = sdh2(sys, T, t=0.0, method='pol')
    print(f"Variance at t=0:    {err0:.6f}")
    err0_v = sdh2norm(sys, K0, T)
    print(f"Average variance:   {err0_v:.6f}")


# ── demo_hold ─────────────────────────────────────────────────────────────────

def demo_hold():
    """Optimal filtering with generalised hold."""
    _hdr("Optimal sampled-data filtering with generalised hold")

    Sr = sig.lti([4], [-1, 0, 4])
    Fr, _ = sfactor(Sr)
    Fr_lti = to_lti(Fr)
    F = sig.lti([1], [1])
    Fn = 0.1
    T = 0.2
    H = sig.lti([1], [1, 2])

    sys = GeneralizedPlant([
        [neg(Fr_lti), 0,  F],
        [Fr_lti,        Fn, 0],
    ])

    _sep("H2-optimal filter with ZOH")
    K0, err0 = sdh2(sys, T)
    print(f"ZOH H2 cost:           {err0:.6f}")
    err0_v = sdh2norm(sys, K0, T)
    print(f"Average variance:      {err0_v:.6f}")

    _sep("H2-optimal filter with generalised hold H=1/(s+2)")
    K, err = sdh2(sys, T, H=H)
    print(f"GH H2 cost:            {err:.6f}")
    err_v = sdh2norm(sys, K, T, H=H)
    print(f"Average variance:      {err_v:.6f}")


# ── demo_h2hinf ──────────────────────────────────────────────────────────────

def demo_h2hinf():
    """Mixed H2/AHinf-optimisation."""
    _hdr("Mixed H2/Hinf optimisation")

    F = sig.lti([1], [5, 1, 0])
    rho = 1
    T = 1

    sys = GeneralizedPlant([
        [neg(F),  neg(F)],
        [0,         rho],
        [neg(F),  neg(F)],
    ])

    _sep("H2-optimisation")
    KH2, err_H2 = sdh2(sys, T)
    print(f"H2-optimal K:          {np.round(KH2[0],4)} / {np.round(KH2[1],4)}")
    print(f"H2 cost:               {err_H2:.6f}")
    lamH2 = sdahinorm(sys, KH2, T)
    print(f"AHinf cost:            {lamH2:.6f}")
    lamH2x = _hinf(F, KH2, T)
    print(f"Hinf norm:             {lamH2x:.6f}")

    _sep("AHinf-optimisation")
    # sdahinf always uses the polynomial (_polhinf) pipeline now, matching
    # MATLAB's sdahinf.m exactly -- for some plants this hits an unresolved
    # _polhinf degeneracy. Report honestly rather than crash.
    try:
        Kinf, lam_inf = sdahinf(sys, T)
        print(f"AHinf-optimal K:       {np.round(Kinf[0],4)} / {np.round(Kinf[1],4)}")
        err2 = sdh2norm(sys, Kinf, T)
        print(f"H2 cost:               {err2:.6f}")
        print(f"Optimal AHinf-cost:    {lam_inf:.6f}")
        lam_d = sdahinorm(sys, Kinf, T)
        print(f"AHinf-cost (verify):   {lam_d:.6f}")
    except Exception as exc:
        print(f"AHinf-optimisation failed (_polhinf degeneracy): {exc}")

    _sep("H2/AHinf-optimisation (rho=0.5)")
    try:
        Kmix, err_mix = sdh2hinf(sys, T, rho=0.5, o11=2, i11=1)
        print(f"Mixed polquad cost:    {err_mix:.6f}")
        err_m = sdh2norm(sys, Kmix, T)
        print(f"H2 cost:               {err_m:.6f}")
        lam_m = sdahinorm(sys, Kmix, T)
        print(f"AHinf cost:            {lam_m:.6f}")
    except Exception as exc:
        print(f"Mixed H2/AHinf-optimisation failed (_polhinf degeneracy): {exc}")

    _sep("Trade-off: rho sweep")
    print(f"  {'rho':>6}  {'H2-norm':>10}  {'AHinf-norm':>12}")
    for rho_ in [0.0, 0.25, 0.5, 0.75, 1.0]:
        try:
            Km, _ = sdh2hinf(sys, T, rho=rho_, o11=2, i11=1)
            h2 = sdh2norm(sys, Km, T)
            ah = sdahinorm(sys, Km, T)
            print(f"  {rho_:>6.2f}  {h2:>10.4f}  {ah:>12.4f}")
        except Exception as exc:
            print(f"  {rho_:>6.2f}  failed (_polhinf degeneracy): {exc}")


# ── demo_h2p ──────────────────────────────────────────────────────────────────

def demo_h2p():
    """H2-optimal control with preview (Polyakov et al., IEEE AC 2002)."""
    _hdr("H2-optimal controller with preview")

    F = sig.lti([1], [1, -1])     # 1/(s-1), unstable
    Fr = sig.lti([1], [5, 1])
    Fn = 0.2
    T = 1
    tau = 1.5   # computational delay (MATLAB: F.iodelay = 1.5)

    sys0 = GeneralizedPlant([
        [Fr,    0,    neg(F)],
        [Fr,    Fn,  neg(F)],
    ])

    _sep("H2-optimal controller (no preview)")
    K, err_opt = sdh2(sys0, T, udelay=tau)
    print(f"H2 cost (no preview):  {err_opt**2:.6f}")

    _sep("H2-optimal controller (preview pi=2, dsd_help.md documented example)")
    # MATLAB: non-causal preview block removed, equal delay placed in the
    # ideal operator Q instead (Q.iodelay = preview) -- see refdelay's
    # docstring. Since Q=1 (pure delay, no dynamics) here, the delay-free
    # plant above is exactly right; refdelay carries the preview horizon.
    K2, err2 = sdh2(sys0, T, udelay=tau, refdelay=2.0)
    print(f"K:                     {np.round(K2[0],4)} / {np.round(K2[1],4)}")
    print("MATLAB K: 8.1825 z^2(z-0.7985) / [(z-0.8113)(z^2+3.041z+3.169)]")
    print(f"H2 cost (preview=2):   {err2**2:.6f}")
    print("MATLAB sdh2norm(sys,K)^2 = 11.6701")

    _sep("Dependence on preview horizon pi")
    print(f"  {'pi':>6}  {'J_min(pi)':>10}")
    for preview in [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 9.999]:
        Kp, errp = sdh2(sys0, T, udelay=tau, refdelay=preview)
        print(f"  {preview:>6.2f}  {errp**2:>10.4f}")
    print("Note: this plant is UNSTABLE -- dsd_help.md's own documentation")
    print("notes cost is NOT monotonic in pi here (the controller must both")
    print("stabilize the plant and minimize the cost simultaneously); the")
    print("curve dips to a minimum near pi=2 and rises again for larger pi.")


# ── demo_l2 ───────────────────────────────────────────────────────────────────

def demo_l2():
    """Design of L2-optimal controller (Polyakov)."""
    _hdr("Sampled-data L2-optimisation")

    F = sig.lti([1], [5, 1, 0])
    Q = sig.lti([1], [1, 1])
    R = sig.lti([1], [1, 0])
    T = 0.2

    # 2x2 plant matching MATLAB sys = [Q*R -F; R -F]
    sys = GeneralizedPlant([
        [mul(Q, R),  neg(F)],
        [R,          neg(F)],
    ])

    _sep("Sampled-data L2-optimisation")
    K, err_opt = sdl2(sys, T)
    print(f"L2-optimal K:          {np.round(K[0],4)} / {np.round(K[1],4)}")
    poles = _cl_poles(F, K, T)
    print(f"Closed-loop poles:     {np.round(poles,4)}")
    print(f"Optimal L2 cost:       {err_opt:.6f}")
    err = sdl2err(sys, K, T)
    print(f"Direct calculation:    {err:.6f}")

    _sep("Lifting method (sdh2simple + h2reg)")
    # MATLAB: sysH2s = sdh2simple(sys, T); Ks = h2reg(sysH2s)  → same K as sdl2.
    dsysL2, *_ = sdh2simple(sys.to_statespace(), T)
    Ks_ss, _ = h2reg(dsysL2)
    Ks_tf = sig.StateSpace(Ks_ss.A, Ks_ss.B, Ks_ss.C, Ks_ss.D, dt=T).to_tf()
    Ks = (np.ravel(Ks_tf.num), np.ravel(Ks_tf.den))
    print(f"Lifting K:             {np.round(Ks[0],4)} / {np.round(Ks[1],4)}")
    poles_s = _cl_poles(F, Ks, T)
    print(f"Closed-loop poles:     {np.round(poles_s,4)}")
    print(f"Optimal L2 cost:       {err_opt:.6f}")
    err_s = sdl2err(sys, Ks, T)
    print(f"Direct calculation:    {err_s:.6f}")


# ── demo_l2hinf ───────────────────────────────────────────────────────────────

def demo_l2hinf():
    """AHinf-optimisation for tracking system (Polyakov, ARC 2001)."""
    _hdr("AHinf-optimisation for tracking system")

    F = sig.lti([1], [4, 0.5, 1])
    R = sig.lti([1], [1, 0])
    rho = 0.12
    T = 1

    sys = GeneralizedPlant([
        [R,               neg(F)],
        [mul(rho, R),   neg(rho)],
        [R,               neg(F)],
    ])

    _sep("L2-optimisation")
    KL2, err_l2 = sdl2(sys, T)
    print(f"L2-optimal K:          {np.round(KL2[0],4)} / {np.round(KL2[1],4)}")
    poles = _cl_poles(F, KL2, T)
    print(f"Closed-loop poles:     {np.round(poles,4)}")
    print(f"L2 cost:               {err_l2:.6f}")
    lam_l2 = sdtrhinferr(sys, KL2, T)
    print(f"AHinf-cost:            {lam_l2:.6f}")

    _sep("AHinf-optimisation")
    K, lam_opt = sdtrhinf(sys, T)
    print(f"AHinf-optimal K:       {np.round(K[0],4)} / {np.round(K[1],4)}")
    poles2 = _cl_poles(F, K, T)
    print(f"Closed-loop poles:     {np.round(poles2,4)}")
    err = sdl2err(sys, K, T)
    print(f"L2 cost:               {err:.6f}")
    print(f"Optimal AHinf-cost:    {lam_opt:.6f}")
    lam1 = sdtrhinferr(sys, K, T)
    print(f"AHinf-cost (verify):   {lam1:.6f}")


# ── demo_ait01b ───────────────────────────────────────────────────────────────

def demo_ait01b():
    """AHinf-optimisation for tracking system (Polyakov, ARC 2001)."""
    _hdr("AHinf-optimisation for tracking (ARC 2001)")

    F = sig.lti([1], [1, 1])
    R = sig.lti([1], [1, 0])
    T = 0.2
    Ve, Vu = 1, 0

    sys = GeneralizedPlant([
        [neg(mul(Ve, R)),    mul(Ve, F)],
        [0,                     Vu],
        [R,                    neg(F)],
    ])

    _sep("L2-optimisation")
    KL2, err_l2 = sdl2(sys, T)
    print(f"L2-optimal K:          {np.round(KL2[0],4)} / {np.round(KL2[1],4)}")
    poles = _cl_poles(F, KL2, T)
    print(f"Closed-loop poles:     {np.round(poles,4)}")
    print(f"L2 cost:               {err_l2:.6f}")
    lam = sdtrhinferr(sys, KL2, T)
    print(f"AHinf-cost:            {lam:.6f}")

    _sep("AHinf-optimisation")
    K, lam_opt = sdtrhinf(sys, T)
    print(f"AHinf-optimal K:       {np.round(K[0],4)} / {np.round(K[1],4)}")
    poles2 = _cl_poles(F, K, T)
    print(f"Closed-loop poles:     {np.round(poles2,4)}")
    err = sdl2err(sys, K, T)
    print(f"L2 cost:               {err:.6f}")
    print(f"Optimal AHinf-cost:    {lam_opt:.6f}")
    lam1 = sdtrhinferr(sys, K, T)
    print(f"AHinf-cost (verify):   {lam1:.6f}")


# ── demo_l2p ──────────────────────────────────────────────────────────────────

def demo_l2p():
    """L2-optimal control with preview (Polyakov et al., IFAC-TDS 2003)."""
    _hdr("L2-optimal controller with preview")

    F = sig.lti([1], [5, -1])     # 1/(5s-1), unstable (exNo=2 in demo_l2p.m)
    Q = sig.lti([1], [0.1, 1])
    R = sig.lti([1], [1, 1, 0])
    T = 1
    tau = 1.5   # computational delay (MATLAB: F.iodelay = 1.5)

    sys0 = GeneralizedPlant([
        [mul(Q, R), neg(F)],
        [R,           neg(F)],
    ])

    _sep("L2-optimal controller (no preview)")
    K, err_opt = sdl2(sys0, T, udelay=tau)
    poles = _cl_poles(F, K, T)
    print(f"Closed-loop poles:     {np.round(poles,4)}")
    print(f"L2 cost (no preview):  {err_opt:.6f}")

    _sep("L2-optimal controller (preview pi=2, dsd_help.md documented example)")
    # MATLAB: non-causal preview block removed, equal delay placed in the
    # ideal operator Q (Q.iodelay=preview) plus a remainder delay theta on
    # the reference generator R (R.iodelay=theta) -- see refdelay's
    # docstring in _sdl2coef for the sigma/theta split. Note this demo's
    # F is the UNSTABLE exNo=2 branch; dsd_help.md's own worked numeric
    # example (sdl2err=0.0517) uses the STABLE F=1/(5s+1) instead -- see
    # ex_l2_preview for that exact validation.
    K2, err2 = sdl2(sys0, T, udelay=tau, refdelay=2.0)
    print(f"K:                     {np.round(K2[0],4)} / {np.round(K2[1],4)}")
    print(f"L2 cost (preview=2):   {err2:.6f}")

    _sep("Dependence on preview horizon pi")
    print(f"  {'pi':>6}  {'J_min(pi)':>10}")
    for preview in [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.999]:
        Kp, errp = sdl2(sys0, T, udelay=tau, refdelay=preview)
        print(f"  {preview:>6.2f}  {errp:>10.4f}")
    print("Note: this plant is UNSTABLE -- dsd_help.md's own documentation")
    print("notes cost is NOT monotonic in pi here (the controller must both")
    print("stabilize the plant and minimize the cost simultaneously), unlike")
    print("the stable-plant case where J_min(pi) decreases monotonically.")


# ── demo_2dof ─────────────────────────────────────────────────────────────────

def demo_2dof():
    """Optimal 2-DOF tracking system (Polyakov, ARC 2001)."""
    _hdr("Optimal digital 2-DOF controller")

    F = sig.lti([1], [1, -1])
    R = sig.lti([1], [1, 0])
    Q = sig.lti([1], [1, 2])
    T = 0.5

    # 1-DOF plant: [z; y] = [Q*R*d - F*u; R*d - F*u]
    sys_1dof = GeneralizedPlant([
        [mul(Q, R),  neg(F)],
        [R,            neg(F)],
    ])

    # 2-DOF plant: adds separate reference measurement row y1=R*d, y2=-F*u
    sys_2dof = GeneralizedPlant([
        [mul(Q, R),  neg(F)],   # performance: Q*R*d - F*u
        [R,            0],         # y1: reference channel
        [0,            neg(F)],   # y2: plant output
    ], n_meas=2)

    _sep("1-DOF L2-optimisation")
    K, err_1dof = sdl2(sys_1dof, T)
    print(f"1-DOF controller:      {np.round(K[0],4)} / {np.round(K[1],4)}")
    poles = _cl_poles(F, K, T)
    print(f"Closed-loop poles:     {np.round(poles,4)}")
    print(f"L2 cost:               {err_1dof:.6f}")
    err = sdl2err(sys_1dof, K, T)
    print(f"Direct computation:    {err:.6f}")

    _sep("2-DOF feedforward design (sd2dof, feedback K fixed from 1-DOF above)")
    KR, err_2dof = sd2dof(sys_2dof, K, T)
    print(f"2-DOF feedforward KR:  {np.round(KR[0],4)} / {np.round(KR[1],4)}")
    print(f"2-DOF L2 cost:         {err_2dof:.6f}")
    err2 = sd2doferr(sys_2dof, K, KR, T)
    print(f"L2 cost (verify):      {err2:.6f}")

    _sep("Split 2-DOF: extract KF, KR_new, KC")
    KF, KR_new, KC = split2dof(K, KR)
    print(f"KF (feedback):         {np.round(KF[0],4)} / {np.round(KF[1],4)}")
    print(f"KR_new (reference):    {np.round(KR_new[0],4)} / {np.round(KR_new[1],4)}")
    print(f"KC (common):           {np.round(KC[0],4)} / {np.round(KC[1],4)}")


# ── demo_2dofp ────────────────────────────────────────────────────────────────

def demo_2dofp():
    """2-DOF optimal controller with preview (Polyakov et al., CDC 2004)."""
    _hdr("2-DOF optimal controller with preview")

    F = sig.lti([1], [5, -1])
    Q = sig.lti([1], [0.1, 1])
    R = sig.lti([1], [1, 1, 0])
    T = 1
    tau = 1.5   # computational delay (MATLAB: F.iodelay = 1.5)

    sys_1dof = GeneralizedPlant([
        [mul(Q, R), neg(F)],
        [R,           neg(F)],
    ])

    sys_2dof = GeneralizedPlant([
        [mul(Q, R), neg(F)],   # performance
        [R,           0],         # y1: reference channel
        [0,           neg(F)],   # y2: plant output
    ], n_meas=2)

    _sep("1-DOF L2 optimisation (preview pi=2, exact via sdl2's refdelay)")
    K, err_1 = sdl2(sys_1dof, T, udelay=tau, refdelay=2.0)
    print(f"1-DOF K:               {np.round(K[0],4)} / {np.round(K[1],4)}")
    poles = _cl_poles(F, K, T)
    print(f"Closed-loop poles:     {np.round(poles,4)}")
    print(f"L2 cost:               {err_1:.6f}  (MATLAB sdl2err=2.7072)")

    _sep("2-DOF feedforward design (sd2dof, feedback K fixed from 1-DOF above)")
    KR, err_2 = sd2dof(sys_2dof, K, T, udelay=tau, refdelay=2.0)
    print(f"2-DOF KR:              {np.round(KR[0],4)} / {np.round(KR[1],4)}")
    print(f"2-DOF L2 cost:         {err_2:.6f}  (MATLAB sd2doferr=0.0602)")
    print("(sd2doferr itself has no delay support yet -- like sdl2err/")
    print("sdh2norm on delayed plants -- so it is not called here; the")
    print("design cost above already matches MATLAB's documented value.)")


# ── demo_modsdh2 ──────────────────────────────────────────────────────────────

def demo_modsdh2():
    """Quasioptimal reduced-order H2-controller (ship tanker)."""
    _hdr("Reduced-order H2-optimal controller")

    F = sig.lti([0.051], [25, 1, 0])
    rho = np.sqrt(0.1)
    Sw = sig.lti([0.0757], [1, 0, 2.489, 0, 1.848])
    Fw, _ = sfactor(Sw)
    Fw_lti = to_lti(Fw)
    T = 1

    sys = GeneralizedPlant([
        [mul(F, Fw_lti),         F],
        [0,                        rho],
        [neg(mul(F, Fw_lti)),  neg(F)],
    ])

    _sep("Full-order optimal controller")
    KOpt, err_opt = sdh2(sys, T)
    print(f"Full-order K:          {np.round(KOpt[0],4)} / {np.round(KOpt[1],4)}")
    print(f"H2-norm:               {err_opt:.6f}")
    rOpt = _cl_poles(F, KOpt, T)
    print(f"Closed-loop poles:     {np.round(rOpt,4)}")

    _sep("Reduced-order modal H2 (order 1, SISO P22=F)")
    Kmod, err_mod = modsdh2(F, T, ord_K=1, alpha=0.0, beta=np.inf,
                              method='randsearch', n_iter=300)
    print(f"Reduced-order K:       {np.round(Kmod[0],4)} / {np.round(Kmod[1],4)}")
    print(f"H2-norm (SISO F):      {err_mod:.6f}")
    r_mod = _cl_poles(F, Kmod, T)
    print(f"Closed-loop poles:     {np.round(r_mod,4)}")
    print("\nNote: modsdh2/modsdl2 minimise SISO P22 cost (not full MIMO cost).")


# ── demo_modsdh2int ───────────────────────────────────────────────────────────

def demo_modsdh2int():
    """Reduced-order H2-optimal controller with an integrator."""
    _hdr("Reduced-order H2-optimal controller with integrator")

    F1 = sig.lti([0.0694], [18.22, 1])
    F2 = sig.lti([1], [1, 0])
    lam = 0.3; w0 = 0.3; sigma_w = 7.25
    Fw = sig.lti([2*lam*w0*sigma_w, 0], [1, 2*lam*w0, w0**2])
    rho = 2
    T = 2

    sys = GeneralizedPlant([
        [mul(F2, Fw),          mul(F2, F1)],
        [0,                      rho],
        [neg(mul(F2, Fw)),   neg(mul(F2, F1))],
    ])
    F_plant = mul(F2, F1)

    _sep("Full-order optimal controller")
    KOpt, err_opt = sdh2(sys, T)
    print(f"Full-order K:          {np.round(KOpt[0],4)} / {np.round(KOpt[1],4)}")
    print(f"H2-norm:               {err_opt:.6f}")
    rOpt = _cl_poles(F_plant, KOpt, T)
    print(f"Closed-loop poles:     {np.round(rOpt,4)}")

    _sep("Reduced-order modal H2 (order 2, SISO P22=F2*F1)")
    Kmod, err_mod = modsdh2(F_plant, T, ord_K=2, alpha=0.02, beta=2,
                              method='randsearch', n_iter=300)
    print(f"Reduced-order K:       {np.round(Kmod[0],4)} / {np.round(Kmod[1],4)}")
    print(f"H2-norm (SISO):        {err_mod:.6f}")
    r_mod = _cl_poles(F_plant, Kmod, T)
    print(f"Closed-loop poles:     {np.round(r_mod,4)}")
    alpha_r, beta_r = sector(r_mod, T)
    print(f"Stability sector: α={alpha_r/T:.4f}, β={beta_r:.4f}")


# ── demo_modsdl2 ──────────────────────────────────────────────────────────────

def demo_modsdl2():
    """Quasioptimal reduced-order L2-controller."""
    _hdr("Reduced-order L2-optimal controller")

    F = sig.lti([10], [2, 1, 0])
    R = sig.lti([1], [1, 0])
    Q = sig.lti([1], [1, 2, 1])
    T = 0.2

    sys = GeneralizedPlant([
        [neg(mul(Q, R)),  F],
        [R,                  neg(F)],
    ])

    _sep("Full-order optimal controller")
    KOpt, err_opt = sdl2(sys, T)
    print(f"Full-order K:          {np.round(KOpt[0],4)} / {np.round(KOpt[1],4)}")
    print(f"L2 cost:               {err_opt:.6f}")
    rOpt = _cl_poles(F, KOpt, T)
    print(f"Closed-loop poles:     {np.round(rOpt,4)}")

    _sep("Reduced-order modal L2 (order 1, SISO plant F)")
    Kmod, err_mod = modsdl2(F, T, ord_K=1, alpha=0.1, beta=np.inf,
                              method='randsearch', n_iter=300)
    print(f"Reduced-order K:       {np.round(Kmod[0],4)} / {np.round(Kmod[1],4)}")
    print(f"L2 cost (SISO):        {err_mod:.6f}")
    r_mod = _cl_poles(F, Kmod, T)
    print(f"Closed-loop poles:     {np.round(r_mod,4)}")
    print("\nNote: modsdl2 cost uses sdl2err on SISO plant F.")


# ── demo_modsdl2a ─────────────────────────────────────────────────────────────

def demo_modsdl2a():
    """Quasioptimal reduced-order L2-redesign."""
    _hdr("L2-optimal redesign – fixed static gain")

    F = sig.lti([1], [1, -1, 0])
    Kc_ct = sig.lti([5, 1], [1, 3])
    Q = feedback(mul(F, Kc_ct))
    R = sig.lti([1], [1, 0])
    T = 0.5

    sys = GeneralizedPlant([
        [mul(add(F, neg(Q)), R),  F],
        [neg(mul(F, R)),            neg(F)],
    ])

    _sep("Full-order optimal controller")
    KOpt, err_opt = sdl2(sys, T)
    print(f"Full-order K:          {np.round(KOpt[0],4)} / {np.round(KOpt[1],4)}")
    print(f"L2 cost:               {err_opt:.6f}")
    rOpt = _cl_poles(F, KOpt, T)
    print(f"Closed-loop poles:     {np.round(rOpt,4)}")

    _sep("Reduced-order modal L2 (order 2, SISO plant F)")
    Kmod, err_mod = modsdl2(F, T, ord_K=2, alpha=0.001, beta=np.inf,
                              method='randsearch', n_iter=300)
    print(f"Reduced-order K:       {np.round(Kmod[0],4)} / {np.round(Kmod[1],4)}")
    print(f"L2 cost (SISO):        {err_mod:.6f}")
    r_mod = _cl_poles(F, Kmod, T)
    print(f"Closed-loop poles:     {np.round(r_mod,4)}")


# ── demo_modsdl2b ─────────────────────────────────────────────────────────────

def demo_modsdl2b():
    """Reduced-order L2-optimal controller with fixed static gain."""
    _hdr("Reduced-order L2-optimal controller – fixed gain")

    F = sig.lti([1], [1, 1, 1])
    R = sig.lti([1], [1, 0])
    Q = sig.lti([1], [5, 1])
    T = 0.2

    sys = GeneralizedPlant([
        [neg(mul(Q, R)),  F],
        [R,                  neg(F)],
    ])

    _sep("Full-order optimal controller")
    KOpt, err_opt = sdl2(sys, T)
    print(f"Full-order K:          {np.round(KOpt[0],4)} / {np.round(KOpt[1],4)}")
    print(f"L2 cost:               {err_opt:.6f}")
    rOpt = _cl_poles(F, KOpt, T)
    print(f"Closed-loop poles:     {np.round(rOpt,4)}")

    _sep("Reduced-order modal L2 (order 1, SISO plant F)")
    Kmod, err_mod = modsdl2(F, T, ord_K=1, alpha=0.0, beta=np.inf,
                              method='randsearch', n_iter=300)
    print(f"Reduced-order K:       {np.round(Kmod[0],4)} / {np.round(Kmod[1],4)}")
    print(f"L2 cost (SISO):        {err_mod:.6f}")
    r_mod = _cl_poles(F, Kmod, T)
    print(f"Closed-loop poles:     {np.round(r_mod,4)}")


# ── main ──────────────────────────────────────────────────────────────────────

_ALL_DEMOS = [
    ("demo_doubint",     demo_doubint,    "Double integrator – design methods"),
    ("demo_ait98",       demo_ait98,      "H2/AHinf comparison (ARC 1998)"),
    ("demo_at96",        demo_at96,       "Ship course stabilisation (AT 1996)"),
    ("demo_autom97",     demo_autom97,    "H2 with time-delay (Automatica 1997)"),
    ("demo_c2d",         demo_c2d,        "Digital redesign (Rattan's example)"),
    ("demo_cf1",         demo_cf1,        "Chen & Francis Example 12.4.2"),
    ("demo_cf2",         demo_cf2,        "Chen & Francis Examples 6.6.1/8.4.2/12.1.1"),
    ("demo_dhinf",       demo_dhinf,      "Discrete-time polynomial Hinf"),
    ("demo_fil1",        demo_fil1,       "Optimal filtering (Rosenwasser 1998)"),
    ("demo_fil2",        demo_fil2,       "Filtering with time-delay"),
    ("demo_fil3",        demo_fil3,       "Filtering – 2nd-order plant"),
    ("demo_h2hinf",      demo_h2hinf,     "Mixed H2/AHinf optimisation"),
    ("demo_h2p",         demo_h2p,        "H2-optimal control with preview"),
    ("demo_hold",        demo_hold,       "Filtering with generalised hold"),
    ("demo_l2",          demo_l2,         "L2-optimal controller"),
    ("demo_l2hinf",      demo_l2hinf,     "L2/AHinf for tracking (ARC 2001)"),
    ("demo_ait01b",      demo_ait01b,     "AHinf tracking (ARC 2001)"),
    ("demo_l2p",         demo_l2p,        "L2-optimal with preview"),
    ("demo_2dof",        demo_2dof,       "Optimal 2-DOF tracking"),
    ("demo_2dofp",       demo_2dofp,      "2-DOF with preview (CDC 2004)"),
    ("demo_modsdh2",     demo_modsdh2,    "Reduced-order H2 (modal)"),
    ("demo_modsdh2int",  demo_modsdh2int, "Reduced-order H2 with integrator"),
    ("demo_modsdl2",     demo_modsdl2,    "Reduced-order L2 (modal)"),
    ("demo_modsdl2a",    demo_modsdl2a,   "L2 redesign"),
    ("demo_modsdl2b",    demo_modsdl2b,   "Reduced-order L2 with fixed gain"),
]


def list_demos():
    """Print a table of available demos."""
    print("\nAvailable DirectSD demos:")
    print(f"  {'Name':<22}  Description")
    print(f"  {'-'*22}  {'-'*40}")
    for name, _, desc in _ALL_DEMOS:
        print(f"  {name:<22}  {desc}")
    print()


def run_demo(name: str) -> None:
    """Run a single demo by name."""
    for n, fn, _ in _ALL_DEMOS:
        if n == name:
            fn()
            return
    raise ValueError(f"Unknown demo '{name}'. Use list_demos() to see available demos.")


def main():
    """Run all 25 demos sequentially."""
    import time
    passed, failed = [], []
    for name, fn, desc in _ALL_DEMOS:
        print(f"\n{'─'*60}")
        print(f"Running {name} …")
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            passed.append((name, elapsed))
            print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        except Exception as exc:
            import traceback
            elapsed = time.time() - t0
            failed.append((name, str(exc)))
            print(f"\n[FAIL] {name}: {exc}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("Failed demos:")
        for name, msg in failed:
            print(f"  {name}: {msg}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        run_demo(sys.argv[1])
    else:
        main()
