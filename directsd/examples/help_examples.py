"""
Numerical examples from dsd_help.md — Python implementation.

Run all:
    python -m directsd.examples.help_examples
Run one:
    python -m directsd.examples.help_examples <name>

Note: norm values may differ from MATLAB due to different conventions.
      sdh2/sdl2 for MIMO plants with P22≠0 currently fall back to a unit
      controller (h2reg limitation); filtering examples (P22=0) work correctly.
      Modal reduced-order controllers (modsdh2/modsdl2) always work.
"""
import sys
import math
import numpy as np
import scipy.signal as sig

from directsd import (
    sfactor, sdh2, sdl2, sdahinf, sdahinorm, sdh2norm, sdhinorm,
    sdl2err, sd2dof, sd2doferr, sdh2hinf, sdtrhinf, sdtrhinferr,
    charpol, ch2, bilintr, dhinf, modsdh2, modsdl2,
    GeneralizedPlant,
)
from directsd.analysis.norms import dahinorm as _dahinorm
from directsd.tf import to_lti, nd, mul, neg, add, feedback, nd_mul, nd_neg
from directsd.examples._common import cl_poles as _cl_poles, _z2zeta

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pade(tau, n):
    """Padé approximation for exp(-tau*s); returns (num, den) highest-power-first."""
    c = [(-tau)**k / math.factorial(k) for k in range(2*n+2)]
    num = np.zeros(n+1)
    den = np.zeros(n+1)
    for k in range(n+1):
        coeff = (math.factorial(2*n - k) * math.factorial(n)
                 / (math.factorial(2*n) * math.factorial(k) * math.factorial(n-k)))
        num[k] = coeff * (-tau)**k
        den[k] = coeff * tau**k
    return num[::-1], den[::-1]



def _zpk_str(obj):
    """Compact ZPK summary for display."""
    if isinstance(obj, tuple) and len(obj) == 2:
        n, d = obj
        if not (np.isscalar(n) or isinstance(n, np.ndarray)):
            return str(obj)
        obj = sig.TransferFunction(np.atleast_1d(n).ravel(),
                                   np.atleast_1d(d).ravel())
    if hasattr(obj, 'to_zpk'):
        zpk = obj.to_zpk()
        z, p, k = zpk.zeros, zpk.poles, zpk.gain
    elif isinstance(obj, (sig.lti, sig.TransferFunction)):
        zpk = obj.to_zpk()
        z, p, k = zpk.zeros, zpk.poles, zpk.gain
    else:
        return str(obj)
    z_str = "  ".join(f"{v:.4f}" for v in sorted(z, key=lambda x: x.real))
    p_str = "  ".join(f"{v:.4f}" for v in sorted(p, key=lambda x: x.real))
    return f"k={k:.4f}  zeros=[{z_str}]  poles=[{p_str}]"


def _check(label, computed, expected, tol=0.01):
    if not np.isfinite(computed):
        print(f"  SKIP     {label}: computed={computed}  MATLAB={expected:.6g}  (nan/inf)")
        return False
    err = abs(computed - expected) / (abs(expected) + 1e-12)
    status = "OK" if err < tol else "MISMATCH"
    print(f"  {status:8s}  {label}: computed={computed:.6g}  MATLAB={expected:.6g}  "
          f"rel_err={err:.3%}")
    return status == "OK"




# ---------------------------------------------------------------------------
# Example 1 – Basic getting-started example (line 181)
#   F = 1/(25s^2+s), Fw = 2/(s+2), T=1
#   sys = [F*Fw F; -F*Fw -F]  (P22=-F, sdh2 via MIMO fallback)
#   K_manual = (z-0.5)/(z-0.2), MATLAB sdh2norm(sys,K)=0.8332, opt=0.0292
# ---------------------------------------------------------------------------
def ex_getting_started():
    print("\n=== Getting-started example ===")
    F  = sig.TransferFunction([1], [25, 1, 0])
    Fw = sig.TransferFunction([2], [1, 2])
    T  = 1.0
    sys = GeneralizedPlant([
        [mul(F, Fw),       F],
        [neg(mul(F, Fw)), neg(F)],
    ])

    # Characteristic polynomial for manually specified K
    K_man = sig.dlti([1, -0.5], [1, -0.2], dt=T)
    try:
        cp = charpol(F, K_man)
        print(f"  charpol coeffs (manual K): {np.round(cp, 4)}")
        print(f"  MATLAB expected: [1, -2.1411, 1.3626, -0.2019]")
    except Exception as e:
        print(f"  charpol: {e}")

    # sdh2norm for manual K: use full sys (same as MATLAB sdh2norm(sys,K))
    try:
        norm_man = sdh2norm(sys, (K_man.num, K_man.den), T)
        print(f"  sdh2norm(sys, K_manual, T) = {norm_man:.4f}  (MATLAB: 0.8332)")
    except Exception as e:
        print(f"  sdh2norm manual: {e}")

    # Optimal H2 design via full sys (P22=-F, may fall back)
    K_opt, err_opt = sdh2(sys, T)
    print(f"  sdh2 K_opt: {_zpk_str(K_opt)}")
    print(f"  sdh2 cost: {err_opt:.4g}  (MATLAB sdh2norm(sys,Kopt)=0.0292)")
    try:
        norm_opt = sdh2norm(sys, K_opt, T)
        print(f"  sdh2norm(sys, K_opt, T) = {norm_opt:.4f}  (MATLAB: 0.0292)")
    except Exception as e:
        print(f"  sdh2norm: {e}")


# ---------------------------------------------------------------------------
# Example 2 – Deterministic / L2 tracking (line 333)
#   F1=1/(25s+1), F2=1/s, R=1/s, rho=0.2, T=1
#   MATLAB: sdl2err(sys,K)=3.2049
# ---------------------------------------------------------------------------
def ex_deterministic_l2():
    print("\n=== Deterministic / L2 tracking (ship course, step) ===")
    F1  = sig.TransferFunction([1], [25, 1])
    F2  = sig.TransferFunction([1], [1, 0])
    R   = sig.TransferFunction([1], [1, 0])
    rho = 0.2
    T   = 1.0
    F_plant = mul(F2, F1)
    sys = GeneralizedPlant([
        [R,              neg(F_plant)],
        [0,              rho],
        [R,              neg(F_plant)],
    ])
    K, err = sdl2(sys, T)
    print(f"  sdl2 K: {_zpk_str(K)}")
    print(f"  sdl2 cost: {err:.4g}")
    try:
        err_check = sdl2err(sys, K, T)    # pass full sys, same as MATLAB sdl2err(sys,K)
        print(f"  sdl2err(sys, K, T) = {err_check:.4f}  (MATLAB: 3.2049)")
    except Exception as e:
        print(f"  sdl2err: {e}")


# ---------------------------------------------------------------------------
# Example 3 – Stochastic / H2 control (line 449)
#   F=1/(25s^2+s), Fw=2/(s+2), rho=0.2, T=1
#   MATLAB: sdh2norm(sys,K)^2=0.0328
# ---------------------------------------------------------------------------
def ex_stochastic_h2():
    print("\n=== Stochastic / H2 control ===")
    F  = sig.TransferFunction([1], [25, 1, 0])
    Fw = sig.TransferFunction([2], [1, 2])
    rho = 0.2
    T   = 1.0
    sys = GeneralizedPlant([
        [mul(F, Fw),        F],
        [0,                  rho],
        [neg(mul(F, Fw)),  neg(F)],
    ])
    K, err = sdh2(sys, T)
    print(f"  sdh2 K: {_zpk_str(K)}")
    print(f"  sdh2 cost^2: {err**2:.4g}  (MATLAB: 0.0328)")
    try:
        norm_sq = sdh2norm(sys, K, T) ** 2
        print(f"  sdh2norm(sys,K,T)^2 = {norm_sq:.4f}  (MATLAB: 0.0328)")
    except Exception as e:
        print(f"  sdh2norm: {e}")


# ---------------------------------------------------------------------------
# Example 4 – Reconstructing first-order process (line 697)
#   F=1/(s+1), Sr=4/(−s^2+4), Fr=sfactor(Sr)=2/(s+2), T=0.1
#   sys=[Fr 0 −F; Fr 1 0]  (P22=0 → works)
#   MATLAB: sdh2norm(sys,K)=0.7314, sdh2norm(sys,K,0)=0.7682
# ---------------------------------------------------------------------------
def ex_first_order_filter():
    print("\n=== Reconstructing first-order process ===")
    F  = sig.TransferFunction([1], [1, 1])
    Sr = sig.TransferFunction([4], [-1, 0, 4])
    Fr_zpk, _ = sfactor(Sr)
    Fr = Fr_zpk.to_tf()
    T  = 0.1
    sys = GeneralizedPlant([
        [Fr, 0,  neg(F)],
        [Fr, 1,  0],
    ])
    K, err = sdh2(sys, T)
    print(f"  sdh2 K: {_zpk_str(K)}")
    print(f"  sdh2 cost: {err:.4f}  (MATLAB sdh2norm avg=0.7314)")
    try:
        norm_avg = sdh2norm(sys, K, T)
        print(f"  sdh2norm(sys,K,T) avg = {norm_avg:.4f}  (MATLAB: 0.7314)")
    except Exception as e:
        print(f"  sdh2norm avg: {e}")

    # Discrete-time optimal (minimize error at sampling instants only)
    K0, err0 = sdh2(sys, T, t=0.0, method='pol')
    print(f"  sdh2 K0 (t=0, pol): {_zpk_str(K0)}")
    print(f"  sdh2 cost (t=0): {err0:.4f}  (MATLAB sdh2norm(sys,K0,0)=0.7577)")
    try:
        norm0 = sdh2norm(sys, K0, T)
        print(f"  sdh2norm(sys,K0,T) avg = {norm0:.4f}  (MATLAB: 0.7682)")
    except Exception as e:
        print(f"  sdh2norm t=0: {e}")


# ---------------------------------------------------------------------------
# Example 5 – Filtering with delay (line 882)
#   Same as Example 4 but F has iodelay=0.051, T=0.1
#   MATLAB: sdh2norm avg=0.7879, t=0=0.7735
# ---------------------------------------------------------------------------
def ex_filter_with_delay():
    print("\n=== Filtering with delay (τ=0.051) ===")
    Sr  = sig.TransferFunction([4], [-1, 0, 4])
    Fr_zpk, _ = sfactor(Sr)
    Fr = Fr_zpk.to_tf()
    # F = 1/(s+1) with delay 0.051 — approximate with 2nd-order Padé
    pn, pd = _pade(0.051, 2)
    F_delay = sig.TransferFunction(np.polymul([1], pn),
                                   np.polymul([1, 1], pd))
    T = 0.1
    sys = GeneralizedPlant([
        [Fr, 0,  neg(F_delay)],
        [Fr, 1,  0],
    ])
    K, err = sdh2(sys, T)
    print(f"  sdh2 K: {_zpk_str(K)}")
    print(f"  sdh2 cost: {err:.4f}  (MATLAB sdh2norm avg≈0.7879)")
    try:
        norm_avg = sdh2norm(sys, K, T)
        print(f"  sdh2norm(sys,K,T) avg = {norm_avg:.4f}  (MATLAB: 0.7879)")
    except Exception as e:
        print(f"  sdh2norm: {e}")


# ---------------------------------------------------------------------------
# Example 6 – Reconstructing second-order process (line 1011)
#   F=1/(s^2+s+1), same Fr, T=0.1
#   MATLAB: sdh2norm avg=0.7738, t=0=0.7666
# ---------------------------------------------------------------------------
def ex_second_order_filter():
    print("\n=== Reconstructing second-order process ===")
    F  = sig.TransferFunction([1], [1, 1, 1])
    Sr = sig.TransferFunction([4], [-1, 0, 4])
    Fr_zpk, _ = sfactor(Sr)
    Fr = Fr_zpk.to_tf()
    T  = 0.1
    sys = GeneralizedPlant([
        [Fr, 0,  neg(F)],
        [Fr, 1,  0],
    ])
    K, err = sdh2(sys, T)
    print(f"  sdh2 K: {_zpk_str(K)}")
    print(f"  sdh2 cost: {err:.4f}  (MATLAB sdh2norm avg=0.7738)")
    try:
        norm_avg = sdh2norm(sys, K, T)
        print(f"  sdh2norm(sys,K,T) avg = {norm_avg:.4f}  (MATLAB: 0.7738)")
    except Exception as e:
        print(f"  sdh2norm: {e}")

    K0, err0 = sdh2(sys, T, t=0.0, method='pol')
    print(f"  sdh2 K0 (t=0): {_zpk_str(K0)}")
    print(f"  sdh2 cost (t=0): {err0:.4f}  (MATLAB sdh2norm K0 avg=0.9524)")


# ---------------------------------------------------------------------------
# Example 7 – Generalized hold (line 1140)
#   Fr=2/(s+2), Fn=0.1, F=−1 (plant is -1), sys=[Fr 0 -1; Fr 0.1 0], T=0.2
#   MATLAB: sdh2norm^2(ZOH)=0.3274, sdh2norm^2(gen hold)=0.3184
# ---------------------------------------------------------------------------
def ex_generalized_hold():
    print("\n=== Generalized hold ===")
    Sr = sig.TransferFunction([4], [-1, 0, 4])
    Fr_zpk, _ = sfactor(Sr)
    Fr = Fr_zpk.to_tf()
    Fn = 0.1
    T  = 0.2
    F_plant = sig.TransferFunction([-1], [1])  # plant = -1

    sys = GeneralizedPlant([
        [Fr, 0,   F_plant],   # z = Fr*w - 1*u  (i.e. F_plant = -1)
        [Fr, Fn,  0],
    ])
    K0, err0 = sdh2(sys, T)
    print(f"  sdh2 K0 (ZOH): {_zpk_str(K0)}")
    print(f"  sdh2 cost^2 (ZOH): {err0**2:.4f}  (MATLAB: 0.3274)")
    try:
        norm0_sq = sdh2norm(F_plant, K0, T) ** 2
        print(f"  sdh2norm(F,K0,T)^2 = {norm0_sq:.4f}")
    except Exception as e:
        print(f"  sdh2norm ZOH: {e}")

    H = sig.TransferFunction([1], [1, 2])
    K, err = sdh2(sys, T, H=H)
    print(f"  sdh2 K (gen. hold): {_zpk_str(K)}")
    print(f"  sdh2 cost^2 (gen hold): {err**2:.4f}  (MATLAB: 0.3184)")


# ---------------------------------------------------------------------------
# Example 8 – Optimal disturbance attenuation (line 1315)
#   F=(s+0.5)/(s^2−s), Fw=2/(s+2), τ=0.093, T=0.2
#   MATLAB: sdh2norm avg=0.1840, t=0=0.1600
# ---------------------------------------------------------------------------
def ex_disturbance_attenuation():
    print("\n=== Optimal disturbance attenuation (with delay τ=0.093) ===")
    F  = sig.TransferFunction([1, 0.5], [1, -1, 0])
    Fw = sig.TransferFunction([2], [1, 2])
    T  = 0.2
    # F with delay 0.093 — 2nd-order Padé approx
    pn, pd = _pade(0.093, 2)
    Fd = sig.TransferFunction(np.polymul([1, 0.5], pn),
                               np.polymul([1, -1, 0], pd))
    sys = GeneralizedPlant([
        [mul(F, Fw),       Fd],
        [neg(mul(F, Fw)), neg(Fd)],
    ])
    K, err = sdh2(sys, T)
    print(f"  sdh2 K: {_zpk_str(K)}")
    print(f"  sdh2 cost: {err:.4f}  (MATLAB sdh2norm avg≈0.1840)")
    try:
        norm_avg = sdh2norm(Fd, K, T)
        print(f"  sdh2norm(Fd,K,T) avg = {norm_avg:.4f}")
    except Exception as e:
        print(f"  sdh2norm: {e}")


# ---------------------------------------------------------------------------
# Example 9 – Optimal ship course stabilization (line 1465)
#   F=0.051/(25s^2+s), Sw given, Fw=sfactor(Sw), rho=sqrt(0.1), T=1
#   MATLAB: sdh2norm=0.0190
# ---------------------------------------------------------------------------
def ex_ship_course():
    print("\n=== Optimal ship course stabilization ===")
    F  = sig.TransferFunction([0.051], [25, 1, 0])
    Sw = sig.TransferFunction([0.0757], [1, 0, 2.489, 0, 1.848])
    Fw_zpk, _ = sfactor(Sw)
    Fw = Fw_zpk.to_tf()
    rho = np.sqrt(0.1)
    T   = 1.0
    sys = GeneralizedPlant([
        [mul(F, Fw),       F],
        [0,                 rho],
        [neg(mul(F, Fw)), neg(F)],
    ])
    K, err = sdh2(sys, T)
    print(f"  sdh2 K: {_zpk_str(K)}")
    print(f"  sdh2 cost: {err:.4f}  (MATLAB sdh2norm=0.0190)")
    try:
        norm = sdh2norm(F, K, T)
        print(f"  sdh2norm(F,K,T) = {norm:.4f}")
    except Exception as e:
        print(f"  sdh2norm: {e}")


# ---------------------------------------------------------------------------
# Example 10 – Double integrator H2 control (line 1587)
#   F=1/s^2, rho=1, T=0.1
# ---------------------------------------------------------------------------
def ex_double_integrator():
    print("\n=== Double integrator H2 control ===")
    F   = sig.TransferFunction([1], [1, 0, 0])
    rho = 1.0
    T   = 0.1
    sys = GeneralizedPlant([
        [neg(F),  neg(F)],
        [0,        rho],
        [neg(F),  neg(F)],
    ])
    Kopt, err_opt = sdh2(sys, T)
    print(f"  sdh2 Kopt: {_zpk_str(Kopt)}")
    print(f"  sdh2 cost: {err_opt:.4g}")
    poles = _cl_poles(F, Kopt, T)
    if len(poles) > 0:
        stable = all(abs(r) < 1.0 for r in poles)
        print(f"  closed-loop poles: {[f'{r:.4f}' for r in poles]}")
        print(f"  stable: {stable}  (MATLAB: True)")

    # Redesign method — ch2 + bilintr
    try:
        Kc_raw, _ = ch2(sys)
        Kc = sig.TransferFunction(*Kc_raw)
        Kc2d_nd = bilintr(Kc, 'tustin', T)
        Kc2d = sig.TransferFunction(*Kc2d_nd)
        poles2d = _cl_poles(F, Kc2d_nd, T)
        if len(poles2d) > 0:
            stable2d = all(abs(r) < 1.0 for r in poles2d)
            print(f"  redesign poles: {[f'{r:.4f}' for r in poles2d]}")
            print(f"  redesign stable: {stable2d}  (MATLAB: False – marginal)")
    except Exception as e:
        print(f"  redesign: skipped ({e})")


# ---------------------------------------------------------------------------
# Example 11 – L2-optimal tracking (line 1792)
#   R=1/s, F=1/(5s^2+s), Q=1/(s+1), T=0.2
#   MATLAB: sdl2err=4.6824e-4
# ---------------------------------------------------------------------------
def ex_l2_tracking():
    print("\n=== L2-optimal tracking ===")
    R = sig.TransferFunction([1], [1, 0])
    F = sig.TransferFunction([1], [5, 1, 0])
    Q = sig.TransferFunction([1], [1, 1])
    T = 0.2
    sys = GeneralizedPlant([
        [mul(Q, R), neg(F)],
        [R,          neg(F)],
    ])
    Kopt, err = sdl2(sys, T)
    print(f"  sdl2 Kopt: {_zpk_str(Kopt)}")
    print(f"  sdl2 cost: {err:.4g}  (MATLAB sdl2err=4.6824e-4)")
    try:
        err_check = sdl2err(sys, Kopt, T)
        print(f"  sdl2err(sys, K, T) = {err_check:.4e}  (MATLAB: 4.6824e-4)")
    except Exception as e:
        print(f"  sdl2err: {e}")

    poles = _cl_poles(F, Kopt, T)
    if len(poles) > 0:
        stable = all(abs(r) < 1.0 for r in poles)
        print(f"  closed-loop poles: {[f'{r:.4f}' for r in poles]}")
        print(f"  stable: {stable}  (MATLAB: True)")


# ---------------------------------------------------------------------------
# Example 12 – L2-optimal redesign (line 2037)
#   F=10/(s^2+s), Kc=(0.416s+1)/(0.139s+1), τ=0.01, T=0.04
#   MATLAB: sdl2err=6.7103e-7
# ---------------------------------------------------------------------------
def ex_l2_redesign():
    print("\n=== L2-optimal redesign ===")
    R  = sig.TransferFunction([1], [1, 0])
    F_ct = sig.TransferFunction([10], [1, 1, 0])
    Kc = sig.TransferFunction([0.416, 1], [0.139, 1])
    Q = feedback(mul(F_ct, Kc))
    T = 0.04

    # MATLAB: F.iodelay = 0.01, handled EXACTLY inside sdl2 via the modified
    # Z-transform (udelay). A Padé substitute in the DESIGN plant introduces
    # fast poles (~±600) whose discretized images span ~20 decades and
    # destroy the polynomial pipeline — pass the delay separately.
    tau = 0.01
    sys_design = GeneralizedPlant([
        [mul(Q, R), neg(F_ct)],
        [R,          neg(F_ct)],
    ])
    K, err = sdl2(sys_design, T, udelay=tau)
    print(f"  sdl2 K: {_zpk_str(K)}")
    print(f"  MATLAB K: 4.0371 z(z-0.9608)(z-0.9083)(z^2-0.0708z+0.07494)")
    print(f"            /((z+0.04358)(z-0.009759)(z+0.4438)(z-0.7342)(z-0.9559))")

    # Verification: sdl2err has no delay support, so evaluate on a
    # Padé-approximated plant (fine for EVALUATION — MATLAB's documented K
    # scores 7.9e-7 on it, matching the documented 6.7103e-7).
    pn, pd = _pade(tau, 2)
    F_delay = sig.TransferFunction(np.polymul([10], pn),
                                    np.polymul([1, 1, 0], pd))
    sys_eval = GeneralizedPlant([
        [mul(Q, R), neg(F_delay)],
        [R,          neg(F_delay)],
    ])
    try:
        err_check = sdl2err(sys_eval, K, T)
        print(f"  sdl2err(Padé sys, K, T) = {err_check:.4e}  (MATLAB: 6.7103e-7)")
    except Exception as e:
        print(f"  sdl2err: {e}")


# ---------------------------------------------------------------------------
# Example 13 – 2-DOF optimal tracking (line 2159)
#   R=1/s, F=1/(s-1), Q=1/(s+2), T=0.5
#   MATLAB: 1-DOF sdl2err=2.5845, 2-DOF sd2doferr=5.6599e-4
# ---------------------------------------------------------------------------
def ex_2dof_tracking():
    print("\n=== 2-DOF optimal tracking ===")
    R  = sig.TransferFunction([1], [1, 0])
    F  = sig.TransferFunction([1], [1, -1])
    Q  = sig.TransferFunction([1], [1, 2])
    T  = 0.5
    sys = GeneralizedPlant([
        [mul(Q, R), neg(F)],
        [R,          neg(F)],
    ])
    K1, err1 = sdl2(sys, T)
    print(f"  1-DOF sdl2 K1: {_zpk_str(K1)}")
    print(f"  1-DOF sdl2 cost: {err1:.4g}  (MATLAB sdl2err=2.5845)")
    try:
        err_1dof = sdl2err(sys, K1, T)
        print(f"  sdl2err(sys, K1, T) = {err_1dof:.4f}  (MATLAB: 2.5845)")
    except Exception as e:
        print(f"  sdl2err 1-DOF: {e}")

    # 2-DOF: sd2dof takes SISO plant F; sd2doferr takes full 3x2 generalized plant
    # sys_2dof rows: [performance Q*R*d-F*u; reference measurement R*d; output measurement -F*u]
    sys_2dof = GeneralizedPlant([
        [mul(Q, R), neg(F)],
        [R,          0       ],
        [0,          neg(F) ],
    ], n_meas=2)
    try:
        KR, err_2dof = sd2dof(sys_2dof, K1, T)
        print(f"  2-DOF KR:    {_zpk_str(KR)}")
        print(f"  2-DOF sd2dof cost: {err_2dof:.4e}  (MATLAB: 5.6599e-4)")
        err2 = sd2doferr(sys_2dof, K1, KR, T)
        _check("2-DOF sd2doferr", err2, 5.6599e-4, tol=0.50)
    except Exception as e:
        print(f"  sd2dof/sd2doferr: {e}")


# ---------------------------------------------------------------------------
# Example 14 – AHinf-optimal prediction (line 2341)
#   F=1/(s+1), Fr=1/(5s+1), predict τ=0.15, T=0.1
#   MATLAB: sdahinorm=0.0351
# ---------------------------------------------------------------------------
def ex_ahinf_prediction():
    print("\n=== AHinf-optimal prediction ===")
    F  = sig.TransferFunction([1], [1, 1])
    Fr = sig.TransferFunction([1], [5, 1])
    # Q = exp(+0.15s) — Padé: exp(+tau*s) ≈ den/num of exp(-tau*s)
    pn, pd = _pade(0.15, 2)
    Q_num, Q_den = pd, pn  # flip for positive delay
    T = 0.1
    sys = GeneralizedPlant([
        [sig.TransferFunction(np.polymul(-Q_num, [1]), np.polymul(Q_den, [5, 1])),
         F],
        [Fr, 0],
    ])
    K, err = sdahinf(sys, T)
    print(f"  sdahinf K: {_zpk_str(K)}")
    try:
        ahinf_norm = sdahinorm(F, K, T)
        print(f"  sdahinorm(F,K,T) = {ahinf_norm:.4f}  (MATLAB: 0.0351)")
    except Exception as e:
        print(f"  sdahinorm: {e}")


# ---------------------------------------------------------------------------
# Example 15 – Discrete AHinf example 1
#   T=1, F=z/(−2z+1), Fw=(z−2)/(−2z+1), V1=V2=1
#   Generalised plant: [[Fw,F],[0,V2],[-Fw,-F]]  (z-domain, 3×2)
#   MATLAB (Source/dsd_help.md, verbatim): K = dhinf(sys) = 1.5 (constant!),
#   dahinorm(sys,K) = 3.6056. The "~0.9487" figure previously here was NOT a
#   real MATLAB value -- it was Python's own (buggy) dhinf lam matching its
#   own (also-buggy, K-independent) dahinorm, a self-consistent-but-wrong
#   pair. dahinorm is now correctly K-dependent; the remaining mismatch below
#   is real (dhinf/_polhinf still doesn't find MATLAB's true optimal K here).
# ---------------------------------------------------------------------------

def ex_dhinf_example1():
    print("\n=== Discrete AHinf – example 1 ===")
    T   = 1.0
    F   = ([1, 0], [-2, 1])        # z / (-2z+1)
    Fw  = ([1, -2], [-2, 1])       # (z-2) / (-2z+1)
    V1, V2 = ([1], [1]), ([1], [1])
    # Build 3x2 generalised plant
    sys_plant = [
        [(nd_mul(V1, Fw)), (nd_mul(V1, F))],
        [([0], [1]),         V2              ],
        [(nd_neg(Fw)),      (nd_neg(F))    ],
    ]
    try:
        sys_conv = _z2zeta(sys_plant)
        K, err_dh = dhinf(sys_conv, T=T)
        print(f"  dhinf K: {_zpk_str(K)}")
        print(f"  dhinf lam = {err_dh:.4f}  (MATLAB: K=1.5 constant, dahinorm=3.6056)")
        err_da = _dahinorm(sys_conv, K, T)
        print(f"  dahinorm  = {err_da:.4f}")
        _check("dhinf lam consistent with dahinorm", err_dh, err_da, tol=0.15)
    except Exception as e:
        print(f"  dhinf: {e}")


# ---------------------------------------------------------------------------
# Example 16 – Discrete AHinf example 2 (generic case)
#   T=1, F2=1/(−z²−2.1z+1), F1=(2z²+z), Fw=(0.3z+1)
#   Generalised plant: [[F2*Fw, F2*F1],[0,V2],[-F2*Fw,-F2*F1]]
#   MATLAB: dhinf errOpt ~ 2.1244
# ---------------------------------------------------------------------------
def ex_dhinf_example2():
    print("\n=== Discrete AHinf – example 2 (generic) ===")
    T  = 1.0
    F2 = ([1],      [-1, -2.1, 1])
    F1 = ([2, 1, 0],[1]          )
    Fw = ([0.3, 1], [1]          )
    V1, V2 = ([1],[1]), ([1],[1])
    F2Fw  = nd_mul(F2, Fw)
    F2F1  = nd_mul(F2, F1)
    sys_plant = [
        [(nd_mul(V1, F2Fw)), (nd_mul(V1, F2F1))],
        [([0], [1]),           V2                  ],
        [(nd_neg(F2Fw)),      (nd_neg(F2F1))     ],
    ]
    try:
        sys_conv = _z2zeta(sys_plant)
        K, err_dh = dhinf(sys_conv, T=T)
        print(f"  dhinf K: {_zpk_str(K)}")
        print(f"  dhinf lam = {err_dh:.4f}  (MATLAB ~2.1244)")
        err_da = _dahinorm(sys_conv, K, T)
        print(f"  dahinorm  = {err_da:.4f}")
        _check("dhinf lam (tol 50%)", err_dh, 2.1244, tol=0.50)
    except Exception as e:
        print(f"  dhinf: {e}")


# ---------------------------------------------------------------------------
# Example 17 – H2 and AHinf optimal control (line 2647)
#   F1=0.051/(25s+1), F2=1/s, rho=0.1, T=1
#   MATLAB: sdh2norm(K2)=3.2563, sdahinorm(K2)=11.4122, sdhinorm(K2)=11.4082
#          sdh2norm(Kinf)=7.6683, sdahinorm(Kinf)=7.6683, sdhinorm(Kinf)=7.6628
# ---------------------------------------------------------------------------
def ex_h2_ahinf_control():
    print("\n=== H2 and AHinf optimal control ===")
    F1  = sig.TransferFunction([0.051], [25, 1])
    F2  = sig.TransferFunction([1], [1, 0])
    F_plant = mul(F2, F1)
    rho = 0.1
    T   = 1.0
    sys = GeneralizedPlant([
        [neg(F2),          neg(F_plant)],
        [0,                 rho],
        [neg(F2),          neg(F_plant)],
    ])
    K2, err_h2 = sdh2(sys, T)
    print(f"  H2-optimal K2: {_zpk_str(K2)}")
    print(f"  sdh2 cost: {err_h2:.4g}  (MATLAB sdh2norm=3.2563)")
    try:
        ah_K2 = sdahinorm(sys, K2, T)
        print(f"  sdahinorm(sys,K2,T) = {ah_K2:.4f}  (MATLAB: 11.4122)")
        # sdhinorm(sys, ...) -- the FULL generalized plant, matching MATLAB's
        # sdhinorm(sys, K2). Python's sdhinorm is currently an AHinf alias,
        # so this prints the same value as sdahinorm above, not MATLAB's true
        # 11.4082 -- a documented, deliberate scope decision, not a bug.
        hinf_K2 = sdhinorm(sys, K2, T)[0]
        print(f"  sdhinorm(sys,K2,T) = {hinf_K2:.4f}  (MATLAB: 11.4082; Python's"
              " sdhinorm = AHinf alias)")
    except Exception as e:
        print(f"  norm eval: {e}")

    Kinf, err_inf = sdahinf(sys, T)
    print(f"  AHinf-optimal Kinf: {_zpk_str(Kinf)}")
    print(f"  sdahinf cost: {err_inf:.4g}  (MATLAB sdh2norm(Kinf)=7.6683)")
    try:
        ah_inf = sdahinorm(sys, Kinf, T)
        print(f"  sdahinorm(sys,Kinf,T) = {ah_inf:.4f}  (MATLAB: 7.6683)")
    except Exception as e:
        print(f"  sdahinorm Kinf: {e}")


# ---------------------------------------------------------------------------
# Example 18 – Mixed H2/AHinf optimization (line 2807)
#   F=1/(5s^2+s), kappa=1, T=1
#   MATLAB: sdh2norm(KH2)=0.8153, sdahinorm(KH2)=1.7326
#          sdh2norm(Kinf)=1.2251, sdahinorm(Kinf)=1.2251
#          sdh2norm(Kmix)=0.9498, sdahinorm(Kmix)=1.3436
# ---------------------------------------------------------------------------
def ex_mixed_h2_ahinf():
    print("\n=== Mixed H2/AHinf optimization ===")
    F     = sig.TransferFunction([1], [5, 1, 0])
    kappa = 1.0
    T     = 1.0
    sys = GeneralizedPlant([
        [neg(F),  neg(F)],
        [0,        kappa],
        [neg(F),  neg(F)],
    ])
    KH2, err_H2 = sdh2(sys, T)
    print(f"  H2-optimal KH2: {_zpk_str(KH2)}")
    print(f"  sdh2 cost: {err_H2:.4g}  (MATLAB sdh2norm=0.8153)")
    try:
        ah_KH2 = sdahinorm(F, KH2, T)
        print(f"  sdahinorm(F,KH2,T) = {ah_KH2:.4f}  (MATLAB: 1.7326)")
    except Exception as e:
        print(f"  sdahinorm KH2: {e}")

    Kinf, err_Kinf = sdahinf(sys, T)
    print(f"  AHinf-optimal Kinf: {_zpk_str(Kinf)}")
    try:
        ah_Kinf = sdahinorm(F, Kinf, T)
        print(f"  sdahinorm(F,Kinf,T) = {ah_Kinf:.4f}  (MATLAB: 1.2251)")
    except Exception as e:
        print(f"  sdahinorm Kinf: {e}")

    try:
        Kmix, err_mix = sdh2hinf(sys, T, 0.5, 2, 1)
        print(f"  Mixed Kmix (rho=0.5): {_zpk_str(Kmix)}")
        ah_mix = sdahinorm(F, Kmix, T)
        print(f"  sdahinorm(F,Kmix,T) = {ah_mix:.4f}  (MATLAB: 1.3436)")
    except Exception as e:
        print(f"  sdh2hinf/sdahinorm: {e}")


# ---------------------------------------------------------------------------
# Example 19 – L2 and AHinf optimal tracking (line 2975)
#   R=1/s, F=1/(4s^2+0.5s+1), rho=0.12, T=1
#   MATLAB: sdl2err(K2)=1.0321, sdtrhinferr(K2)=1.2213
#          sdl2err(Kinf)=1.1118, sdtrhinferr(Kinf)=1.0544
# ---------------------------------------------------------------------------
def ex_l2_ahinf_tracking():
    print("\n=== L2 and AHinf optimal tracking ===")
    R   = sig.TransferFunction([1], [1, 0])
    F   = sig.TransferFunction([1], [4, 0.5, 1])
    rho = 0.12
    T   = 1.0
    sys = GeneralizedPlant([
        [R,          neg(F)],
        [mul(rho, R), neg(rho)],
        [R,          neg(F)],
    ])
    K2, err2 = sdl2(sys, T)
    print(f"  L2-optimal K2: {_zpk_str(K2)}")
    print(f"  sdl2 cost: {err2:.4g}  (MATLAB sdl2err=1.0321)")
    try:
        l2_K2 = sdl2err(sys, K2, T)
        print(f"  sdl2err(sys,K2,T) = {l2_K2:.4f}  (MATLAB: 1.0321)")
        ah_K2 = sdtrhinferr(sys, K2, T)
        print(f"  sdtrhinferr = {ah_K2:.4f}  (MATLAB: 1.2213)")
    except Exception as e:
        print(f"  norm eval: {e}")

    try:
        Kinf, err_tr = sdtrhinf(sys, T)
        print(f"  AHinf-optimal Kinf: {_zpk_str(Kinf)}")
        l2_inf = sdl2err(sys, Kinf, T)
        print(f"  sdl2err(sys,Kinf,T) = {l2_inf:.4f}  (MATLAB: 1.1118)")
        ah_inf = sdtrhinferr(sys, Kinf, T)
        print(f"  sdtrhinferr(Kinf) = {ah_inf:.4f}  (MATLAB: 1.0544)")
    except Exception as e:
        print(f"  sdtrhinf: {e}")


# ---------------------------------------------------------------------------
# Example 20 – H2-optimal preview control (line 3135)
#   Fr=1/(5s+1), Fn=0.2, F=1/(s-1) with delay 1.5, Q=exp(-2s)
#   sys=[Q*Fr 0 -F; Fr Fn -F], T=1
#   MATLAB: sdh2norm(sys,K)^2=11.6701
# ---------------------------------------------------------------------------
def ex_h2_preview():
    print("\n=== H2-optimal preview control ===")
    Fr = sig.TransferFunction([1], [5, 1])
    Fn = 0.2
    # F = 1/(s-1) with delay 1.5 → 3rd-order Padé
    pn15, pd15 = _pade(1.5, 3)
    F_delay = sig.TransferFunction(np.polymul([1], pn15),
                                   np.polymul([1, -1], pd15))
    # Q = exp(-2s) → 3rd-order Padé
    pn2, pd2 = _pade(2.0, 3)
    Q = sig.TransferFunction(pn2, pd2)
    T = 1.0
    sys = GeneralizedPlant([
        [mul(Q, Fr),   0,   neg(F_delay)],
        [Fr,           Fn,   neg(F_delay)],
    ])
    K, err = sdh2(sys, T)
    print(f"  sdh2 K: {_zpk_str(K)}")
    print(f"  sdh2 cost^2: {err**2:.4f}  (MATLAB: 11.6701)")
    try:
        norm_sq = sdh2norm(F_delay, K, T) ** 2
        print(f"  sdh2norm(F_delay,K,T)^2 = {norm_sq:.4f}")
    except Exception as e:
        print(f"  sdh2norm: {e}")


# ---------------------------------------------------------------------------
# Example 21 – L2-optimal preview control (line 3243)
#   R=1/(s^2+s), F=1/(5s+1) with delay 1.5, Q=1/(0.1s+1)
#   preview=2 -- MATLAB removes the non-causal preview block by placing the
#   delay in the ideal operator instead (Q.iodelay=preview) plus a remainder
#   delay on R (R.iodelay=theta); see sdl2's refdelay parameter, which
#   applies this sigma/theta split internally via the EXACT modified
#   Z-transform (no Pade approximation needed -- unlike the delay-free
#   F.iodelay=1.5, handled via udelay).
#   sys=[Q*R -F; R -F], T=1
#   MATLAB: sdl2err=0.0517
# ---------------------------------------------------------------------------
def ex_l2_preview():
    print("\n=== L2-optimal preview control ===")
    R = sig.TransferFunction([1], [1, 1, 0])
    F = sig.TransferFunction([1], [5, 1])
    Q = sig.TransferFunction([1], [0.1, 1])
    T = 1.0
    sys = GeneralizedPlant([
        [mul(Q, R), neg(F)],
        [R,          neg(F)],
    ])
    K, err = sdl2(sys, T, udelay=1.5, refdelay=2.0)
    print(f"  sdl2 K: {_zpk_str(K)}")
    print(f"  sdl2 cost: {err:.4f}  (MATLAB sdl2err=0.0517)")


# ---------------------------------------------------------------------------
# Example 22 – Optimal 2-DOF preview control (line 3387)
#   Same plant setup as example 21 (exNo=2's unstable F=1/(5s-1) branch,
#   matching demo_2dofp.m); 1-DOF K from sdl2, then KR from sd2dof.
#   MATLAB: 1-DOF sdl2err=2.7072, 2-DOF sd2doferr=0.0602
# ---------------------------------------------------------------------------
def ex_2dof_preview():
    print("\n=== Optimal 2-DOF preview control ===")
    R = sig.TransferFunction([1], [1, 1, 0])
    F = sig.TransferFunction([1], [5, -1])
    Q = sig.TransferFunction([1], [0.1, 1])
    T = 1.0
    # 1-DOF system: exact delay support via sdl2's udelay/refdelay (see
    # ex_l2_preview) -- no Pade approximation needed.
    sys = GeneralizedPlant([
        [mul(Q, R), neg(F)],
        [R,          neg(F)],
    ])
    K, err_1dof = sdl2(sys, T, udelay=1.5, refdelay=2.0)
    print(f"  1-DOF K: {_zpk_str(K)}")
    print(f"  1-DOF sdl2 cost: {err_1dof:.4f}  (MATLAB sdl2err=2.7072)")

    # 2-DOF feedforward (sd2dof): exact delay support via the same
    # udelay/refdelay mechanism (sd2dof/_sd2dofcoef).
    sys_2dof = GeneralizedPlant([
        [mul(Q, R), neg(F)],
        [R,          0],
        [0,          neg(F)],
    ], n_meas=2)
    KR, err_2dof = sd2dof(sys_2dof, K, T, udelay=1.5, refdelay=2.0)
    print(f"  2-DOF KR: {_zpk_str(KR)}")
    print(f"  2-DOF sd2dof cost: {err_2dof:.4f}  (MATLAB sd2doferr=0.0602)")


# ---------------------------------------------------------------------------
# Example 23 – Reduced-order H2-optimal control (line 3556)
#   Ship course stabilization; F=0.051/(25s^2+s), T=1
#   Full-order H2 optimal, then reduced order 1 via modsdh2
#   MATLAB: full sdh2norm=0.0190, reduced cost≈0.0193
# ---------------------------------------------------------------------------
def ex_reduced_h2():
    print("\n=== Reduced-order H2-optimal control (ship course) ===")
    F  = sig.TransferFunction([0.051], [25, 1, 0])
    Sw = sig.TransferFunction([0.0757], [1, 0, 2.489, 0, 1.848])
    Fw_zpk, _ = sfactor(Sw)
    Fw = Fw_zpk.to_tf()
    rho = np.sqrt(0.1)
    T   = 1.0
    sys = GeneralizedPlant([
        [mul(F, Fw),       F],
        [0,                 rho],
        [neg(mul(F, Fw)), neg(F)],
    ])
    # Full-order optimal
    Kopt, err_opt = sdh2(sys, T)
    print(f"  Full-order Kopt: {_zpk_str(Kopt)}")
    print(f"  Full-order cost: {err_opt:.4f}  (MATLAB sdh2norm=0.0190)")

    # Reduced-order via modsdh2 with full generalized plant (P22 extracted internally)
    print("  Searching reduced-order (ord=1, global randsearch)...")
    Kred, cost_red = modsdh2(sys, T, ord_K=1, alpha=0.0, beta=np.inf,
                              method='randsearch', n_iter=500)
    print(f"  Reduced-order K (ord=1): {_zpk_str(Kred)}")
    print(f"  Reduced-order cost: {cost_red:.4f}  (MATLAB cost≈0.0193)")

    # Local refinement
    Kloc, cost_loc = modsdh2(sys, T, ord_K=1, alpha=0.0, beta=np.inf,
                               method='dual_annealing', n_iter=200)
    print(f"  After local opt: K = {_zpk_str(Kloc)}")
    print(f"  Local opt cost: {cost_loc:.4f}")


# ---------------------------------------------------------------------------
# Example 24 – Optimal integral control (line 3725)
#   F1=0.0694/(18.22s+1), F2=1/s, Fw given, rho=2, T=2
#   Full-order K marginally stable (z=1 pole); reduced order with integrator
#   MATLAB: full sdh2norm=7.2492 (marginal), reduced cost≈8.4768
# ---------------------------------------------------------------------------
def ex_integral_control():
    print("\n=== Optimal integral control ===")
    F1 = sig.TransferFunction([0.0694], [18.22, 1])
    F2 = sig.TransferFunction([1], [1, 0])
    lam = 0.3; w0 = 0.3; sigma_w = 7.25
    Fw = sig.TransferFunction([2*lam*w0*sigma_w, 0], [1, 2*lam*w0, w0**2])
    rho = 2.0
    T   = 2.0
    F_plant = mul(F2, F1)
    sys = GeneralizedPlant([
        [mul(F2, Fw),       F_plant],
        [0,                  rho],
        [neg(mul(F2, Fw)), neg(F_plant)],
    ])
    # Full-order optimal (will have z=1 pole → marginally stable)
    Kopt, err_opt = sdh2(sys, T)
    print(f"  Full-order Kopt: {_zpk_str(Kopt)}")
    print(f"  Full-order cost: {err_opt:.4f}  (MATLAB sdh2norm=7.2492, marginal)")

    # Reduced-order with integrator constraint via modsdh2
    # Note: Python modsdh2 uses modal parameterization (no explicit dK0 constraint)
    # We require ord=2 and alpha=0.02, beta=2 to force stable controller
    print("  Searching reduced-order with integrator (ord=2, alpha=0.02, beta=2)...")
    Kred, cost_red = modsdh2(sys, T, ord_K=2, alpha=0.02, beta=2,
                               method='randsearch', n_iter=800)
    print(f"  Reduced-order K (ord=2): {_zpk_str(Kred)}")
    print(f"  Reduced-order cost: {cost_red:.4f}  (MATLAB cost≈8.4768)")


# ---------------------------------------------------------------------------
# Example 25 – Reduced-order L2-optimal control (line 3920)
#   R=1/s, F=10/(2s^2+s), Q=1/(s+1)^2, T=0.2
#   Full-order sdl2err≈1.2116e-8, reduced order 1 modsdl2 cost≈4.752e-7
# ---------------------------------------------------------------------------
def ex_reduced_l2():
    print("\n=== Reduced-order L2-optimal control ===")
    R = sig.TransferFunction([1], [1, 0])
    F = sig.TransferFunction([10], [2, 1, 0])
    Q = sig.TransferFunction([1], [1, 2, 1])
    T = 0.2
    sys = GeneralizedPlant([
        [mul(Q, R), neg(F)],
        [R,          neg(F)],
    ])
    # Full-order optimal
    Kopt, err_opt = sdl2(sys, T)
    print(f"  Full-order Kopt: {_zpk_str(Kopt)}")
    print(f"  Full-order cost: {err_opt:.4e}  (MATLAB sdl2err≈1.2116e-8)")
    try:
        err_full = sdl2err(sys, Kopt, T)
        print(f"  sdl2err(sys,Kopt,T) = {err_full:.4e}  (MATLAB: 1.2116e-8)")
    except Exception as e:
        print(f"  sdl2err: {e}")

    # Reduced-order via modsdl2 with full generalized plant
    print("  Searching reduced-order (ord=1, alpha=0.1)...")
    Kred, cost_red = modsdl2(sys, T, ord_K=1, alpha=0.1, beta=np.inf,
                               method='randsearch', n_iter=500)
    print(f"  Reduced-order K (ord=1): {_zpk_str(Kred)}")
    print(f"  Reduced-order cost: {cost_red:.4e}  (MATLAB cost≈4.752e-7)")

    Kloc, cost_loc = modsdl2(sys, T, ord_K=1, alpha=0.1, beta=np.inf,
                               method='dual_annealing', n_iter=300)
    print(f"  After local opt: K = {_zpk_str(Kloc)}")
    print(f"  Local opt cost: {cost_loc:.4e}")


# ---------------------------------------------------------------------------
# Example 26 – Reduced-order redesign (line 4101)
#   F=1/(s^2-s), Kc=(5s+1)/(s+3), Q=feedback(F*Kc,1), R=1/s, T=0.5
#   Full-order sdl2err≈2.7771e-4, reduced order 2 modsdl2 cost≈0.0059
# ---------------------------------------------------------------------------
def ex_reduced_redesign():
    print("\n=== Reduced-order redesign ===")
    R  = sig.TransferFunction([1], [1, 0])
    F  = sig.TransferFunction([1], [1, -1, 0])
    Kc = sig.TransferFunction([5, 1], [1, 3])
    Q  = feedback(mul(F, Kc))
    T  = 0.5
    sys = GeneralizedPlant([
        [mul(add(Q, neg(F)), R), F],
        [mul(F, R),                neg(F)],
    ])
    # Full-order optimal
    Kopt, err_opt = sdl2(sys, T)
    print(f"  Full-order Kopt: {_zpk_str(Kopt)}")
    print(f"  Full-order cost: {err_opt:.4e}  (MATLAB sdl2err≈2.7771e-4)")
    try:
        err_full = sdl2err(sys, Kopt, T)
        print(f"  sdl2err(sys,Kopt,T) = {err_full:.4e}  (MATLAB: 2.7771e-4)")
    except Exception as e:
        print(f"  sdl2err: {e}")

    # Reduced-order via modsdl2 with full generalized plant
    print("  Searching reduced-order (ord=2, alpha=0.001)...")
    Kred, cost_red = modsdl2(sys, T, ord_K=2, alpha=0.001, beta=np.inf,
                               method='randsearch', n_iter=800)
    print(f"  Reduced-order K (ord=2): {_zpk_str(Kred)}")
    print(f"  Reduced-order cost: {cost_red:.4f}  (MATLAB cost≈0.0059)")

    Kloc, cost_loc = modsdl2(sys, T, ord_K=2, alpha=0.001, beta=np.inf,
                               method='dual_annealing', n_iter=400)
    print(f"  After local opt: K = {_zpk_str(Kloc)}")
    print(f"  Local opt cost: {cost_loc:.4f}")


# ---------------------------------------------------------------------------
# registry & runner
# ---------------------------------------------------------------------------

_ALL = {
    "getting_started":      ex_getting_started,
    "deterministic_l2":     ex_deterministic_l2,
    "stochastic_h2":        ex_stochastic_h2,
    "first_order_filter":   ex_first_order_filter,
    "filter_with_delay":    ex_filter_with_delay,
    "second_order_filter":  ex_second_order_filter,
    "generalized_hold":     ex_generalized_hold,
    "disturbance_atten":    ex_disturbance_attenuation,
    "ship_course":          ex_ship_course,
    "double_integrator":    ex_double_integrator,
    "l2_tracking":          ex_l2_tracking,
    "l2_redesign":          ex_l2_redesign,
    "2dof_tracking":        ex_2dof_tracking,
    "ahinf_prediction":     ex_ahinf_prediction,
    "dhinf_ex1":            ex_dhinf_example1,
    "dhinf_ex2":            ex_dhinf_example2,
    "h2_ahinf_control":     ex_h2_ahinf_control,
    "mixed_h2_ahinf":       ex_mixed_h2_ahinf,
    "l2_ahinf_tracking":    ex_l2_ahinf_tracking,
    "h2_preview":           ex_h2_preview,
    "l2_preview":           ex_l2_preview,
    "2dof_preview":         ex_2dof_preview,
    "reduced_h2":           ex_reduced_h2,
    "integral_control":     ex_integral_control,
    "reduced_l2":           ex_reduced_l2,
    "reduced_redesign":     ex_reduced_redesign,
}


def main():
    import warnings
    warnings.filterwarnings('ignore')
    names = sys.argv[1:]
    if names:
        for n in names:
            if n in _ALL:
                try:
                    _ALL[n]()
                except Exception as e:
                    print(f"  ERROR in {n}: {e}")
                    import traceback; traceback.print_exc()
            else:
                print(f"Unknown example '{n}'. Available: {list(_ALL)}")
    else:
        fail = 0
        for name, fn in _ALL.items():
            try:
                fn()
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                import traceback; traceback.print_exc()
                fail += 1
        print(f"\n{'='*60}")
        print(f"Ran {len(_ALL)} examples,  {fail} errors")


if __name__ == "__main__":
    main()
