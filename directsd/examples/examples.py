"""
DirectSD Python Toolbox – Examples
===================================
Ports of the original MATLAB demo scripts.
Run this file directly or copy snippets into a Jupyter notebook.

Examples:
  1. Polynomial arithmetic and spectral factorization
  2. Diophantine equation solver
  3. Double integrator: H2-optimal sampled-data design
  4. L2-optimal controller design
  5. H-infinity norm computation
  6. State-space H2 controller design
  7. Stability margins and characteristic polynomial
  8. Global optimization (Nelder-Mead, Simulated Annealing)
  9. Bilinear transformation (Tustin discretization)
"""

import numpy as np
import scipy.signal as sig

# ── Import DirectSD ─────────────────────────────────────────────────────────
from directsd import (
    Poln, s_var, z_var,
    deg, gcd, coprime, factor, sfactor,
    dioph, striplz,
    dtfm, ztrm,
    charpol, sdmargin, sdh2norm, sdhinorm, dinfnorm,
    sdh2, sdl2, ch2, dhinf,
    h2reg, hinfreg, sdh2reg, sdfast,
    neldermead, simanneal, sector, banana,
    toep, hank, linsys, lyap,
)
from directsd.polynomial.utils import bilintr, improper, sumzpk, tf2nd


def separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# 1. Polynomial arithmetic and spectral factorization
# ============================================================
separator("1. Polynomial arithmetic")

s = s_var()
p = Poln([1, 3, 2], 's')   # s² + 3s + 2 = (s+1)(s+2)
q = Poln([1, 1], 's')       # s + 1

print(f"p(s)  = {p}")
print(f"q(s)  = {q}")
print(f"p + q = {p + q}")
print(f"p * q = {p * q}")
print(f"p(0)  = {p(0):.4f}  (expected 2)")
print(f"p roots = {p.roots}")

# Spectral factorization: find f such that f(s)*f(-s) = R(s)
# R(s) = (s+1)(-s+1) = 1 - s² – a valid Hermitian polynomial
R = Poln([-1, 0, 1], 's')   # 1 - s²
fs, _ = sfactor(R)
print(f"\nSpectral factorization of R = {R}")
print(f"  fs = {fs}  (expected ≈ s+1)")

# GCD
a = Poln([1, 3, 2], 's')    # (s+1)(s+2)
b = Poln([1, 2], 's')       # (s+2)
g = gcd(a, b)
print(f"\ngcd({a}, {b}) = {g}  (expected ≈ s+2)")


# ============================================================
# 2. Diophantine equation
# ============================================================
separator("2. Diophantine equation  X·A + Y·B = C")

A = Poln([1, 2, 1], 's')  # (s+1)²
B = Poln([1, 0], 's')     # s
C = Poln([1, 1], 's')     # s+1

X, Y, err, cond = dioph(A, B, C)
print(f"A = {A},  B = {B},  C = {C}")
print(f"Solution: X = {X},  Y = {Y}")
print(f"Residual ||X·A + Y·B - C|| = {err:.2e}  (cond = {cond:.2f})")


# ============================================================
# 3. Double integrator: H2-optimal sampled-data design
# ============================================================
separator("3. Double integrator – H2-optimal sampled-data design")

# Plant: F(s) = 1/(s²)  (double integrator)
F = sig.lti([1], [1, 0, 0])
T = 0.1  # sampling period

# Design H2-optimal discrete controller
K_h2, err_h2 = sdh2(F, T)
print(f"Plant: F(s) = 1/s²,  T = {T}")
print(f"H2-optimal controller numerator:   {K_h2[0]}")
print(f"H2-optimal controller denominator: {K_h2[1]}")
print(f"Optimal H2 cost: {err_h2:.6f}")

# Evaluate the H2-norm of closed loop
h2n = sdh2norm(F, K_h2, T)
print(f"H2-norm (verification):  {h2n:.6f}")

# Closed-loop poles
delta = charpol(F, K_h2, T)
poles = np.roots(delta)
print(f"Closed-loop poles: {poles}")
margin, _ = sdmargin(F, K_h2, T)
print(f"Stability margin (1 - max|pole|): {margin:.4f}")


# ============================================================
# 4. L2-optimal controller
# ============================================================
separator("4. L2-optimal controller design")

# Plant: F(s) = 1/(5s² + s)
F2 = sig.lti([1], [5, 1, 0])
T2 = 0.2

K_l2, err_l2 = sdl2(F2, T2)
print(f"Plant: 1/(5s²+s),  T = {T2}")
print(f"L2-optimal controller: {K_l2[0]} / {K_l2[1]}")
print(f"Optimal L2 cost: {err_l2:.6f}")

# Compare with H2
K_h2b, err_h2b = sdh2(F2, T2)
print(f"H2 cost (for comparison): {err_h2b:.6f}")


# ============================================================
# 5. H-infinity norm
# ============================================================
separator("5. H-infinity norm of discrete system")

# Discrete lowpass filter: G(z) = 0.1 / (z - 0.9)
num_d = np.array([0.1])
den_d = np.array([1.0, -0.9])
sys_d = sig.dlti(num_d, den_d, dt=0.1)

gamma, w_max = dinfnorm(sys_d)
print(f"G(z) = 0.1 / (z - 0.9)")
print(f"H∞ norm: {gamma:.4f}  at ω = {w_max:.3f} rad/sample")

# H-infinity of closed loop
K_test = ([1.0], [1.0])
gamma_cl, w_cl = sdhinorm(sig.lti([1], [1, 1]), K_test, T=0.1)
print(f"\nClosed-loop H∞ norm: {gamma_cl:.4f}")


# ============================================================
# 6. State-space H2 controller design
# ============================================================
separator("6. State-space H2 controller (h2reg)")

# Generalized plant (2×2 state-space)
A = np.array([[-1.0, 0.5],
              [-0.5, -2.0]])
B = np.array([[1.0, 0.0],
              [0.0, 1.0]])  # [B1 | B2]
C = np.array([[1.0, 0.0],
              [0.0, 1.0]])  # [C1; C2]
D = np.array([[0.0, 1.0],
              [1.0, 0.0]])  # [0 D12; D21 0]

plant_ss = sig.StateSpace(A, B, C, D)
K_ss, h2n_ss = h2reg(plant_ss, n_meas=1, n_ctrl=1)

print(f"Plant: 2×2 state-space system")
print(f"H2-optimal controller A_k:\n{K_ss.A}")
print(f"H2-optimal controller B_k:\n{K_ss.B}")
print(f"H2-optimal controller C_k:\n{K_ss.C}")
print(f"H2-norm of closed loop: {h2n_ss:.4f}")

# Sampled-data H2 via lifting
K_sd, h2n_sd = sdh2reg(plant_ss, T=0.1)
print(f"\nSampled-data H2 (lifting, T=0.1): H2-norm = {h2n_sd:.4f}")


# ============================================================
# 7. Fast discretization (sdfast)
# ============================================================
separator("7. Fast (exact ZOH) discretization")

plant_ct = sig.StateSpace(
    np.array([[-1.0, 1.0], [0.0, -2.0]]),
    np.array([[0.0], [1.0]]),
    np.array([[1.0, 0.0]]),
    np.array([[0.0]])
)
plant_dt = sdfast(plant_ct, T=0.1)
print(f"Continuous A:\n{plant_ct.A}")
print(f"Discrete A (T=0.1, exact ZOH):\n{np.round(plant_dt.A, 6)}")


# ============================================================
# 8. Global optimization
# ============================================================
separator("8. Global optimization")

# Nelder-Mead: Rosenbrock's banana function
print("Nelder-Mead on banana function (min at [1,1]):")
x_nm, f_nm, n_nm = neldermead(banana, [-1.2, 1.0], tol=1e-8, max_iter=5000)
print(f"  x* = [{x_nm[0]:.6f}, {x_nm[1]:.6f}]")
print(f"  f* = {f_nm:.2e}  (evals: {n_nm})")

# Simulated Annealing on a multimodal function
print("\nSimulated Annealing on f(x) = x² - 4x + sin(5x) + 4:")
f_sa = lambda x: x[0]**2 - 4*x[0] + np.sin(5*x[0]) + 4
np.random.seed(0)
x_sa, f_sa_val, n_sa, _ = simanneal(
    f_sa, [3.0],
    options={'maxFunEvals': 2000, 'startTemp': 5.0,
             'tempDecRate': 0.97, 'display': 'off'}
)
print(f"  x* = {x_sa[0]:.4f},  f* = {f_sa_val:.6f}  (evals: {n_sa})")

# Stability sector analysis
print("\nStability sector analysis:")
cl_poles = np.array([-0.5 + 1.5j, -0.5 - 1.5j, -2.0])
alpha, beta = sector(cl_poles)
print(f"  Poles: {cl_poles}")
print(f"  Degree of stability α = {alpha:.3f}")
print(f"  Degree of oscillation β = {beta:.3f}")


# ============================================================
# 9. Bilinear transformation (Tustin)
# ============================================================
separator("9. Bilinear transformation (Tustin discretization)")

# Continuous PID-like controller: K(s) = (s² + 3s + 2) / (s² + 0.1s)
K_ct = sig.lti([1, 3, 2], [1, 0.1, 0])
print(f"Continuous controller: {K_ct.num} / {K_ct.den}")

T_disc = 0.05
K_tustin_num, K_tustin_den = bilintr(K_ct, 'tustin', T_disc)
print(f"Tustin (T={T_disc}): num = {np.round(K_tustin_num, 4)}")
print(f"                    den = {np.round(K_tustin_den, 4)}")

# Verify: inverse Tustin should give back original (approximately)
K_back_num, K_back_den = bilintr((K_tustin_num, K_tustin_den), 'z2s', T_disc)
print(f"Inverse Tustin (back to CT):")
print(f"  num ≈ {np.round(K_back_num / K_back_num[-1] * K_ct.num[-1], 3)}")
print(f"  den ≈ {np.round(K_back_den / K_back_den[-1] * K_ct.den[-1], 3)}")


# ============================================================
# 10. DTF transform (discrete Laplace with ZOH)
# ============================================================
separator("10. Modified Z-transform (dtfm)")

# Discretize 1/(s+1) with ZOH, T=0.1
G_ct = sig.lti([1], [1, 1])
G_num_d, G_den_d = dtfm(G_ct, T=0.1)
print(f"G(s) = 1/(s+1)  →  G_d(z) num = {np.round(G_num_d, 6)}")
print(f"                             den = {np.round(G_den_d, 6)}")

# Verify with scipy
G_d_scipy = G_ct.to_discrete(0.1, method='zoh')
print(f"scipy ZOH:              num = {np.round(G_d_scipy.num, 6)}")
print(f"                        den = {np.round(G_d_scipy.den, 6)}")


# ============================================================
# 11. Linear algebra utilities
# ============================================================
separator("11. Linear algebra utilities")

# Toeplitz matrix from polynomial
a = np.array([1, 2, 3])
T_mat = toep(a, 4, 3)
print(f"Toeplitz matrix from [1,2,3]:\n{T_mat}")

# Lyapunov equation: A*P + P*A' + Q = 0
A_lyap = np.array([[-2.0, 1.0], [0.0, -3.0]])
Q_lyap = np.eye(2)
P_lyap = lyap(A_lyap, Q_lyap)
residual = A_lyap @ P_lyap + P_lyap @ A_lyap.T + Q_lyap
print(f"\nLyapunov residual ||A·P + P·A' + Q|| = {np.linalg.norm(residual):.2e}")

# Linear system solver (SVD)
A_mat = np.array([[2.0, 1.0], [1.0, 3.0], [0.5, 0.5]])
b_vec = np.array([[4.0], [5.0], [1.5]])
x_sol = linsys(A_mat, b_vec, method='svd')
print(f"\nLeast-squares solution to Ax=b: x = {x_sol.ravel()}")
print(f"Residual ||Ax - b|| = {np.linalg.norm(A_mat @ x_sol - b_vec):.2e}")


print("\n" + "=" * 60)
print("  All examples completed successfully!")
print("=" * 60 + "\n")
