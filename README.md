# DirectSD — Python Package

[![PyPI version](https://img.shields.io/pypi/v/DirectSD-Python.svg)](https://pypi.org/project/DirectSD-Python/)
[![Python versions](https://img.shields.io/pypi/pyversions/DirectSD-Python.svg)](https://pypi.org/project/DirectSD-Python/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

[github.com/RooDK0x7b4/DirectSD-Python](https://github.com/RooDK0x7b4/DirectSD-Python)

Python port of the **DirectSD MATLAB Toolbox v3.0** by Konstantin Polyakov (1995–2006).

DirectSD is a toolbox for analysis and design of **sampled-data (hybrid) control systems** using:
- Polynomial (Diophantine-equation) methods
- Lifting / FR-operator approach (exact inter-sample modelling)
- State-space (LQR / Kalman / Riccati ARE) methods
- H2 and H∞ optimal control
- Global optimization (Strongin information algorithm, SA simplex, Hilbert-curve search)
- **L1 / L∞ convex synthesis** via Youla parameterisation (requires `cvxpy`)

---

## Installation

```bash
pip install numpy scipy          # runtime dependencies
pip install -e .                 # development install from source
```

Or with optional extras:

```bash
pip install -e ".[dev]"          # adds pytest + matplotlib
pip install -e ".[convex]"       # adds cvxpy for L1/L∞ convex synthesis
pip install -e ".[control]"      # adds python-control interoperability
pip install -e ".[all]"          # installs all optional dependencies
```

---

## Package Structure

```
directsd/
├── __init__.py                  # flat public API — import everything from here
│
├── polynomial/                  # Poln class, operations, spectral factorisation, transforms
│   ├── poln.py                  # Poln — the @poln OOP class equivalent
│   ├── operations.py            # compat, deg, gcd, coprime, triple, factor,
│   │                            #   striplz, recip, vec, dioph, dioph2
│   ├── spectral.py              # sfactor, sfactfft
│   ├── transforms.py            # dtfm (ZOH discrete Laplace), ztrm (modified Z)
│   └── utils.py                 # bilintr, improper, diophsys, tf2nd,
│                                #   separtf, sumzpk, bilinss
│
├── zpk/                         # Zpk class (root-list zero-pole-gain representation)
│   └── zpk.py
│
├── linalg/                      # Linear algebra utilities
│   ├── matrices.py              # toep, hank, givens, house, lyap, dlyap
│   └── linsys.py                # linsys (QR / SVD with iterative refinement)
│
├── analysis/                    # Sampled-data system analysis
│   ├── norms.py                 # sdh2norm, sdhinorm, dinfnorm, dahinorm,
│   │                            #   h2norm_ct, hinfnorm_ct
│   │                            #   ── all via Lyapunov / Hamiltonian ARE,
│   │                            #      no frequency gridding
│   ├── charpol.py               # charpol, sdmargin
│   └── errors.py                # quaderr, sdl2err, sd2doferr
│
├── design/                      # Controller design
│   ├── lifting.py               # lift_h2, lift_l2, compute_gamma
│   │                            #   ── Van Loan exact inter-sample lifting
│   │                            #      (FR-operator / Chen-Francis method)
│   ├── polynomial.py            # ch2, sdh2, sdl2, dhinf, sd2dof, polquad
│   └── convex.py                # sdl1_reg, sd_mixed_h2_l1, sd_constrained,
│                                #   sdl1norm, youla_basis  [requires cvxpy]
│
├── sspace/                      # State-space design
│   ├── plant.py                 # GeneralizedPlant class (augmented plant representation)
│   ├── design.py                # h2reg, hinfreg, sdh2reg, sdhinfreg,
│   │                            #   sdfast, separss
│   └── hinfsd.py                # Native discrete Hinf synthesis: sdhimod (all 5
│                                #   discretization formulas) + hinfone/hinfone1
│                                #   (both gamma=1 synthesis methods) -- sdhinfreg's
│                                #   primary path, exact rather than the bilinear
│                                #   DT->CT->DT approximation
│
├── tf/                          # Block-diagram arithmetic
│   ├── interconnect.py          # mul, neg, add, feedback, nd, to_lti
│   └── __init__.py              # re-exports all interconnect functions
│
├── glopt/                       # Global optimization
│   ├── optimize.py              # neldermead, simanneal, randsearch,
│   │                            #   crandsearch, dual_annealing, sector, banana
│   └── advanced.py              # Full dsdglopt port: updateopt, uniproj, u2range,
│                                #   randbeta, randgamma, sa_testfun,
│                                #   val2bin, bin2val, coord2hilb, hilb2coord,
│                                #   r2range, r1range, admproj, par2cp, cp2par,
│                                #   guesspoles, k2ksi, go_par2k,
│                                #   f_sdh2p, f_sdl2p, go_sdh2p, go_sdl2p,
│                                #   sasimplex, arandsearch,
│                                #   infglob, infglobc, optglob, optglobc
│
├── examples/                    # demos.py (25 MATLAB demo ports), help_examples.py
│                                #   (26 dsd_help.chm worked examples), examples.py
│                                #   (11 core-utility examples), _common.py (shared helpers)
│
└── tests/
    └── test_directsd.py         # 254 pytest tests (235 core + 19 convex,
                                 #   convex skipped automatically without cvxpy)
```

---

## Quick Start

### Polynomial arithmetic

```python
from directsd import Poln, s_var, gcd, sfactor, dioph

s = s_var()
p = Poln([1, 3, 2], 's')   # s² + 3s + 2
q = Poln([1, 1], 's')      # s + 1

print(p * q)                # s³ + 4s² + 5s + 2
print(p.roots)              # [-2. -1.]
print(gcd(p, q))            # s + 1

# Spectral factorization: R(s) = N(s)·N(-s) → find N
R  = Poln([-1, 0, 1], 's') # 1 − s²  (Hermitian)
fs, _ = sfactor(R)
print(fs)                   # s + 1

# Diophantine equation: X·A + Y·B = C
x, y, err, cond = dioph(Poln([1,1],'s'), Poln([1,0],'s'), Poln([1],'s'))
```

### Lifting → H2 design (recommended workflow)

```python
import scipy.signal as sig
from directsd import lift_h2, h2reg, compute_gamma
import numpy as np

# Continuous-time generalized plant (standard 2×2 form)
plant = sig.StateSpace(
    np.array([[-1., 0.5], [0., -2.]]),
    np.array([[1., 0.], [0., 1.]]),
    np.array([[1., 0.], [0., 1.]]),
    np.array([[0., 1.], [1., 0.]])
)

T = 0.1   # sampling period

# Step 1 — lift to exact discrete equivalent (Van Loan / FR-operator)
P_lifted, gamma, _ = lift_h2(plant, T)
print(f"Lifted plant: {P_lifted.A.shape}, γ = {gamma:.4f}")

# Step 2 — H2-optimal controller on lifted plant
K, h2n = h2reg(P_lifted, n_meas=1, n_ctrl=1)

# Step 3 — exact total H2 cost
total_cost = np.sqrt(h2n**2 + gamma)
print(f"Optimal H2 cost: {total_cost:.4f}")
```

### Norms (algebraic, no frequency gridding)

```python
from directsd import sdh2norm, sdhinorm, h2norm_ct, hinfnorm_ct, dinfnorm
import scipy.signal as sig

plant = sig.lti([1], [1, 1])

# Continuous-time norms
print(h2norm_ct(plant))           # 0.7071  (= 1/√2)
gamma, w = hinfnorm_ct(plant)     # 1.0 at ω=0
print(gamma, w)

# Sampled-data norms
K = ([0.5], [1.0])
print(sdh2norm(plant, K, T=0.1))
gamma, w = sdhinorm(plant, K, T=0.1)
print(gamma, w)
```

### Stability analysis

```python
from directsd import charpol, sdmargin
import scipy.signal as sig

plant = sig.lti([1], [1, 1])
K     = ([1.0], [1.0])
T     = 0.1

delta  = charpol(plant, K, T)     # characteristic polynomial coefficients
margin, poles = sdmargin(plant, K, T)
print(f"Stability margin: {margin:.4f}")   # positive = stable
print(f"Closed-loop poles: {poles}")
```

### Global optimization

```python
from directsd import neldermead, dual_annealing, simanneal, sector, banana

# Nelder-Mead (deterministic)
x, f, n = neldermead(banana, [-1.2, 1.0], tol=1e-8)
print(f"Banana min: x={x}, f={f:.2e}")

# Dual annealing (scipy hybrid, faster and more robust than simanneal)
x, f, res = dual_annealing(banana, [(-3,3),(-3,3)], seed=42)
print(f"Dual SA min: x={x}, f={f:.2e}, evals={res.nfev}")

# Stability sector
alpha, beta = sector([-1+2j, -1-2j, -3+0j])
print(f"Degree of stability α={alpha}, oscillation β={beta}")
```

### Advanced global optimization (dsdglopt full port)

```python
from directsd import (
    infglob, optglob, sasimplex, arandsearch,
    par2cp, cp2par, admproj, guesspoles,
    coord2hilb, hilb2coord, sa_testfun, randbeta, randgamma,
)

# Strongin information algorithm — 1D Lipschitz minimisation
f = lambda x: (x - 0.3)**2 + 0.1*np.sin(20*x)
x_best, z_best, n_iter, trace = infglob(f, {'maxIter': 300, 'r': 2.5})

# Multi-run with zooming
x_best, z_best, coef, loops, trace = optglob(f, {'maxLoop': 10, 'maxIter': 100})

# SA + Nelder-Mead hybrid
x, y, nevals = sasimplex(sa_testfun, [0.5, 0.5], {'startTemp': 5.0, 'maxFunEvals': 5000})

# Hilbert space-filling curve (N-D ↔ 1D)
coord = np.array([0.3, 0.7, 0.5])
x1d   = coord2hilb(coord, precision=8)   # 3D → scalar in [0,1]
back  = hilb2coord(x1d, dimensions=3, precision=8)

# Stability-sector polynomial parameterisation
Delta, DeltaZ, poles = par2cp([0.4, 0.6, 0.5], alpha=0.2, beta=np.inf)
rho = cp2par(DeltaZ, alpha=0.2, beta=np.inf)  # inverse
```

### Convex synthesis — L1 and mixed H2/L1 (requires `pip install cvxpy`)

The Youla (Q) parameterisation turns all stabilising controllers into an
affine family, making L1 and mixed-norm problems convex.

```python
import numpy as np
import scipy.signal as sig
from directsd import lift_h2, sdl1_reg, sd_mixed_h2_l1, sd_constrained, sdl1norm

# Step 1 — continuous-time generalised plant (standard 2×2 form)
ct_plant = sig.StateSpace(
    np.array([[-1.]]),
    np.array([[1., 1.]]),
    np.array([[1.], [1.]]),
    np.array([[0., 1.], [1., 0.]])
)
T = 0.1

# Step 2 — lift to exact discrete equivalent
P_lifted, _, _ = lift_h2(ct_plant, T)

# L1-optimal controller (minimises peak output amplitude)
K_l1, l1_norm, Q_fir, info = sdl1_reg(P_lifted, N_fir=20)
print(f"L1-norm (peak-to-peak gain): {l1_norm:.4f}  [{info['status']}]")

# Mixed H2/L1: minimise H2 subject to L1 ≤ bound
K_mix, cost, _, _ = sd_mixed_h2_l1(P_lifted, N_fir=20, l1_bound=5.0)
print(f"H2 cost: {cost['h2']:.4f},  L1: {cost['l1']:.4f}")

# Hard envelope constraint
lo = -1.5 * np.ones((15, 1))
hi =  1.5 * np.ones((15, 1))
K_con, achieved, _, _ = sd_constrained(
    P_lifted, N_fir=20, objective='h2', envelope=(lo, hi)
)
print(f"Constrained H2: {achieved['h2']:.4f},  linf: {achieved['linf']:.4f}")

# L1-norm analysis of any existing closed loop (no cvxpy needed)
l1, h = sdl1norm(sig.lti([1], [1, 2, 1]), ([0.5], [1.0]), T=0.1)
print(f"Closed-loop L1-norm: {l1:.4f}")
```

---

## MATLAB → Python mapping

| MATLAB | Python |
|--------|--------|
| `poln(a,'s')` | `Poln(a, 's')` |
| `s`, `z`, `p`, `q`, `d` | `s_var()`, `z_var()`, … |
| `gcd(a,b)` | `gcd(a, b)` |
| `coprime(a,b)` | `coprime(a, b)` |
| `factor(p,'s')` | `factor(p, 's')` |
| `sfactor(p)` | `sfactor(p)` |
| `sfactfft(p)` | `sfactfft(p)` |
| `dioph(a,b,c)` | `dioph(a, b, c)` |
| `dtfm(G,T,0)` | `dtfm(G, T)` |
| `ztrm(G,T,mu)` | `ztrm(G, T, mu)` |
| `bilintr(F,'tustin',T)` | `bilintr(F, 'tustin', T)` |
| `improper(R)` | `improper(R)` |
| `toep(a,r,c)` | `toep(a, r, c)` |
| `linsys(A,B,'svd')` | `linsys(A, B, method='svd')` |
| `sdgh2mod(sys,T)` | `lift_h2(plant_ss, T)` |
| `sdh2simple(sys,T)` | `lift_l2(plant_ss, T)` |
| `charpol(sys,K)` | `charpol(plant, K, T)` |
| `sdmargin(sys,K)` | `sdmargin(plant, K, T)` |
| `sdh2norm(sys,K)` | `sdh2norm(plant, K, T)` |
| `sdhinorm(sys,K)` | `sdhinorm(plant, K, T)` |
| `dinfnorm(sys)` | `dinfnorm(sys)` |
| `quaderr(A,B,E,M)` | `quaderr(plant_cl, T)` |
| `sdl2err(sys,K)` | `sdl2err(plant, K, T)` |
| `ch2(sys)` | `ch2(plant)` |
| `sdh2(sys,T)` | `sdh2(plant, T)` |
| `sdl2(sys,T)` | `sdl2(plant, T)` |
| `dhinf(plant_d)` | `dhinf(plant_d)` |
| `sd2dof(sys,K,T)` | `sd2dof(plant, K_fb, T)` |
| `h2reg(sys,1,1)` | `h2reg(sys, n_meas=1, n_ctrl=1)` |
| `hinfreg(sys,1,1)` | `hinfreg(sys, n_meas=1, n_ctrl=1)` |
| `sdh2reg(sys,T)` | `sdh2reg(plant, T)` |
| `sdhinfreg(sys,T)` | `sdhinfreg(plant, T)` |
| `sdfast(sys,T)` | `sdfast(plant, T)` |
| `neldermead(f,x0)` | `neldermead(f, x0)` |
| `simanneal(f,x0,opt)` | `simanneal(f, x0, options=opt)` |
| `sector(poles)` | `sector(poles)` |
| `sasimplex(f,x0,opt)` | `sasimplex(f, x0, options=opt)` |
| `arandsearch(f,x,opt)` | `arandsearch(f, x0, options=opt)` |
| `infglob(f,opt)` | `infglob(f, options=opt)` |
| `infglobc(f,opt)` | `infglobc(f, options=opt)` |
| `optglob(f,opt)` | `optglob(f, options=opt)` |
| `optglobc(f,opt)` | `optglobc(f, options=opt)` |
| `par2cp(rho,alpha,beta)` | `par2cp(rho, alpha, beta)` |
| `cp2par(DeltaZ,alpha,beta)` | `cp2par(DeltaZ, alpha, beta)` |
| `admproj(p,alpha,beta,dom,T)` | `admproj(p, alpha, beta, dom, T)` |
| `guesspoles(poles,n)` | `guesspoles(poles, n_poles)` |
| `r1range(r2,Ea,beta,sh)` | `r1range(r2, Ea, beta, shifted)` |
| `r2range(Ea,Eb,sh)` | `r2range(Ea, Eb, shifted)` |
| `coord2hilb(c,p)` | `coord2hilb(coord, precision)` |
| `hilb2coord(x,d,p)` | `hilb2coord(x, dimensions, precision)` |
| `val2bin(x,nf)` | `val2bin(x, nfrac)` |
| `bin2val(bi,bf)` | `bin2val(bint, bfrac)` |
| `k2ksi(sys,K,dK0)` | `k2ksi(plant, K, dK0, T)` |
| `go_par2k(coef)` | `go_par2k(coef, ctx)` |
| `f_sdh2p(coef)` | `f_sdh2p(coef, ctx)` |
| `f_sdl2p(coef)` | `f_sdl2p(coef, ctx)` |
| `go_sdh2p(x)` | `go_sdh2p(x, ctx)` |
| `go_sdl2p(x)` | `go_sdl2p(x, ctx)` |
| `randbeta(a,b,r,c)` | `randbeta(a, b, rows, cols)` |
| `randgamma(a,r,c)` | `randgamma(a, rows, cols)` |
| `sa_testfun(x)` | `sa_testfun(x)` |
| `updateopt(opt,new)` | `updateopt(options, opt)` |
| `uniproj(x)` | `uniproj(x)` |
| `u2range(u,lo,hi)` | `u2range(u, lo, hi)` |
| `sdnorm(sys,K,'l1')` | `sdl1norm(plant, K, T)` — L1/peak-to-peak norm analysis |
| *(no MATLAB equivalent)* | `sdl1_reg(P_lifted, N_fir)` — L1-optimal synthesis |
| *(no MATLAB equivalent)* | `sd_mixed_h2_l1(P_lifted, N_fir, l1_bound=…)` — mixed H2/L1 |
| *(no MATLAB equivalent)* | `sd_constrained(P_lifted, N_fir, envelope=(lo,hi))` — hard constraints |
| *(no MATLAB equivalent)* | `youla_basis(P_lifted, K0, N_fir)` — raw Youla maps for custom problems |

---

## Key design decisions

**Lifting is first-class.** `design/lifting.py` implements the exact Van Loan
matrix-exponential method (Hagiwara-Araki 1995, Chen-Francis 1995).
Unlike simple ZOH discretization, the lifted plant accounts for inter-sample
signal energy — the `gamma` term captures what a continuous-time H2 design
recovers that a purely discrete design misses.

**No frequency gridding for norms.** All norm functions in `analysis/norms.py`
use Lyapunov equations or Hamiltonian bisection.  This eliminates
`BadCoefficients` warnings from ill-conditioned transfer functions and gives
machine-precision results even for systems with integrators or sharp resonances.

**SVD-based Diophantine solver.** `dioph()` uses `scipy.linalg.lstsq` instead
of a custom QR solver, making it robust to rank-deficient Sylvester matrices
that arise when the plant has repeated or near-common poles.

**Convex synthesis goes beyond the MATLAB toolbox.** `design/convex.py`
implements L1 (peak-to-peak) optimal control and mixed H2/L1 design via the
Youla Q-parameterisation — a capability not present in the original MATLAB
DirectSD toolbox. All stabilising controllers are parameterised as
`K = K0 + Δ(Q)` where Q is a stable FIR sequence, reducing the synthesis
to a Linear Programme solved by CVXPY. This is an optional dependency:
the rest of the package works without it.

---

## Notes

- Simulink `.mdl` files (`dsdmodel/`) have no Python equivalent; use `scipy.signal.lsim` for simulation.
- The `@lti`, `@tf`, `@zpk` MATLAB overrides are replaced throughout by `scipy.signal`.
- All functions accept `scipy.signal.lti`, `scipy.signal.dlti`, `(num, den)` tuples, and `scipy.signal.StateSpace` objects interchangeably.

---

## How to Cite

If you use this package in your research, please cite it — see
[`CITATION.cff`](CITATION.cff), or use GitHub's "Cite this repository" button
on the repo page.

## License

This Python port is licensed under the [BSD 3-Clause License](LICENSE) ©
2026 Roozbeh Dargahi. The original MATLAB toolbox is © 1995–2006 Konstantin
Polyakov; this license covers only the Python implementation in this
repository, not the original MATLAB source.
