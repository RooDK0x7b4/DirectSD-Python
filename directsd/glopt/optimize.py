"""
Global optimization routines for DirectSD.

Ports of: simanneal, neldermead, randsearch, optglob, sector
"""

import numpy as np


# ---------------------------------------------------------------------------
# Nelder-Mead simplex
# ---------------------------------------------------------------------------

def neldermead(func, x0, tol=1e-5, max_iter=1000, max_feval=5000):
    """
    Simplex minimization by Nelder and Mead.

    Parameters
    ----------
    func : callable
        Objective function f(x) -> float.
    x0 : array-like
        Initial guess.
    tol : float
        Convergence tolerance on x.
    max_iter : int
    max_feval : int

    Returns
    -------
    x_best : np.ndarray
    f_best : float
    n_evals : int
    """
    x0 = np.array(x0, dtype=float).ravel()
    n = len(x0)

    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5

    # Build initial simplex
    X = np.zeros((n, n + 1))
    X[:, 0] = x0
    y = np.zeros(n + 1)
    y[0] = func(x0)
    n_evals = 1

    for j in range(n):
        z = x0.copy()
        z[j] = z[j] * 1.05 if z[j] != 0 else 0.00025
        X[:, j + 1] = z
        y[j + 1] = func(z)
        n_evals += 1

    for _ in range(max_iter):
        if n_evals >= max_feval:
            break

        # Sort
        order = np.argsort(y)
        y = y[order]
        X = X[:, order]

        # Convergence check
        if np.max(np.abs(X[:, 1:] - X[:, :1])) < tol:
            break

        # Centroid of best n points
        xbar = X[:, :n].mean(axis=1)
        xr = (1 + rho) * xbar - rho * X[:, n]  # reflection
        fr = func(xr); n_evals += 1

        if fr < y[0]:
            # Expansion
            xe = (1 + rho * chi) * xbar - rho * chi * X[:, n]
            fe = func(xe); n_evals += 1
            if fe < fr:
                X[:, n] = xe; y[n] = fe
            else:
                X[:, n] = xr; y[n] = fr
        elif fr < y[n - 1]:
            X[:, n] = xr; y[n] = fr
        else:
            if fr < y[n]:
                # Outside contraction
                xc = (1 + psi * rho) * xbar - psi * rho * X[:, n]
            else:
                # Inside contraction
                xc = (1 - psi) * xbar + psi * X[:, n]
            fc = func(xc); n_evals += 1
            if fc < min(fr, y[n]):
                X[:, n] = xc; y[n] = fc
            else:
                # Shrink
                for j in range(1, n + 1):
                    X[:, j] = X[:, 0] + sigma * (X[:, j] - X[:, 0])
                    y[j] = func(X[:, j]); n_evals += 1

    order = np.argsort(y)
    return X[:, order[0]], y[order[0]], n_evals


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

def simanneal(func, x0, options=None, constraint_func=None, proj_func=None):
    """
    Direct search using simulated annealing.

    Parameters
    ----------
    func : callable
        Cost function f(x) -> float.
    x0 : array-like
        Initial point.
    options : dict, optional
        Fields: display, tol, maxFunEvals, iniStep, multiStep,
                maxFail, decStepBy, startTemp, tempDecRate, adaptRate.
    constraint_func : callable, optional
        g(x) -> array; values should be <= 0 for feasibility.
    proj_func : callable, optional
        proj(x) -> x_projected.

    Returns
    -------
    x_best : np.ndarray
    f_best : float
    n_evals : int
    x_trace : list of np.ndarray
    """
    defaults = dict(
        display='off',
        tol=1e-4,
        maxFunEvals=1000,
        iniStep=0.1,
        multiStep=20,
        maxFail=100,
        decStepBy=0.5,
        decFailRate=1.1,
        adaptRate=0.1,
        startTemp=100.0,
        tempDecRate=0.95,
        dispIter=100,
    )
    if options:
        defaults.update(options)
    opt = defaults

    x0 = np.array(x0, dtype=float).ravel()
    n = len(x0)

    x_cur = x0.copy()
    if proj_func:
        x_cur = proj_func(x_cur)

    f_cur = func(x_cur)
    x_best = x_cur.copy()
    f_best = f_cur
    n_evals = 1

    step = opt['iniStep'] * (np.max(np.abs(x0)) + 1.0)  # scale step to problem
    T = opt['startTemp']
    # Auto-scale temperature if too high relative to objective
    if abs(f_cur) > 0:
        T = min(T, abs(f_cur) * 10)
    fail = 0
    x_trace = [x_cur.copy()]

    for iteration in range(opt['maxFunEvals'] - 1):
        for _ in range(opt['multiStep']):
            # Propose new point
            x_new = x_cur + step * (2 * np.random.rand(n) - 1)
            if proj_func:
                x_new = proj_func(x_new)

            # Check constraints
            if constraint_func and np.any(constraint_func(x_new) > 0):
                fail += 1
                continue

            f_new = func(x_new)
            n_evals += 1

            dE = f_new - f_cur
            if dE < 0 or (T > 1e-10 and np.random.rand() < np.exp(-dE / T)):
                x_cur = x_new
                f_cur = f_new
                fail = 0
                if f_cur < f_best:
                    x_best = x_cur.copy()
                    f_best = f_cur
                    x_trace.append(x_best.copy())
            else:
                fail += 1

            if fail >= opt['maxFail']:
                step *= opt['decStepBy']
                opt['maxFail'] = int(opt['maxFail'] * opt['decFailRate'])
                fail = 0

        T *= opt['tempDecRate']

        if opt['display'] == 'on' and (iteration + 1) % opt['dispIter'] == 0:
            print(f"  Iter {iteration + 1}: f_best = {f_best:.6g}, step = {step:.4g}, T = {T:.4g}")

        if n_evals >= opt['maxFunEvals']:
            break
        if step < opt['tol']:
            break

    if opt['display'] in ('on', 'final'):
        print(f"SA finished: f_best = {f_best:.6g}, evals = {n_evals}")

    return x_best, f_best, n_evals, x_trace


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def randsearch(func, x0, bounds=None, n_iter=500, seed=None):
    """
    Simple random search optimization.

    Parameters
    ----------
    func : callable
    x0 : array-like
        Initial guess.
    bounds : list of (min, max) tuples, optional
        Search bounds per dimension.
    n_iter : int
    seed : int, optional

    Returns
    -------
    x_best, f_best, n_evals
    """
    rng = np.random.default_rng(seed)
    x0 = np.array(x0, dtype=float).ravel()
    n = len(x0)

    x_best = x0.copy()
    f_best = func(x0)
    n_evals = 1

    for _ in range(n_iter - 1):
        if bounds:
            x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        else:
            x = x_best + rng.normal(0, 0.1, n)

        f = func(x)
        n_evals += 1
        if f < f_best:
            f_best = f
            x_best = x.copy()

    return x_best, f_best, n_evals


# ---------------------------------------------------------------------------
# Controlled Random Search (CRS)
# ---------------------------------------------------------------------------

def crandsearch(func, bounds, n_pop=None, max_feval=10000, tol=1e-6, seed=None):
    """
    Controlled Random Search (CRS) global optimization (Price, 1983).

    Parameters
    ----------
    func : callable
        Objective function f(x) -> float.
    bounds : list of (lo, hi) tuples
        Search domain bounds.
    n_pop : int, optional
        Population size (default: 10*n).
    max_feval : int
    tol : float
    seed : int, optional

    Returns
    -------
    x_best, f_best, n_evals
    """
    rng = np.random.default_rng(seed)
    n = len(bounds)
    if n_pop is None:
        n_pop = 10 * n

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Initialize population
    pop = rng.uniform(0, 1, (n_pop, n)) * (hi - lo) + lo
    vals = np.array([func(p) for p in pop])
    n_evals = n_pop

    for _ in range(max_feval - n_pop):
        # Select worst point
        worst_idx = np.argmax(vals)

        # Select n random distinct points (not worst)
        candidates = [i for i in range(n_pop) if i != worst_idx]
        sel = rng.choice(candidates, n, replace=False)

        # Form trial point: centroid of selected minus worst
        centroid = pop[sel].mean(axis=0)
        x_new = 2 * centroid - pop[worst_idx]
        x_new = np.clip(x_new, lo, hi)

        f_new = func(x_new)
        n_evals += 1

        if f_new < vals[worst_idx]:
            pop[worst_idx] = x_new
            vals[worst_idx] = f_new

        best_idx = np.argmin(vals)
        if np.max(np.abs(pop - pop[best_idx])) < tol:
            break

    best_idx = np.argmin(vals)
    return pop[best_idx], vals[best_idx], n_evals


# ---------------------------------------------------------------------------
# Sector (stability degree / oscillation)
# ---------------------------------------------------------------------------

def sector(poles, T=None):
    """
    Find degree of stability and oscillation for closed-loop poles.

    Parameters
    ----------
    poles : array-like
        Closed-loop poles (continuous or discrete).
    T : float, optional
        Sampling period (if None, treats poles as continuous-time).

    Returns
    -------
    alpha : float
        Degree of stability (distance from imaginary axis / unit circle).
    beta : float
        Degree of oscillation (max |imag part| / |real part|).
    """
    poles = np.atleast_1d(np.array(poles))

    if T is not None and T > 0:
        # Convert discrete poles to continuous
        poles = np.log(poles) / T

    real_parts = np.real(poles)
    alpha = -np.max(real_parts)  # positive = stable

    with np.errstate(divide='ignore', invalid='ignore'):
        osc = np.where(np.abs(real_parts) > 1e-10,
                       np.abs(np.imag(poles)) / np.abs(real_parts),
                       np.inf)
    beta = np.max(osc[np.isfinite(osc)]) if np.any(np.isfinite(osc)) else 0.0

    return float(alpha), float(beta)


# ---------------------------------------------------------------------------
# Utility: banana (Rosenbrock test function)
# ---------------------------------------------------------------------------

def banana(x):
    """
    Rosenbrock's banana function.
    f(x, y) = 100*(y - x^2)^2 + (1 - x)^2
    """
    x = np.atleast_1d(x)
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


# ---------------------------------------------------------------------------
# dual_annealing – scipy-based replacement for simanneal (faster, more robust)
# ---------------------------------------------------------------------------

def dual_annealing(func, bounds, seed=None, maxiter=1000, **kwargs):
    """
    Global optimization using scipy.optimize.dual_annealing.

    A hybrid approach combining classical simulated annealing with a fast
    local search.  Substantially faster and more reliable than the custom
    simanneal() for reduced-order controller design (e.g. modsdh2).

    Parameters
    ----------
    func : callable
        Objective function f(x) -> float.
    bounds : list of (lo, hi) tuples
        Search bounds for each dimension.
    seed : int, optional
        Random seed for reproducibility.
    maxiter : int
        Maximum number of function evaluations.
    **kwargs : dict
        Passed directly to scipy.optimize.dual_annealing.

    Returns
    -------
    x_best : np.ndarray
    f_best : float
    result : scipy.optimize.OptimizeResult
        Full scipy result object (includes nfev, nit, message, etc.).

    Example
    -------
    >>> from directsd.glopt import dual_annealing
    >>> x, f, res = dual_annealing(banana, [(-3,3),(-3,3)], seed=42)
    >>> print(f"minimum at x={x}, f={f:.2e}")
    """
    from scipy.optimize import dual_annealing as _scipy_da
    result = _scipy_da(func, bounds, seed=seed, maxfun=maxiter, **kwargs)
    return result.x, float(result.fun), result
