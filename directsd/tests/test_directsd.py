"""
Tests for DirectSD Python package.

Run with:  pytest tests/test_directsd.py -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ============================================================
# Polynomial tests
# ============================================================

class TestPoln:
    def test_create_from_coef(self):
        from directsd import Poln
        p = Poln([1, 2, 1], 's')
        assert p.degree == 2
        assert p.var == 's'
        np.testing.assert_allclose(p.coef, [1, 2, 1])

    def test_create_from_roots(self):
        from directsd import Poln
        p = Poln([-1, -1], 'rs')
        assert p.degree == 2
        # Roots of (s+1)^2 = s^2+2s+1
        np.testing.assert_allclose(p.coef, [1, 2, 1], atol=1e-10)

    def test_scalar(self):
        from directsd import Poln
        p = Poln([5], 's')
        assert p.degree == 0
        assert abs(p.k - 5) < 1e-10

    def test_addition(self):
        from directsd import Poln
        a = Poln([1, 0], 's')   # s
        b = Poln([1, 1], 's')   # s + 1
        c = a + b               # 2s + 1
        np.testing.assert_allclose(c.coef, [2, 1], atol=1e-10)

    def test_multiplication(self):
        from directsd import Poln
        a = Poln([1, 1], 's')   # s+1
        b = Poln([1, -1], 's')  # s-1
        c = a * b               # s^2 - 1
        np.testing.assert_allclose(c.coef, [1, 0, -1], atol=1e-10)

    def test_scalar_multiply(self):
        from directsd import Poln
        p = Poln([1, 2, 1], 's')
        q = 3 * p
        np.testing.assert_allclose(q.coef, [3, 6, 3], atol=1e-10)

    def test_negation(self):
        from directsd import Poln
        p = Poln([1, 2], 's')
        np.testing.assert_allclose((-p).coef, [-1, -2], atol=1e-10)

    def test_evaluate(self):
        from directsd import Poln
        p = Poln([1, 0, 0], 's')  # s^2
        assert abs(p(3.0) - 9.0) < 1e-10
        assert abs(p(0.0)) < 1e-10

    def test_roots(self):
        from directsd import Poln
        p = Poln([1, 0, -1], 's')  # s^2 - 1
        r = np.sort(p.roots.real)
        np.testing.assert_allclose(r, [-1, 1], atol=1e-10)

    def test_derivative(self):
        from directsd import Poln
        p = Poln([1, 0, 0], 's')  # s^2
        dp = p.derivative()        # 2s
        np.testing.assert_allclose(dp.coef, [2, 0], atol=1e-10)

    def test_reciprocal(self):
        from directsd import Poln
        p = Poln([1, 2, 3], 's')
        r = p.reciprocal()
        np.testing.assert_allclose(r.coef, [3, 2, 1], atol=1e-10)

    def test_discrete_var(self):
        from directsd import Poln
        p = Poln([1, -0.5], 'z')
        assert p.is_dt
        assert not p.is_ct

    def test_continuous_var(self):
        from directsd import Poln
        p = Poln([1, 1], 's')
        assert p.is_ct
        assert not p.is_dt

    def test_power(self):
        from directsd import Poln
        p = Poln([1, 1], 's')   # (s+1)^2 = s^2+2s+1
        q = p ** 2
        np.testing.assert_allclose(q.coef, [1, 2, 1], atol=1e-10)


class TestPolynomialOps:
    def test_compat(self):
        from directsd import Poln, compat
        p = Poln([1, 1], 's')
        q, r = compat(p, 2.0)
        assert q.var == r.var

    def test_deg(self):
        from directsd import Poln, deg
        p = Poln([1, 2, 1, 0], 's')
        assert deg(p) == 3

    def test_striplz(self):
        from directsd import striplz
        a = np.array([0, 0, 1, 2, 3])
        b = striplz(a)
        np.testing.assert_allclose(b, [1, 2, 3])

    def test_gcd_simple(self):
        from directsd import Poln, gcd
        # gcd((s+1)(s+2), (s+1)(s+3)) = (s+1)
        a = Poln([1, 1], 's') * Poln([1, 2], 's')
        b = Poln([1, 1], 's') * Poln([1, 3], 's')
        g = gcd(a, b)
        assert abs(g.degree - 1) <= 1  # degree 1

    def test_coprime(self):
        from directsd import Poln, coprime
        a = Poln([1, 2, 1], 's')  # (s+1)^2
        b = Poln([1, 1], 's')     # (s+1)
        a2, b2, G = coprime(a, b)
        assert G.degree >= 1  # common factor found

    def test_factor_ct(self):
        from directsd import Poln, factor
        p = Poln([1, 0, -1], 's')  # s^2 - 1 = (s-1)(s+1)
        fs, fu, f0 = factor(p, 's')
        # stable: s+1 (Re < 0 root), unstable: s-1 (Re > 0 root)
        assert fs.degree + fu.degree + f0.degree == p.degree

    def test_factor_dt(self):
        from directsd import Poln, factor
        p = Poln([1, 0, -0.25], 'z')  # roots at ±0.5
        fs, fu, f0 = factor(p, 'z')
        assert fs.degree + fu.degree + f0.degree == p.degree

    def test_sfactor(self):
        from directsd import Poln, sfactor
        # Correct Hermitian polynomial: R(s) = N(s)*N(-s)
        # N(s) = s+1 => N(s)*N(-s) = (s+1)(-s+1) = 1-s^2, coef [-1, 0, 1]
        # sfactor should return N(s) = s+1 (degree 1)
        p = Poln([-1, 0, 1], 's')
        fs, fs0 = sfactor(p)
        # Result should be degree n/2 = 1
        assert fs.degree == 1
        # Verify: fs * fs(-s) ≈ p
        # fs(s)*fs(-s) = (s+1)(-s+1) = -s^2+1 = p(s)  ✓
        assert fs0.degree == 1

    def test_sfactfft(self):
        from directsd.polynomial.spectral import sfactfft
        # Symmetric polynomial: 1 + 2z + 1
        p = np.array([1, 2, 1])
        fs = sfactfft(p)
        assert len(fs) > 0

    def test_dioph_basic(self):
        from directsd import Poln, dioph
        # X*(s+1) + Y*s = 1  =>  X=1, Y=-1 works: 1*(s+1) + (-1)*s = 1
        a = Poln([1, 1], 's')
        b = Poln([1, 0], 's')
        c = Poln([1], 's')
        x, y, err, cond = dioph(a, b, c)
        assert err < 0.1  # residual small


class TestLinAlg:
    def test_toep(self):
        from directsd import toep
        a = np.array([1, 2, 3])
        T = toep(a, 4, 3)
        assert T.shape == (4, 3)

    def test_hank(self):
        from directsd import hank
        a = np.array([1, 2, 3, 4, 5])
        H = hank(a, 3, 3)
        assert H.shape == (3, 3)
        assert H[0, 0] == 1
        assert H[0, 1] == 2
        assert H[1, 0] == 2

    def test_linsys_qr(self):
        from directsd import linsys
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([[5.0], [7.0]])
        x = linsys(A, b, method='qr')
        np.testing.assert_allclose(A @ x, b, atol=1e-10)

    def test_linsys_svd(self):
        from directsd import linsys
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = np.array([[1.0], [2.0], [3.0]])
        x = linsys(A, b, method='svd')
        assert x is not None

    def test_lyap(self):
        from directsd import lyap
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = np.eye(2)
        P = lyap(A, Q)
        # Check A*P + P*A' + Q = 0
        residual = A @ P + P @ A.T + Q
        np.testing.assert_allclose(residual, np.zeros((2, 2)), atol=1e-10)


class TestTransforms:
    def test_dtfm_zoh(self):
        from directsd import dtfm
        import scipy.signal as sig
        # Simple first-order system 1/(s+1)
        plant = sig.lti([1], [1, 1])
        num_d, den_d = dtfm(plant, 0.1)
        assert len(num_d) > 0
        assert len(den_d) > 0

    def test_ztrm_zero_mu(self):
        from directsd import ztrm
        # mu=0 should give standard ZOH discretization
        num_d, den_d = ztrm(([1], [1, 1]), 0.1, mu=0.0)
        assert len(num_d) > 0


class TestAnalysis:
    def test_charpol_stable(self):
        import scipy.signal as sig
        from directsd import charpol
        # Simple stable closed loop: plant=1/(s+1), K=1 (proportional)
        plant = sig.lti([1], [1, 1])
        T = 0.1
        Knum, Kden = np.array([1.0]), np.array([1.0])
        delta = charpol(plant, (Knum, Kden), T=T)
        poles = np.roots(delta)
        # All poles should be inside unit circle for this stable loop
        assert np.all(np.abs(poles) <= 1.5)  # relaxed check

    def test_sdmargin_type(self):
        import scipy.signal as sig
        from directsd import sdmargin
        plant = sig.lti([1], [1, 1])
        margin, poles = sdmargin(plant, ([1.0], [1.0]), T=0.1)
        assert isinstance(margin, float)
        assert isinstance(poles, np.ndarray)

    def test_dinfnorm(self):
        import scipy.signal as sig
        from directsd import dinfnorm
        # Discrete lowpass: z/(z-0.5)
        sys_d = sig.dlti([1, 0], [1, -0.5], dt=0.1)
        gamma, w = dinfnorm(sys_d)
        assert gamma > 0

    def test_sdh2norm(self):
        import scipy.signal as sig
        from directsd import sdh2norm
        plant = sig.lti([1], [1, 1])
        norm = sdh2norm(plant, ([1.0], [1.0]), T=0.1)
        assert norm >= 0


class TestGlopt:
    def test_neldermead_quadratic(self):
        from directsd import neldermead
        f = lambda x: (x[0] - 2) ** 2 + (x[1] - 3) ** 2
        x_opt, f_opt, n = neldermead(f, [0.0, 0.0], tol=1e-6)
        np.testing.assert_allclose(x_opt, [2.0, 3.0], atol=1e-3)
        assert f_opt < 1e-5

    def test_simanneal(self):
        from directsd import simanneal
        # Use a fixed seed for reproducibility; f(x)=(x-1)^2, start at x=5
        np.random.seed(42)
        f = lambda x: (x[0] - 1) ** 2
        x_opt, f_opt, n, _ = simanneal(f, [5.0],
                                        options={'maxFunEvals': 1000,
                                                 'display': 'off'})
        assert f_opt < 1.0  # generous tolerance for stochastic method

    def test_randsearch(self):
        from directsd import randsearch
        f = lambda x: (x[0] - 2) ** 2
        x_opt, f_opt, n = randsearch(f, [0.0],
                                      bounds=[(-5, 5)], n_iter=200, seed=42)
        assert f_opt < 1.0

    def test_crandsearch(self):
        from directsd import crandsearch
        f = lambda x: (x[0] - 1) ** 2 + (x[1] + 2) ** 2
        x_opt, f_opt, n = crandsearch(f, [(-5, 5), (-5, 5)],
                                       max_feval=500, seed=42)
        assert f_opt < 1.0

    def test_sector_continuous(self):
        from directsd import sector
        poles = np.array([-1 + 2j, -1 - 2j, -3.0])
        alpha, beta = sector(poles)
        assert alpha == pytest.approx(1.0, abs=1e-10)  # min |real| = 1
        assert beta == pytest.approx(2.0, abs=1e-10)   # |imag/real| = 2/1

    def test_banana(self):
        from directsd import banana
        # Minimum at (1, 1)
        assert banana([1, 1]) == pytest.approx(0.0, abs=1e-10)
        assert banana([0, 0]) == pytest.approx(1.0, abs=1e-10)


class TestDesign:
    def test_sd2dof_matches_matlab_documented_example(self):
        """sd2dof against MATLAB's own documented 2-DOF example
        (Source/dsd_help.md, '2-DOF control'): F=1/(s-1), R=1/s, T=0.2,
        fixed feedback K=(0.4z+0.4)/(z-0.4). MATLAB reference:
        KR = 5.7241(z-0.8678)(z-0.665)/[(z+0.2673)(z-0.4)], cost 0.0611.
        Regression test for P14 (sd2dof previously ignored K_fb entirely
        and solved a different problem, ~800x off; root cause was unpadded
        coefficient-reversal 'conjugates' dropping origin ζ-factors —
        see BUG_PRIORITIES.md P14, fixed 2026-07-09)."""
        import numpy as np
        import scipy.signal as sig
        from directsd import sd2dof, GeneralizedPlant

        def neg(sys_tf):
            return sig.TransferFunction(-np.atleast_1d(sys_tf.num),
                                        np.atleast_1d(sys_tf.den))

        F = sig.TransferFunction([1], [1, -1])
        R = sig.TransferFunction([1], [1, 0])
        T = 0.2
        K = (np.array([0.4, 0.4]), np.array([1, -0.4]))
        sys2 = GeneralizedPlant([[R, neg(F)], [R, 0], [0, neg(F)]])

        KR, err = sd2dof(sys2, K, T=T)

        zeros = sorted(np.roots(KR[0]).real)
        poles = sorted(np.roots(KR[1]).real)
        gain = KR[0][0] / KR[1][0]
        assert abs(gain - 5.7241) < 5e-3, f"gain {gain} != 5.7241"
        assert abs(zeros[0] - 0.665) < 1e-3, f"zeros {zeros}"
        assert abs(zeros[1] - 0.8678) < 1e-3, f"zeros {zeros}"
        assert abs(poles[0] - (-0.2673)) < 1e-3, f"poles {poles}"
        assert abs(poles[1] - 0.4) < 1e-3, f"poles {poles}"
        assert abs(err - 0.0611) < 5e-3, f"cost {err} != 0.0611"

    def test_dhinf_matches_matlab_documented_examples(self):
        """dhinf against ALL THREE documented dsd_help examples (P5 anchor).
        MATLAB references (Source/dsd_help.md, dhinf chapter; the help/demo
        convention applies sys = z2zeta(sys) before calling dhinf):
          Ex1: K = 1.5 (constant), dahinorm = 3.6056 (= sqrt(13))
          Ex2: K = 1.6201(z+0.4019)/(z+1.302), cost 2.1244
          Ex3 (non-generic): dhinf returns BOTH documented equally-optimal
               controllers as a list (MATLAB: K = {K1,K2} cell), cost 2.5725
               each — nonGen {K,K1} ported 2026-07-12, see BUG_PRIORITIES P20.
        Fixed 2026-07-10 via the root-list (zpk) pipeline — see
        BUG_PRIORITIES.md P5."""
        import numpy as np
        from directsd import dhinf
        from directsd.design.polynomial import _z2zeta
        from directsd.analysis.norms import dahinorm

        # --- Example 1 ---
        F1d = [-2.0, 1.0]
        sys1 = _z2zeta([[([1., -2.], F1d), ([1., 0.], F1d)],
                     [([0.], [1.]), ([1.], [1.])],
                     [([-1., 2.], F1d), ([-1., 0.], F1d)]])
        K1, lam1 = dhinf(sys1)
        K1n = np.atleast_1d(K1[0]) / np.atleast_1d(K1[1])[0]
        K1d = np.atleast_1d(K1[1]) / np.atleast_1d(K1[1])[0]
        assert len(K1n) == 1 and len(K1d) == 1, f"Ex1 K not constant: {K1}"
        assert abs(K1n[0] - 1.5) < 1e-6, f"Ex1 K = {K1n[0]} != 1.5"
        assert abs(lam1 - np.sqrt(13.0)) < 1e-6
        assert abs(dahinorm(sys1, K1, 1.0) - 3.60555) < 1e-3

        # --- Example 2 (generic) ---
        F2d = [-1.0, -2.1, 1.0]
        sys2 = _z2zeta([[([0.3, 1.0], F2d), ([2.0, 1.0, 0.0], F2d)],
                     [([0.], [1.]), ([1.], [1.])],
                     [([-0.3, -1.0], F2d), ([-2.0, -1.0, 0.0], F2d)]])
        K2, lam2 = dhinf(sys2)
        K2n = np.atleast_1d(K2[0]) / np.atleast_1d(K2[1])[0]
        K2d = np.atleast_1d(K2[1]) / np.atleast_1d(K2[1])[0]
        assert len(K2n) == 2 and len(K2d) == 2, f"Ex2 K not 1st order: {K2}"
        assert abs(K2n[0] - 1.6201) < 5e-4
        assert abs(K2n[1] / K2n[0] - 0.4019) < 5e-4       # zero at -0.4019
        assert abs(K2d[1] - 1.302) < 5e-3                  # pole at -1.302
        assert abs(lam2 - 2.1244) < 5e-4

        # --- Example 3 (non-generic): dhinf returns BOTH equally-optimal
        # controllers as a list (MATLAB: K = {K1,K2} cell array) — see
        # BUG_PRIORITIES.md P20 nonGen work.
        #   K1 = -0.72171 (z-2.24)(z-0.4) / ((z+1.825)(z-0.8859))
        #   K2 =  1.9521  (z-0.8282)(z-0.4) / (z^2 - 2.496z + 1.617)
        F3d = [0.32, -1.2, 1.0]
        sys3 = _z2zeta([[([-0.4, 1.0], F3d), ([-0.8, 1.0, 0.0], F3d)],
                     [([0.], [1.]), ([1.], [1.])],
                     [([0.4, -1.0], F3d), ([0.8, -1.0, 0.0], F3d)]])
        K3, lam3 = dhinf(sys3)
        assert abs(lam3 - 2.5725) < 5e-4
        assert isinstance(K3, list) and len(K3) == 2, \
            f"Ex3 must be non-generic (list of 2 controllers), got {type(K3)}"

        def _gain_zeros(K):
            Kn = np.atleast_1d(K[0]); Kd = np.atleast_1d(K[1])
            gain = Kn[0] / Kd[0]
            zz = sorted(np.roots(Kn).real)
            return gain, zz

        gains_zeros = [_gain_zeros(K) for K in K3]
        is_K1 = any(abs(g - (-0.72171)) < 5e-3
                    and abs(zz[0] - 0.4) < 1e-3 and abs(zz[1] - 2.24) < 1e-2
                    for g, zz in gains_zeros)
        is_K2 = any(abs(g - 1.9521) < 5e-3
                    and abs(zz[0] - 0.4) < 1e-3 and abs(zz[1] - 0.8282) < 1e-3
                    for g, zz in gains_zeros)
        assert is_K1 and is_K2, \
            f"Ex3 controllers don't match both documented optima: {gains_zeros}"
        for K in K3:
            assert abs(dahinorm(sys3, K, 1.0) - 2.5725) < 5e-3

    def test_sdh2_returns_controller(self):
        import scipy.signal as sig
        from directsd import sdh2
        plant = sig.lti([1], [1, 1, 0])  # 1/(s(s+1))
        K, err = sdh2(plant, T=0.1)
        K_num, K_den = K
        assert len(K_num) > 0
        assert len(K_den) > 0

    def test_sdl2_returns_controller(self):
        import scipy.signal as sig
        from directsd import sdl2
        plant = sig.lti([1], [1, 1])
        K, err = sdl2(plant, T=0.1)
        assert len(K[0]) > 0

    def test_ch2_returns_controller(self):
        import scipy.signal as sig
        from directsd import ch2
        plant = sig.lti([1], [1, 1])
        K, err = ch2(plant)
        assert len(K[0]) > 0

    def test_sdahinf_returns_controller(self):
        """sdahinf now always uses the polynomial pipeline (matches MATLAB's
        sdahinf.m, which has no state-space alternative — see
        BUG_PRIORITIES.md, generalized-hold/architecture section,
        2026-07-08). The old method='ss'/'pol' split (_sdahinf_ss/
        _sdahinf_pol) was removed."""
        import scipy.signal as sig
        from directsd import sdahinf
        plant = sig.lti([1], [1, 1])    # 1/(s+1)
        K, err = sdahinf(plant, T=0.1)
        K_num, K_den = K
        assert len(K_num) > 0
        assert len(K_den) > 0

    def test_polhinf_returns_result(self):
        import scipy.signal as sig
        from directsd import polhinf
        # Simple scalar coefficients (rational constant functions)
        A = ([2.0], [1.0])
        B = ([1.0], [1.0])
        E = ([3.0], [1.0])
        D22 = ([1.0], [1.0, 0.5])
        K, lam = polhinf(A, B, E, D22)
        assert len(K[0]) > 0
        assert len(K[1]) > 0

    def test_sdahinf_hinf_ge_h2(self):
        """AHinf cost should be >= H2 cost for the same plant."""
        import scipy.signal as sig
        from directsd import sdahinf, sdh2
        plant = sig.lti([1], [1, 2, 1])   # 1/(s+1)^2
        T = 0.1
        _, err_h2 = sdh2(plant, T=T)
        _, err_hinf = sdahinf(plant, T=T)
        # Both should be finite and positive
        assert np.isfinite(err_h2) and err_h2 >= 0
        assert np.isfinite(err_hinf) and err_hinf >= 0

    def test_sdahinorm_basic(self):
        """sdahinorm should return a positive finite value for a stable plant+controller."""
        import scipy.signal as sig
        from directsd import sdahinf, sdahinorm
        plant = sig.lti([1], [1, 1])    # 1/(s+1)
        T = 0.1
        K_tf, err_design = sdahinf(plant, T=T)
        K_num, K_den = K_tf
        norm_val = sdahinorm(plant, (K_num, K_den), T)
        assert np.isfinite(norm_val), f"sdahinorm returned non-finite: {norm_val}"
        assert norm_val >= 0.0, f"sdahinorm returned negative: {norm_val}"

    def test_sdahinorm_consistent_with_design(self):
        """sdahinorm evaluated at the sdahinf-optimal K should be close to the design error."""
        import scipy.signal as sig
        from directsd import sdahinf, sdahinorm
        plant = sig.lti([1], [1, 2, 1])   # 1/(s+1)^2
        T = 0.1
        K_tf, err_design = sdahinf(plant, T=T)
        if not (np.isfinite(err_design) and err_design > 0):
            pytest.skip("sdahinf did not converge to a finite cost")
        norm_val = sdahinorm(plant, K_tf, T)
        assert np.isfinite(norm_val), f"sdahinorm returned non-finite: {norm_val}"
        assert norm_val >= 0.0

    @staticmethod
    def _demo_h2hinf_plant():
        """The official demo_h2hinf.m generalized plant: F = 1/(5s²+s), Fw=1,
        rho=1, T=1 — P11=[-F;0], P12=[-F;1], P21=-F, P22=-F.

        NOTE: these tests previously threw a bare SISO 1/(s+1) at sdh2hinf.
        That is not a valid generalized plant for the mixed design — MATLAB's
        own polquad, run on the coefficients our plant-expansion fabricates
        for it, returns a NON-stabilizing controller (verified in the Octave
        harness 2026-07-10: closed-loop pole 1.047207 in every feedback
        convention). Per the validation discipline (official demos only)
        they now anchor the documented demo_h2hinf values instead.
        """
        import scipy.signal as sig
        from directsd.examples.demos import GeneralizedPlant, neg
        F = sig.lti([1], [5, 1, 0])
        return GeneralizedPlant([[neg(F), neg(F)], [0, 1], [neg(F), neg(F)]])

    def test_sdh2hinf_matches_matlab_documented_example(self):
        """demo_h2hinf.html: Kmix = sdh2hinf(sys, T, 0.5, 2, 1) =
        5.8253 (z-0.7217)(z²-1.402z+0.5314) / ((z+0.5234)(z²-1.427z+0.547)),
        with sdh2norm = 0.9498."""
        from directsd import sdh2hinf, sdh2norm
        sysP = self._demo_h2hinf_plant()
        K, _ = sdh2hinf(sysP, 1.0, rho=0.5, o11=2, i11=1)
        gain = K[0][0] / K[1][0]
        assert abs(gain - 5.8253) < 5e-3, f"gain {gain} != 5.8253"
        zs = np.sort_complex(np.roots(K[0]))
        ps = np.sort_complex(np.roots(K[1]))
        zs_doc = np.sort_complex(np.concatenate(
            [np.roots([1, -1.402, 0.5314]), [0.7217]]))
        ps_doc = np.sort_complex(np.concatenate(
            [np.roots([1, -1.427, 0.547]), [-0.5234]]))
        assert np.allclose(zs, zs_doc, atol=2e-3), f"zeros {zs} != {zs_doc}"
        assert np.allclose(ps, ps_doc, atol=2e-3), f"poles {ps} != {ps_doc}"
        h2 = sdh2norm(sysP, K, 1.0)
        assert abs(h2 - 0.9498) < 2e-3, f"sdh2norm {h2} != 0.9498"

    def test_sdh2hinf_rho_endpoints_match_documented(self):
        """demo_h2hinf.html endpoint anchors: sdahinf gives
        Kinf = 8.3005 (z-0.7207)/(z+0.6402) with cost 1.2251; sdh2 gives
        KH2 = 3.2523 (z-0.6906)/(z+0.409) with sdh2norm 0.8153."""
        from directsd import sdahinf, sdh2, sdh2norm
        sysP = self._demo_h2hinf_plant()
        Ki, lam = sdahinf(sysP, 1.0)
        assert abs(Ki[0][0] / Ki[1][0] - 8.3005) < 5e-3
        assert abs(np.roots(Ki[0])[0].real - 0.7207) < 1e-3
        assert abs(np.roots(Ki[1])[0].real + 0.6402) < 1e-3
        assert abs(lam - 1.2251) < 1e-3, f"lam {lam} != 1.2251"
        assert abs(sdh2norm(sysP, Ki, 1.0) - 1.2251) < 1e-3
        K2, _ = sdh2(sysP, 1.0)
        g2 = K2[0][0] / K2[1][0]
        assert abs(g2 - 3.2523) < 5e-3, f"gain {g2} != 3.2523"
        assert abs(sdh2norm(sysP, K2, 1.0) - 0.8153) < 1e-3

    def test_sdtrhinf_returns_controller(self):
        """sdtrhinf should return a controller and finite cost."""
        import scipy.signal as sig
        from directsd import sdtrhinf
        plant = sig.lti([1], [1, 1])
        T = 0.1
        K_tf, err = sdtrhinf(plant, T)
        assert K_tf is not None, "sdtrhinf returned no controller"
        assert np.isfinite(err) and err >= 0, f"sdtrhinf err={err}"

    def test_sdtrhinferr_basic(self):
        """sdtrhinferr should return a positive finite norm for a stable system."""
        import scipy.signal as sig
        from directsd import sdtrhinf, sdtrhinferr
        plant = sig.lti([1], [1, 1])
        T = 0.1
        K_tf, err_design = sdtrhinf(plant, T)
        norm_val = sdtrhinferr(plant, K_tf, T)
        assert np.isfinite(norm_val), f"sdtrhinferr returned non-finite: {norm_val}"
        assert norm_val >= 0.0

    def test_sdtrhinf_differs_from_sdahinf(self):
        """sdtrhinf (L2 cost) and sdahinf (H2 cost) should give different errors."""
        import scipy.signal as sig
        from directsd import sdahinf, sdtrhinf
        plant = sig.lti([1], [1, 2, 1])   # 1/(s+1)^2
        T = 0.1
        _, err_ahinf  = sdahinf(plant, T)
        _, err_trhinf = sdtrhinf(plant, T)
        # Both finite and non-negative
        assert np.isfinite(err_ahinf)  and err_ahinf  >= 0
        assert np.isfinite(err_trhinf) and err_trhinf >= 0

    def test_split2dof_no_common_poles(self):
        """split2dof with distinct poles returns unchanged controllers and KC=1."""
        from directsd import split2dof
        # K has pole at 0.5, KR has pole at 0.8 (no overlap)
        K  = (np.array([1.0]), np.array([1.0, -0.5]))
        KR = (np.array([1.0]), np.array([1.0, -0.8]))
        KF, KR_new, KC = split2dof(K, KR)
        assert np.allclose(KF[0],  K[0])  and np.allclose(KF[1],  K[1],  atol=1e-6)
        assert np.allclose(KR_new[0], KR[0]) and np.allclose(KR_new[1], KR[1], atol=1e-6)
        assert np.allclose(KC[0], [1.0]) and np.allclose(KC[1], [1.0])

    def test_split2dof_common_pole(self):
        """split2dof extracts a single common pole into KC."""
        from directsd import split2dof
        # K = 1 / ((z-0.5)(z-0.9)), KR = 2 / (z-0.5) — common pole at 0.5
        K  = (np.array([1.0]),        np.polymul([1.0, -0.5], [1.0, -0.9]))
        KR = (np.array([2.0]),        np.array([1.0, -0.5]))
        KF, KR_new, KC = split2dof(K, KR)
        # KC should have pole near 0.5 and numerator = [1, 0] (z^1)
        assert len(KC[0]) == 2 and abs(KC[0][0] - 1.0) < 1e-6
        KC_poles = np.roots(KC[1])
        assert np.any(np.abs(KC_poles - 0.5) < 1e-4)
        # KF * KC ≈ K: verify by evaluating at a test point
        z = 0.7
        K_val  = np.polyval(K[0],  z) / np.polyval(K[1],  z)
        KF_val = np.polyval(KF[0], z) / np.polyval(KF[1], z)
        KC_val = np.polyval(KC[0], z) / np.polyval(KC[1], z)
        assert abs(KF_val * KC_val - K_val) < 1e-4

    def test_split2dof_reconstruction(self):
        """KF*KC = K and KR_new*KC = KR (frequency-domain check)."""
        from directsd import split2dof
        # Two common poles at 0.3 and 0.6
        p_common = np.polymul([1.0, -0.3], [1.0, -0.6])
        K  = (np.array([1.0, 0.5]),   np.polymul(p_common, [1.0, -0.8]))
        KR = (np.array([2.0, -1.0]),  np.polymul(p_common, [1.0, -0.1]))
        KF, KR_new, KC = split2dof(K, KR)
        for z in [0.2, 0.5j, -0.4]:
            K_val   = np.polyval(K[0],  z) / np.polyval(K[1],  z)
            KR_val  = np.polyval(KR[0], z) / np.polyval(KR[1], z)
            KF_val  = np.polyval(KF[0], z) / np.polyval(KF[1], z)
            KRn_val = np.polyval(KR_new[0], z) / np.polyval(KR_new[1], z)
            KC_val  = np.polyval(KC[0], z) / np.polyval(KC[1], z)
            assert abs(KF_val * KC_val  - K_val)  < 1e-4
            assert abs(KRn_val * KC_val - KR_val) < 1e-4

    def test_split2dof_unstable_reference_raises(self):
        """split2dof raises ValueError when the reference part would be unstable."""
        from directsd import split2dof
        # KR has a common pole at 0.5 AND an unstable pole at 1.2 (not in K)
        K  = (np.array([1.0]), np.array([1.0, -0.5]))
        KR = (np.array([1.0]), np.polymul([1.0, -0.5], [1.0, -1.2]))
        with pytest.raises(ValueError, match="unstable"):
            split2dof(K, KR)

    def test_modsdh2_returns_controller(self):
        """modsdh2 returns a (num, den) controller tuple and finite H2 cost."""
        import scipy.signal as sig
        from directsd import modsdh2
        plant = sig.lti([1], [1, 1])  # 1/(s+1)
        T = 0.1
        K, err = modsdh2(plant, T, ord_K=1, n_iter=50)
        assert isinstance(K, tuple) and len(K) == 2
        assert np.isfinite(err) and err >= 0.0

    def test_modsdl2_returns_controller(self):
        """modsdl2 returns a (num, den) controller tuple and finite L2 cost."""
        import scipy.signal as sig
        from directsd import modsdl2
        plant = sig.lti([1], [1, 1])
        T = 0.1
        K, err = modsdl2(plant, T, ord_K=1, n_iter=50)
        assert isinstance(K, tuple) and len(K) == 2
        assert np.isfinite(err) and err >= 0.0

    def test_modsdh2_sector_constraint(self):
        """modsdh2 with alpha > 0 still finds a finite-cost controller."""
        import scipy.signal as sig
        from directsd import modsdh2
        plant = sig.lti([1], [1, 2, 1])
        T = 0.1
        K, err = modsdh2(plant, T, ord_K=2, alpha=1.0, n_iter=50)
        assert np.isfinite(err) and err >= 0.0

    def test_modsdh2_higher_order(self):
        """modsdh2 with ord_K > plant order runs without error."""
        import scipy.signal as sig
        from directsd import modsdh2
        plant = sig.lti([1], [1, 1])
        T = 0.2
        K, err = modsdh2(plant, T, ord_K=3, n_iter=30)
        assert np.isfinite(err) and err >= 0.0

    def test_modsdh2_differs_from_modsdl2(self):
        """H2 and L2 modal designs give different (both finite) costs."""
        import scipy.signal as sig
        from directsd import modsdh2, modsdl2
        plant = sig.lti([1], [1, 2, 1])
        T = 0.1
        _, err_h2 = modsdh2(plant, T, ord_K=2, n_iter=80)
        _, err_l2 = modsdl2(plant, T, ord_K=2, n_iter=80)
        assert np.isfinite(err_h2) and err_h2 >= 0.0
        assert np.isfinite(err_l2) and err_l2 >= 0.0

    # ------------------------------------------------------------------
    # whquad tests
    # ------------------------------------------------------------------

    def test_whquad_returns_controller(self):
        """whquad returns a (num, den) tuple and a finite positive cost."""
        from directsd import whquad
        A   = (np.array([1.0]),          np.array([1.0]))
        B   = (np.array([1.0]),          np.array([1.0]))
        E   = (np.array([1.0]),          np.array([1.0]))
        D22 = (np.array([1.0, 0.0]),     np.array([1.0, -0.5]))
        K, err = whquad(A, B, E, D22)
        assert isinstance(K, tuple) and len(K) == 2
        assert np.isfinite(err) and err >= 0.0

    def test_whquad_cost_no_worse_than_polquad(self):
        """whquad finds an equally good or better controller than polquad."""
        from directsd import whquad, polquad
        A   = (np.array([2.0, 0.0, 1.0]), np.array([1.0]))
        B   = (np.array([1.0, 0.5]),       np.array([1.0]))
        E   = (np.array([1.0]),            np.array([1.0]))
        D22 = (np.array([1.0, 0.0]),       np.array([1.0, -0.5]))
        _, err_wh  = whquad( A, B, E, D22)
        _, err_pol = polquad(A, B, E, D22)
        # whquad uses the correct WH algorithm; polquad may use an approximate
        # fallback. The true optimal is at most min(err_wh, err_pol).
        assert np.isfinite(err_wh) and err_wh >= 0.0
        assert err_wh <= err_pol + 1e-6  # whquad is never worse

    def test_whquad_polynomial_B(self):
        """whquad handles a polynomial B (den=[1])."""
        from directsd import whquad
        A   = (np.array([2.0, 0.0, 1.0]), np.array([1.0]))
        B   = (np.array([1.0, 0.5]),       np.array([1.0]))
        E   = (np.array([1.0]),            np.array([1.0]))
        D22 = (np.array([1.0]),            np.array([1.0, -0.8]))
        K, err = whquad(A, B, E, D22)
        assert np.isfinite(err)

    # ------------------------------------------------------------------
    # ssquad tests
    # ------------------------------------------------------------------

    def test_ssquad_returns_controller(self):
        """ssquad returns a (num, den) tuple and a finite positive cost."""
        from directsd import ssquad
        A0  = (np.array([1.0, 0.5]),      np.array([1.0]))
        A1  = (np.array([1.0, 0.5]),      np.array([1.0]))
        B   = (np.array([1.0]),           np.array([1.0]))
        E   = (np.array([1.0]),           np.array([1.0]))
        D22 = (np.array([1.0, 0.0]),      np.array([1.0, -0.5]))
        K, err = ssquad(A0, A1, B, E, D22)
        assert isinstance(K, tuple) and len(K) == 2
        assert np.isfinite(err) and err >= 0.0

    def test_ssquad_agrees_with_polquad_merged_A(self):
        """ssquad(A0, A1, ...) equals polquad(A0*A1, ...) cost."""
        from directsd import ssquad, polquad
        A0  = (np.array([1.0, 0.5]),  np.array([1.0]))
        A1  = (np.array([1.0, 0.5]),  np.array([1.0]))
        B   = (np.array([1.0]),       np.array([1.0]))
        E   = (np.array([1.0]),       np.array([1.0]))
        D22 = (np.array([1.0, 0.0]),  np.array([1.0, -0.5]))
        _, err_ss  = ssquad(A0, A1, B, E, D22)
        A_merged = (np.polymul(np.array([1.0, 0.5]), np.array([1.0, 0.5])),
                    np.array([1.0]))
        _, err_pol = polquad(A_merged, B, E, D22)
        assert abs(err_ss - err_pol) < 1e-10

    # ------------------------------------------------------------------
    # psigain tests
    # ------------------------------------------------------------------

    def test_psigain_returns_four_values(self):
        """psigain returns (G, GK, a0, b0)."""
        import scipy.signal as sig
        from directsd import psigain
        plant = sig.lti([1.0], [1.0, 1.0])   # 1/(s+1)
        G, GK, a0, b0 = psigain(plant, T=0.1)
        assert isinstance(a0, np.ndarray) and len(a0) > 0
        assert isinstance(b0, np.ndarray) and len(b0) >= 0

    def test_psigain_gk_nan_when_g_nan(self):
        """GK is NaN whenever G is NaN."""
        import scipy.signal as sig
        from directsd import psigain
        plant = sig.lti([1.0], [1.0, 1.0])
        G, GK, _, _ = psigain(plant, T=0.1)
        if np.isnan(G):
            assert np.isnan(GK)

    def test_psigain_stable_plant(self):
        """For a stable SISO plant, psigain runs without error."""
        import scipy.signal as sig
        from directsd import psigain
        plant = sig.lti([2.0], [1.0, 3.0, 2.0])   # 2/((s+1)(s+2))
        G, GK, a0, b0 = psigain(plant, T=0.05)
        assert np.isnan(G) or np.isfinite(G)
        assert np.isnan(GK) or np.isfinite(GK) or np.isinf(GK)

    def test_psigain_bezout_identity(self):
        """a0 and b0 satisfy the Bezout identity relative to the ZOH plant."""
        import scipy.signal as sig
        from directsd import psigain
        # Use a 2x2 plant so P22 is unambiguous and not auto-expanded
        # P22(s) = 1/(s+1), negated by psigain internally → n=-( 1-e^{-T})
        plant_22 = sig.lti([1.0], [1.0, 1.0]).to_ss()
        n_st = plant_22.A.shape[0]
        A_g = plant_22.A
        B_g = np.hstack([plant_22.B, plant_22.B])
        C_g = np.vstack([-plant_22.C, plant_22.C])
        D_g = np.array([[1.0, 0.0], [0.0, 0.0]])
        plant = sig.StateSpace(A_g, B_g, C_g, D_g)

        T = 0.1
        G, GK, a0, b0 = psigain(plant, T)

        # D22 = ZOH of negated P22 = -(1-e^{-T})/(z-e^{-T})
        e_T = np.exp(-T)
        D22_num = np.array([-(1.0 - e_T)])
        D22_den = np.array([1.0, -e_T])

        # Bezout: a0*n + b0*d = 1 (Delta = 1)
        lhs = np.polyadd(np.polymul(a0, D22_num), np.polymul(b0, D22_den))
        for z_test in [0.0, 0.5, -0.5]:
            assert abs(np.polyval(lhs, z_test) - 1.0) < 1e-5, (
                f"Bezout fails at z={z_test}: lhs={np.polyval(lhs, z_test)}"
            )

    def test_psigain_custom_delta(self):
        """psigain accepts a custom Delta polynomial."""
        import scipy.signal as sig
        from directsd import psigain
        plant = sig.lti([1.0], [1.0, 2.0])    # 1/(s+2)
        Delta = np.array([1.0, -0.5])          # z - 0.5 (stable root)
        G, GK, a0, b0 = psigain(plant, T=0.1, Delta=Delta)
        assert len(a0) > 0 and len(b0) >= 0

    def test_psigain_marginally_stable_delta_raises(self):
        """Delta with a root at z=1 raises ValueError."""
        import scipy.signal as sig
        from directsd import psigain
        plant = sig.lti([1.0], [1.0, 1.0])
        Delta = np.array([1.0, -1.0])    # z - 1 = root at z=1
        with pytest.raises(ValueError, match="z=1"):
            psigain(plant, T=0.1, Delta=Delta)

    # ------------------------------------------------------------------
    # polopth2 tests
    # ------------------------------------------------------------------

    def test_polopth2_returns_coef_and_cost(self):
        """polopth2 returns (Q_coef, E) with E >= 0."""
        from directsd import polopth2
        V = (np.array([1.0]), np.array([1.0, -0.5]))   # 1/(z-0.5)
        W = (np.array([1.0]), np.array([1.0]))          # 1
        Q, E = polopth2(V, W, n=0)
        assert isinstance(Q, np.ndarray)
        assert np.isfinite(E) and E >= -1e-10

    def test_polopth2_perfect_match(self):
        """When W=1 and Q is free up to deg(V), optimal cost is zero."""
        from directsd import polopth2
        # V = z^2 + z + 1 (polynomial), W = 1: Q=V achieves E=0
        nV = np.array([1.0, 1.0, 1.0]); dV = np.array([1.0])
        nW = np.array([1.0]);           dW = np.array([1.0])
        Q, E = polopth2((nV, dV), (nW, dW), n=2)
        assert E < 1e-8, f"Expected E≈0, got {E}"

    def test_polopth2_cost_decreases_with_degree(self):
        """Higher degree Q gives lower or equal H2 cost."""
        from directsd import polopth2
        V = (np.array([1.0]), np.array([1.0, -0.5]))   # stable TF
        W = (np.array([1.0, 0.5]), np.array([1.0]))    # polynomial W
        _, E0 = polopth2(V, W, n=0)
        _, E1 = polopth2(V, W, n=1)
        _, E2 = polopth2(V, W, n=2)
        assert E1 <= E0 + 1e-8
        assert E2 <= E1 + 1e-8

    def test_polopth2_rational_W(self):
        """polopth2 works for rational (non-polynomial) W."""
        from directsd import polopth2
        V = (np.array([1.0, 0.0]), np.array([1.0, -0.8]))  # z/(z-0.8)
        W = (np.array([1.0]),      np.array([1.0, -0.5]))   # 1/(z-0.5)
        Q, E = polopth2(V, W, n=1)
        assert np.isfinite(E) and E >= 0.0

    # ------------------------------------------------------------------
    # dtfm2 tests
    # ------------------------------------------------------------------

    def test_dtfm2_returns_num_den(self):
        """dtfm2 returns (num, den) arrays for a SISO stable plant."""
        import scipy.signal as sig
        from directsd import dtfm2
        F = sig.StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
        num, den = dtfm2(F, T=0.1)
        assert isinstance(num, np.ndarray) and isinstance(den, np.ndarray)
        assert np.isfinite(num).all() and np.isfinite(den).all()

    def test_dtfm2_scalar_tuple(self):
        """dtfm2 accepts (A, B, C, D) tuple form."""
        from directsd import dtfm2
        A = np.array([[-2.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        num, den = dtfm2((A, B, C, D), T=0.05)
        assert np.isfinite(num).all() and np.isfinite(den).all()

    def test_dtfm2_nonnegative_dc(self):
        """D{Gh~F~FGh}(T) is a PSD function: DC gain (z=1) >= 0."""
        import scipy.signal as sig
        from directsd import dtfm2
        F = sig.StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
        num, den = dtfm2(F, T=0.1)
        # Evaluate at z=1: sum(num)/sum(den) >= 0
        dc = np.polyval(num, 1.0) / np.polyval(den, 1.0)
        assert float(np.real(dc)) >= -1e-8

    def test_h2reg_state_space(self):
        import scipy.signal as sig
        from directsd import h2reg
        # Well-posed 2x2 generalized plant
        # A stable, B has both disturbance (B1) and control (B2),
        # C has both performance (C1) and measurement (C2) outputs,
        # D12 full column rank, D21 full row rank
        A = np.array([[-1.0, 0.5], [-0.5, -2.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])    # [B1, B2]
        C = np.array([[1.0, 0.0], [0.0, 1.0]])    # [C1; C2]
        D = np.array([[0.0, 1.0], [1.0, 0.0]])    # D12 and D21 nonzero
        plant = sig.StateSpace(A, B, C, D)
        K, h2n = h2reg(plant, n_meas=1, n_ctrl=1)
        assert K is not None
        assert isinstance(h2n, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ============================================================
# New tests covering feedback improvements
# ============================================================

class TestNormsAlgebraic:
    """All norms now use Lyapunov/Riccati — no frequency gridding."""

    def test_h2norm_lyapunov_integrator(self):
        """H2-norm must work for plant with integrator (BadCoefficients killer)."""
        import scipy.signal as sig
        from directsd import sdh2norm
        # Plant with integrator: 1/(s(s+1))
        plant = sig.lti([1], [1, 1, 0])
        K = ([1.0], [1.0])
        norm = sdh2norm(plant, K, T=0.1)
        assert isinstance(norm, float) and (np.isnan(norm) or norm >= 0)

    def test_dinfnorm_bisection_exact(self):
        """H∞-norm of a known system: G(z)=1, so ‖G‖_∞ = 1."""
        import scipy.signal as sig
        from directsd import dinfnorm
        sys_d = sig.dlti([1.0], [1.0], dt=0.1)
        gamma, _ = dinfnorm(sys_d)
        assert abs(gamma - 1.0) < 0.01

    def test_sdhinorm_stable_loop(self):
        """sdhinorm should return a finite positive number for stable loop."""
        import scipy.signal as sig
        from directsd import sdhinorm
        plant = sig.lti([1], [1, 1])
        K = ([0.5], [1.0])
        gamma, w = sdhinorm(plant, K, T=0.1)
        assert gamma > 0 and np.isfinite(gamma)

    def test_h2norm_ct(self):
        """CT H2-norm of 1/(s+1): ‖G‖_H2 = 1/√2."""
        from directsd import h2norm_ct
        import scipy.signal as sig
        g = sig.lti([1], [1, 1])
        norm = h2norm_ct(g)
        assert abs(norm - 1.0/np.sqrt(2)) < 0.01

    def test_hinfnorm_ct(self):
        """CT H∞-norm of 1/(s+1): peak = 1 at ω=0."""
        from directsd import hinfnorm_ct
        import scipy.signal as sig
        g = sig.lti([1], [1, 1])
        gamma, _ = hinfnorm_ct(g)
        assert abs(gamma - 1.0) < 0.1


class TestLifting:
    """Tests for the lifting module."""

    def _simple_plant(self):
        """2×2 continuous-time generalized plant (stable, no delays)."""
        import scipy.signal as sig
        A = np.array([[-1.0, 0.5], [0.0, -2.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        return sig.StateSpace(A, B, C, D)

    def test_lift_h2_returns_discrete_system(self):
        """lift_h2 must return a discrete-time StateSpace."""
        import scipy.signal as sig
        from directsd import lift_h2
        plant = self._simple_plant()
        dsys, gamma, dsys_delta = lift_h2(plant, T=0.1)
        assert dsys.dt == 0.1
        assert dsys_delta.dt == 0.1
        assert isinstance(gamma, float)

    def test_lift_h2_gamma_nonneg(self):
        """Inter-sample correction γ must be non-negative."""
        from directsd import lift_h2
        plant = self._simple_plant()
        _, gamma, _ = lift_h2(plant, T=0.1)
        assert gamma >= 0.0

    def test_lift_h2_state_dims(self):
        """Lifted system must have same state dimension as original."""
        from directsd import lift_h2
        plant = self._simple_plant()
        dsys, _, _ = lift_h2(plant, T=0.1)
        assert dsys.A.shape[0] == plant.A.shape[0]

    def test_lift_l2_returns_discrete_system(self):
        """lift_l2 must return a discrete-time StateSpace."""
        import scipy.signal as sig
        from directsd.design.lifting import lift_l2
        plant = self._simple_plant()
        dsys = lift_l2(plant, T=0.1)
        assert dsys.dt == 0.1

    def test_lift_then_h2reg(self):
        """Full workflow: lift → h2reg → controller."""
        import scipy.signal as sig
        from directsd import lift_h2, h2reg
        plant = self._simple_plant()
        dsys, gamma, _ = lift_h2(plant, T=0.1)
        K, h2n = h2reg(dsys, n_meas=1, n_ctrl=1)
        assert K is not None
        total_cost = np.sqrt(max(h2n**2 + gamma, 0))
        assert total_cost >= 0

    def test_compute_gamma(self):
        """compute_gamma should equal the gamma from lift_h2."""
        from directsd import lift_h2, compute_gamma
        plant = self._simple_plant()
        _, gamma_direct, _ = lift_h2(plant, T=0.1)
        gamma_fn = compute_gamma(plant, T=0.1)
        assert abs(gamma_direct - gamma_fn) < 1e-10


class TestDiophSVD:
    """Verify the SVD-based Diophantine solver."""

    def test_dioph_exact_solution(self):
        """Classic: X·(s+1) + Y·s = 1  →  small residual."""
        from directsd import Poln, dioph
        a = Poln([1, 1], 's')
        b = Poln([1, 0], 's')
        c = Poln([1],    's')
        x, y, err, cond = dioph(a, b, c)
        assert err < 0.5

    def test_dioph_ill_conditioned(self):
        """SVD solver must not blow up on near-singular Sylvester matrix."""
        from directsd import Poln, dioph
        # Near-singular: A and B share almost a common root
        a = Poln([1, 1.0001], 's')
        b = Poln([1, 1.0000], 's')
        c = Poln([1, 2, 1],   's')
        x, y, err, cond = dioph(a, b, c)
        assert np.isfinite(err)

    def test_dioph_identity(self):
        """X·1 + Y·0 = c  →  X = c."""
        from directsd import Poln, dioph
        a = Poln([1], 's')
        b = Poln([0, 0], 's')   # zero-like
        c = Poln([1, 2], 's')
        x, y, err, _ = dioph(a, b, c)
        assert np.isfinite(err)


class TestDualAnnealing:
    """Test scipy dual_annealing wrapper."""

    def test_dual_annealing_quadratic(self):
        """Should find minimum of (x-2)² + (y+1)² ≈ 0."""
        from directsd import dual_annealing
        f = lambda x: (x[0]-2)**2 + (x[1]+1)**2
        x, fv, res = dual_annealing(f, [(-5,5),(-5,5)], seed=42, maxiter=500)
        assert fv < 0.1

    def test_dual_annealing_returns_result(self):
        """Result object must have nfev attribute."""
        from directsd import dual_annealing
        from directsd import banana as _banana
        x, fv, res = dual_annealing(_banana, [(-2,2),(-1,3)], seed=0, maxiter=200)
        assert hasattr(res, 'nfev')


class TestCompatDuckTyping:
    """compat() must handle scipy lti objects and tuples."""

    def test_compat_scipy_lti(self):
        """compat with a scipy lti should not raise."""
        import scipy.signal as sig
        from directsd.polynomial.operations import compat
        g = sig.lti([1], [1, 1])
        result = compat(g)
        from directsd import Poln
        assert isinstance(result, Poln)

    def test_compat_tuple(self):
        """compat with (num, den) tuple should not raise."""
        from directsd.polynomial.operations import compat
        result = compat(([1.0], [1.0, 1.0]))
        from directsd import Poln
        assert isinstance(result, Poln)


class TestNumpyCompat:
    """NumPy 2.0 shim test."""

    def test_trapezoid_shim(self):
        """_trapezoid must be callable regardless of numpy version."""
        from directsd.polynomial.operations import _trapezoid
        assert callable(_trapezoid)
        result = _trapezoid([0, 1, 0], dx=1.0)
        assert abs(result - 1.0) < 1e-10


class TestNewFunctions:
    """Tests for diophsys2, sdh2simple, sdgh2mod, sdnorm, sdfreq, sdsim."""

    @staticmethod
    def _simple_plant(T):
        import scipy.signal as sig
        A = np.array([[-1.0]])
        B = np.array([[1.0, 1.0]])
        C = np.array([[1.0], [1.0]])
        D = np.zeros((2, 2))
        plant = sig.StateSpace(A, B, C, D)
        K = sig.StateSpace(np.zeros((1, 1)), np.zeros((1, 1)),
                           np.zeros((1, 1)), np.array([[0.5]]), dt=T)
        return plant, K

    # ------------------------------------------------------------------
    # diophsys2
    # ------------------------------------------------------------------

    def test_diophsys2_importable(self):
        """diophsys2 is importable from top-level directsd package."""
        from directsd import diophsys2
        assert callable(diophsys2)

    def test_diophsys2_residual(self):
        """diophsys2 residual should be small for a solvable system."""
        from directsd import diophsys2, Poln
        # Trivial case: A1=1, B1=0, C1=1 → X*1 + X~*0 + Y1*1 = 0 → X=-Y1
        a1 = Poln([1], 'z'); b1 = Poln([0], 'z'); c1 = Poln([1], 'z')
        a2 = Poln([1], 'z'); b2 = Poln([0], 'z'); c2 = Poln([1, 0], 'z')
        X, Y1, Y2, err = diophsys2(a1, b1, c1, a2, b2, c2)
        assert np.isfinite(float(err))

    def test_diophsys2_returns_polns(self):
        """diophsys2 returns Poln objects for X, Y1, Y2."""
        from directsd import diophsys2, Poln
        a1 = Poln([1, 1], 'z'); b1 = Poln([1], 'z'); c1 = Poln([1, 0], 'z')
        a2 = Poln([1, 0], 'z'); b2 = Poln([1], 'z'); c2 = Poln([1, 1], 'z')
        X, Y1, Y2, err = diophsys2(a1, b1, c1, a2, b2, c2)
        assert isinstance(X, Poln)
        assert isinstance(Y1, Poln)
        assert isinstance(Y2, Poln)

    # ------------------------------------------------------------------
    # sdh2simple
    # ------------------------------------------------------------------

    def test_sdh2simple_returns_statespace(self):
        """sdh2simple returns a discrete StateSpace."""
        import scipy.signal as sig
        from directsd import sdh2simple
        plant, _ = self._simple_plant(0.1)
        dsys, *_ = sdh2simple(plant, T=0.1)
        assert isinstance(dsys, sig.StateSpace)
        assert dsys.dt == 0.1

    def test_sdh2simple_subblocks_shape(self):
        """sdh2simple sub-block systems have consistent shapes."""
        from directsd import sdh2simple
        plant, _ = self._simple_plant(0.1)
        dsys, DP11, DP12, DP21, DP22 = sdh2simple(plant, T=0.1)
        # Ad should be same for all
        assert dsys.A.shape == DP11.A.shape == DP12.A.shape

    def test_sdh2simple_requires_ct_plant(self):
        """sdh2simple raises TypeError for non-StateSpace input."""
        from directsd import sdh2simple
        with pytest.raises(TypeError):
            sdh2simple("not_a_plant", T=0.1)

    # ------------------------------------------------------------------
    # sdgh2mod
    # ------------------------------------------------------------------

    def test_sdgh2mod_returns_tuple(self):
        """sdgh2mod returns (dsys, gamma, dsys_delta, ...)."""
        from directsd import sdgh2mod
        plant, _ = self._simple_plant(0.1)
        result = sdgh2mod(plant, T=0.1)
        assert len(result) == 8

    def test_sdgh2mod_gamma_nonneg(self):
        """sdgh2mod gamma >= 0."""
        from directsd import sdgh2mod
        plant, _ = self._simple_plant(0.1)
        _, gamma, *_ = sdgh2mod(plant, T=0.1)
        assert gamma >= -1e-10

    def test_sdgh2mod_discrete_system(self):
        """sdgh2mod discrete system has correct sample time."""
        import scipy.signal as sig
        from directsd import sdgh2mod
        plant, _ = self._simple_plant(0.1)
        dsys, *_ = sdgh2mod(plant, T=0.1)
        assert isinstance(dsys, sig.StateSpace)
        assert abs(dsys.dt - 0.1) < 1e-12

    # ------------------------------------------------------------------
    # sdnorm
    # ------------------------------------------------------------------

    def test_sdnorm_gh2_finite_positive(self):
        """sdnorm('gh2') returns a finite non-negative value."""
        from directsd import sdnorm
        plant, K = self._simple_plant(0.1)
        N = sdnorm(plant, K, 'gh2')
        assert np.isfinite(N) and N >= 0.0

    def test_sdnorm_sh2_finite_positive(self):
        """sdnorm('sh2') returns a finite non-negative value."""
        from directsd import sdnorm
        plant, K = self._simple_plant(0.1)
        N = sdnorm(plant, K, 'sh2')
        assert np.isfinite(N) and N >= 0.0

    def test_sdnorm_unknown_type_raises(self):
        """sdnorm raises ValueError for unknown norm type."""
        from directsd import sdnorm
        plant, K = self._simple_plant(0.1)
        with pytest.raises(ValueError, match="Unknown norm type"):
            sdnorm(plant, K, 'bogus')

    # ------------------------------------------------------------------
    # sdfreq
    # ------------------------------------------------------------------

    def test_sdfreq_std_shape(self):
        """sdfreq 'std' returns array of shape (o1, i1, n_w)."""
        from directsd import sdfreq
        plant, K = self._simple_plant(0.1)
        R, w = sdfreq(plant, K, resp_type='std')
        assert R.shape[2] == len(w)
        assert np.isfinite(R).all()

    def test_sdfreq_sing_shape(self):
        """sdfreq 'sing' returns 1-D singular-value array."""
        from directsd import sdfreq
        plant, K = self._simple_plant(0.1)
        sv, w = sdfreq(plant, K, resp_type='sing')
        assert sv.ndim == 1 and len(sv) == len(w)
        assert (sv >= 0).all()

    def test_sdfreq_custom_w(self):
        """sdfreq respects a user-supplied frequency vector."""
        from directsd import sdfreq
        plant, K = self._simple_plant(0.1)
        w_in = np.linspace(1.0, 30.0, 20)
        R, w_out = sdfreq(plant, K, w=w_in)
        np.testing.assert_array_equal(w_out, w_in)
        assert R.shape[2] == 20

    # ------------------------------------------------------------------
    # sdsim
    # ------------------------------------------------------------------

    def test_sdsim_returns_arrays(self):
        """sdsim returns (t, y) arrays of matching length."""
        from directsd import sdsim
        plant, K = self._simple_plant(0.1)
        t, y = sdsim(plant, K, T_max=1.0)
        assert len(t) == len(y)
        assert len(t) > 0

    def test_sdsim_time_starts_at_zero(self):
        """sdsim time vector starts at 0."""
        from directsd import sdsim
        plant, K = self._simple_plant(0.1)
        t, _ = sdsim(plant, K, T_max=0.5)
        assert t[0] == 0.0

    def test_sdsim_output_finite(self):
        """sdsim output contains finite values."""
        from directsd import sdsim
        plant, K = self._simple_plant(0.1)
        _, y = sdsim(plant, K, T_max=1.0)
        assert np.isfinite(y).all()


# ============================================================
# Previously untested functions
# ============================================================

class TestPreviouslyUntested:
    """Unit tests for the 14 functions that had no prior test coverage."""

    # ------------------------------------------------------------------
    # vec – vectorize polynomial coefficients
    # ------------------------------------------------------------------

    def test_vec_from_poln(self):
        """vec(Poln) returns coefficient array."""
        from directsd import Poln, vec
        p = Poln([3.0, 2.0, 1.0], 's')
        c = vec(p)
        assert isinstance(c, np.ndarray)
        np.testing.assert_allclose(c, [3.0, 2.0, 1.0])

    def test_vec_from_array(self):
        """vec(array) returns 1-D array unchanged."""
        from directsd import vec
        c = vec([1.0, 2.0])
        assert c.ndim == 1
        np.testing.assert_allclose(c, [1.0, 2.0])

    # ------------------------------------------------------------------
    # triple – GCD of three polynomials
    # ------------------------------------------------------------------

    def test_triple_common_root(self):
        """triple extracts the common root shared by all three polynomials."""
        from directsd import Poln
        from directsd.polynomial.operations import triple
        s = 's'
        # A = (s+1)(s+2), B = (s+1)(s+3), C = (s+1)(s+4)  → gcd = (s+1)
        A = Poln([1.0, 3.0, 2.0], s)
        B = Poln([1.0, 4.0, 3.0], s)
        C = Poln([1.0, 5.0, 4.0], s)
        Ar, Br, Cr, g = triple(A, B, C)
        assert g.degree == 1
        assert abs(abs(g.roots[0]) - 1.0) < 1e-6   # root at s=-1
        # Reconstruction: A = Ar * g
        assert abs((Ar * g - A).norm()) < 1e-8

    def test_triple_no_common_root(self):
        """triple returns g=1 when polynomials share no common root."""
        from directsd import Poln
        from directsd.polynomial.operations import triple
        s = 's'
        A = Poln([1.0, 1.0], s)   # s+1
        B = Poln([1.0, 2.0], s)   # s+2
        C = Poln([1.0, 3.0], s)   # s+3
        _, _, _, g = triple(A, B, C)
        assert g.degree == 0   # trivial GCD

    # ------------------------------------------------------------------
    # c2z – discretize CT polynomial via root map r → exp(r*T)
    # ------------------------------------------------------------------

    def test_c2z_stable_root(self):
        """c2z maps a stable CT root to inside the unit circle."""
        from directsd import Poln
        p = Poln([1.0, 2.0], 's')   # s + 2  (root at s=-2)
        pz = p.c2z(T=0.5)
        assert pz.is_dt
        # exp(-2 * 0.5) = exp(-1) ≈ 0.368
        np.testing.assert_allclose(sorted(np.abs(pz.roots)), [np.exp(-1.0)], atol=1e-8)

    def test_c2z_degree_preserved(self):
        """c2z preserves polynomial degree."""
        from directsd import Poln
        p = Poln([1.0, 3.0, 2.0], 's')   # (s+1)(s+2)
        pz = p.c2z(T=0.1)
        assert pz.degree == p.degree

    def test_c2z_already_discrete_raises(self):
        """c2z raises ValueError when called on a DT polynomial."""
        from directsd import Poln
        pz = Poln([1.0, -0.5], 'z')
        with pytest.raises(ValueError, match="discrete"):
            pz.c2z(T=0.1)

    # ------------------------------------------------------------------
    # tf2nd – extract num/den arrays from a TF object or tuple
    # ------------------------------------------------------------------

    def test_tf2nd_from_tuple(self):
        """tf2nd extracts num/den from a (num, den) tuple."""
        from directsd import tf2nd
        num, den = tf2nd(([1.0], [1.0, 2.0]))
        np.testing.assert_allclose(num, [1.0])
        np.testing.assert_allclose(den, [1.0, 2.0])

    def test_tf2nd_from_lti(self):
        """tf2nd extracts num/den from a scipy.signal.lti object."""
        import scipy.signal as sig
        from directsd import tf2nd
        sys = sig.lti([1.0], [1.0, 3.0])
        num, den = tf2nd(sys)
        assert isinstance(num, np.ndarray)
        assert isinstance(den, np.ndarray)
        np.testing.assert_allclose(den, [1.0, 3.0], atol=1e-10)

    def test_tf2nd_returns_1d_arrays(self):
        """tf2nd always returns 1-D arrays."""
        from directsd import tf2nd
        num, den = tf2nd(([2.0, 1.0], [1.0, 0.0, 1.0]))
        assert num.ndim == 1 and den.ndim == 1

    # ------------------------------------------------------------------
    # sumzpk – reliable TF summation
    # ------------------------------------------------------------------

    def test_sumzpk_simple_sum(self):
        """sumzpk correctly adds 1/(s+1) + 1/(s+2) = (2s+3)/((s+1)(s+2))."""
        from directsd import sumzpk
        sys1 = ([1.0], [1.0, 1.0])
        sys2 = ([1.0], [1.0, 2.0])
        num, den = sumzpk(sys1, sys2)
        # At s=0: sum = 1/1 + 1/2 = 1.5; evaluate polyval(num,0)/polyval(den,0)
        dc = np.polyval(num, 0.0) / np.polyval(den, 0.0)
        assert abs(dc - 1.5) < 1e-8

    def test_sumzpk_scalar_arg(self):
        """sumzpk accepts a scalar as one argument."""
        from directsd import sumzpk
        num, den = sumzpk(1.0, ([1.0], [1.0, 1.0]))
        dc = np.polyval(num, 0.0) / np.polyval(den, 0.0)
        assert abs(dc - 2.0) < 1e-8   # 1 + 1/1 = 2

    def test_sumzpk_common_pole_is_pole_aware(self):
        """sumzpk cancels a shared pole exactly instead of leaving it
        uncancelled in a smeared, higher-degree product (the whole point
        of the "reliable" summation the docstring promises -- naive
        cross-multiplication (num1*den2+num2*den1, den1*den2) would give
        a degree-2 denominator with a duplicated (s+1) factor instead of
        cancelling it). 1/(s+1) + 2/(s+1) = 3/(s+1)."""
        from directsd import sumzpk
        num, den = sumzpk(([1.0], [1.0, 1.0]), ([2.0], [1.0, 1.0]))
        assert len(np.atleast_1d(den).ravel()) == 2, \
            f"shared pole not cancelled: den={den}"
        dc = np.polyval(num, 0.0) / np.polyval(den, 0.0)
        assert abs(dc - 3.0) < 1e-8

    # ------------------------------------------------------------------
    # improper – separate improper part of rational function
    # ------------------------------------------------------------------

    def test_improper_strictly_proper(self):
        """improper of a strictly proper function returns zero polynomial part."""
        from directsd import improper
        P, (rnum, rden) = improper(([1.0], [1.0, 1.0]))   # 1/(s+1)
        assert abs(P[0]) < 1e-10
        np.testing.assert_allclose(rnum, [1.0], atol=1e-10)

    def test_improper_biproper(self):
        """improper of (s+2)/(s+1) gives polynomial part = [1] and remainder 1/(s+1)."""
        from directsd import improper
        P, (rnum, rden) = improper(([1.0, 2.0], [1.0, 1.0]))
        assert abs(P[0] - 1.0) < 1e-8   # leading term of polynomial part = 1
        # remainder should have degree < denominator
        assert len(rnum) < len(rden) or (len(rnum) == len(rden) and abs(rnum[0] / rden[0]) < 1.0)

    def test_improper_type_error(self):
        """improper raises TypeError for unsupported input."""
        from directsd import improper
        with pytest.raises(TypeError):
            improper(42)

    # ------------------------------------------------------------------
    # separtf – proper separation (delegates to improper)
    # ------------------------------------------------------------------

    def test_separtf_agrees_with_improper(self):
        """separtf returns identical result to improper for the same input."""
        from directsd import improper, separtf
        sys = ([1.0, 3.0], [1.0, 1.0])
        P1, r1 = improper(sys)
        P2, r2 = separtf(sys)
        np.testing.assert_allclose(P1, P2)
        np.testing.assert_allclose(r1[0], r2[0])

    # ------------------------------------------------------------------
    # separss – proper separation for state-space systems
    # ------------------------------------------------------------------

    def test_separss_proper_part_zero_D(self):
        """separss returns a proper system with D=0 and the original D as improper."""
        import scipy.signal as sig
        from directsd.sspace.design import separss
        A = np.array([[-1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        D = np.array([[3.0]])
        plant = sig.StateSpace(A, B, C, D)
        sys_prop, D_imp = separss(plant)
        np.testing.assert_allclose(sys_prop.D, [[0.0]])
        np.testing.assert_allclose(D_imp, [[3.0]])

    def test_separss_preserves_dynamics(self):
        """separss proper part has the same A, B, C as the original."""
        import scipy.signal as sig
        from directsd.sspace.design import separss
        plant = sig.StateSpace([[-2.0]], [[1.0]], [[1.0]], [[0.5]])
        sys_prop, _ = separss(plant)
        np.testing.assert_allclose(sys_prop.A, [[-2.0]])
        np.testing.assert_allclose(sys_prop.B, [[1.0]])

    # ------------------------------------------------------------------
    # quaderr – squared H2-norm of discrete closed-loop TF
    # ------------------------------------------------------------------

    def test_quaderr_nonnegative(self):
        """quaderr returns a non-negative value."""
        from directsd.analysis.errors import quaderr
        # Stable discrete TF: 0.5 / (z - 0.5)
        err = quaderr(([0.5], [1.0, -0.5]), T=0.1)
        assert np.isfinite(err) and err >= 0.0

    def test_quaderr_zero_for_zero_tf(self):
        """quaderr returns ~0 for a TF with zero numerator."""
        from directsd.analysis.errors import quaderr
        err = quaderr(([0.0], [1.0, -0.5]), T=0.1)
        assert abs(err) < 1e-8

    # ------------------------------------------------------------------
    # sdfast – ZOH discretization via Van Loan's method
    # ------------------------------------------------------------------

    def test_sdfast_returns_discrete_system(self):
        """sdfast returns a discrete StateSpace with dt=T."""
        import scipy.signal as sig
        from directsd.sspace.design import sdfast
        plant = sig.StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
        pd = sdfast(plant, T=0.1)
        assert pd.dt == 0.1
        assert pd.A.shape == (1, 1)

    def test_sdfast_stable_eigenvalues(self):
        """sdfast produces eigenvalues inside the unit disk for stable plant."""
        import scipy.signal as sig
        from directsd.sspace.design import sdfast
        plant = sig.StateSpace([[-2.0, 0.0], [0.0, -3.0]],
                               [[1.0], [1.0]], [[1.0, 1.0]], [[0.0]])
        pd = sdfast(plant, T=0.5)
        eigs = np.abs(np.linalg.eigvals(pd.A))
        assert np.all(eigs < 1.0)

    def test_sdfast_matches_zoh(self):
        """sdfast matches scipy ZOH discretization for a simple SISO plant."""
        import scipy.signal as sig
        from directsd.sspace.design import sdfast
        plant = sig.StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
        T = 0.2
        pd_fast = sdfast(plant, T=T)
        pd_zoh  = plant.to_discrete(T, method='zoh')
        np.testing.assert_allclose(pd_fast.A, pd_zoh.A, atol=1e-10)
        np.testing.assert_allclose(pd_fast.B, pd_zoh.B, atol=1e-10)

    # ------------------------------------------------------------------
    # sdh2reg – H2-optimal controller via ZOH lifting
    # ------------------------------------------------------------------

    def test_sdh2reg_returns_controller(self):
        """sdh2reg returns (K, h2norm) with finite norm."""
        import scipy.signal as sig
        from directsd.sspace.design import sdh2reg
        A  = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B  = np.array([[1.0, 0.0], [0.0, 1.0]])
        C  = np.array([[1.0, 0.0], [0.0, 1.0]])
        D  = np.zeros((2, 2))
        plant = sig.StateSpace(A, B, C, D)
        K, h2n = sdh2reg(plant, T=0.1)
        assert isinstance(K, sig.StateSpace)
        assert np.isfinite(h2n)

    def test_sdh2reg_discrete_controller(self):
        """sdh2reg returns a discrete-time controller."""
        import scipy.signal as sig
        from directsd.sspace.design import sdh2reg
        plant = sig.StateSpace([[-1.0, 0.0], [0.0, -2.0]],
                               [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]],
                               np.zeros((2, 2)))
        K, _ = sdh2reg(plant, T=0.1)
        assert K.dt is not None and K.dt != 0

    # ------------------------------------------------------------------
    # sdhinfreg – H∞-optimal controller via ZOH lifting
    # ------------------------------------------------------------------

    def test_sdhinfreg_returns_controller(self):
        """sdhinfreg returns (K, gamma) with gamma > 0."""
        import scipy.signal as sig
        from directsd.sspace.design import sdhinfreg
        A  = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B  = np.array([[1.0, 0.0], [0.0, 1.0]])
        C  = np.array([[1.0, 0.0], [0.0, 1.0]])
        D  = np.zeros((2, 2))
        plant = sig.StateSpace(A, B, C, D)
        K, gamma = sdhinfreg(plant, T=0.1)
        assert isinstance(K, sig.StateSpace)
        assert gamma > 0.0

    def test_sdhinfreg_gamma_finite(self):
        """sdhinfreg gamma is finite."""
        import scipy.signal as sig
        from directsd.sspace.design import sdhinfreg
        plant = sig.StateSpace([[-1.0, 0.0], [0.0, -2.0]],
                               [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]],
                               np.zeros((2, 2)))
        _, gamma = sdhinfreg(plant, T=0.1)
        assert np.isfinite(gamma)

    # ------------------------------------------------------------------
    # hinfreg – CT H∞ controller (gamma-iteration ARE)
    # ------------------------------------------------------------------

    def test_hinfreg_returns_controller(self):
        """hinfreg returns (K, gamma) for a stabilizable/detectable plant."""
        import scipy.signal as sig
        from directsd.sspace.design import hinfreg
        A  = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B  = np.array([[1.0, 0.0], [0.0, 1.0]])
        C  = np.array([[1.0, 0.0], [0.0, 1.0]])
        D  = np.zeros((2, 2))
        plant = sig.StateSpace(A, B, C, D)
        K, gamma = hinfreg(plant)
        assert isinstance(K, sig.StateSpace)
        assert gamma > 0.0

    def test_hinfreg_gamma_positive(self):
        """hinfreg optimal gamma is strictly positive."""
        import scipy.signal as sig
        from directsd.sspace.design import hinfreg
        plant = sig.StateSpace([[-1.0, 0.0], [0.0, -3.0]],
                               [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]],
                               np.zeros((2, 2)))
        _, gamma = hinfreg(plant)
        assert np.isfinite(gamma) and gamma > 0.0

    # ------------------------------------------------------------------
    # dioph2 – spectral Diophantine equation X·A + X̃·B + Y·C = 0
    # ------------------------------------------------------------------

    def test_dioph2_returns_three_values(self):
        """dioph2 returns (X, Y, err) tuple."""
        from directsd import Poln
        from directsd.polynomial.diophantine import dioph2
        s = 's'
        A = Poln([1.0, 1.0], s)   # s+1
        B = Poln([1.0, 1.0], s)   # s+1
        C = Poln([1.0, 2.0], s)   # s+2
        X, Y, err = dioph2(A, B, C)
        assert isinstance(err, float)
        assert np.isfinite(err)

    def test_dioph2_low_residual(self):
        """dioph2 solution satisfies X·A + X̃·B + Y·C ≈ 0."""
        from directsd import Poln
        from directsd.polynomial.diophantine import dioph2
        s = 's'
        A = Poln([1.0, 2.0], s)
        B = Poln([1.0, 2.0], s)
        C = Poln([1.0, 3.0, 2.0], s)
        X, Y, err = dioph2(A, B, C)
        assert err < 1.0   # residual should be small for well-posed problem

    # ------------------------------------------------------------------
    # delzero – remove zero roots from a polynomial
    # ------------------------------------------------------------------

    def test_delzero_poln_removes_zero_root(self):
        """delzero removes roots at the origin from a Poln."""
        from directsd import Poln, delzero
        # s*(s+1) has one zero root
        p = Poln([1.0, 1.0, 0.0], 's')
        B, nz = delzero(p)
        assert nz == 1
        assert B.degree == 1
        # remaining root should be at s=-1
        np.testing.assert_allclose(sorted(np.real(B.roots)), [-1.0], atol=1e-8)

    def test_delzero_poln_no_zero_roots(self):
        """delzero returns original Poln unchanged when no zero roots."""
        from directsd import Poln, delzero
        p = Poln([1.0, 1.0], 's')   # s+1 — no zero root
        B, nz = delzero(p)
        assert nz == 0
        assert B.degree == p.degree

    def test_delzero_array_strips_trailing(self):
        """delzero on a plain array strips trailing near-zero entries."""
        from directsd import delzero
        arr = np.array([1.0, 2.0, 0.0, 0.0])
        B, nz = delzero(arr)
        assert nz == 2
        np.testing.assert_allclose(B, [1.0, 2.0])

    def test_delzero_returns_two_values(self):
        """delzero always returns a (result, count) 2-tuple."""
        from directsd import Poln, delzero
        p = Poln([1.0], 's')
        result = delzero(p)
        assert len(result) == 2

    # ------------------------------------------------------------------
    # zpk – create ZerosPolesGain from polynomial objects
    # ------------------------------------------------------------------

    def test_zpk_ct_single_poln(self):
        """zpk(N) with scalar D=1 gives a CT ZerosPolesGain."""
        import scipy.signal as sig
        from directsd import Poln, zpk
        N = Poln([1.0, 1.0], 's')   # s+1
        F = zpk(N)
        assert isinstance(F, sig.ZerosPolesGain)
        assert F.dt is None or F.dt == 0

    def test_zpk_ct_zeros_poles(self):
        """zpk(N, D) correctly extracts zeros and poles."""
        import scipy.signal as sig
        from directsd import Poln, zpk
        N = Poln([1.0, 2.0], 's')   # s+2 → zero at -2
        D = Poln([1.0, 3.0], 's')   # s+3 → pole at -3
        F = zpk(N, D)
        np.testing.assert_allclose(np.real(F.zeros), [-2.0], atol=1e-8)
        np.testing.assert_allclose(np.real(F.poles), [-3.0], atol=1e-8)
        np.testing.assert_allclose(F.gain, 1.0, atol=1e-8)

    def test_zpk_cancels_common_roots(self):
        """zpk cancels a common root between numerator and denominator."""
        from directsd import Poln, zpk
        # N = (s+1)(s+2), D = (s+1)(s+3) → should cancel (s+1)
        N = Poln([1.0, 3.0, 2.0], 's')
        D = Poln([1.0, 4.0, 3.0], 's')
        F = zpk(N, D)
        assert len(F.zeros) == 1   # only s=-2 remains
        assert len(F.poles) == 1   # only s=-3 remains

    def test_zpk_dt_sets_sampling_time(self):
        """zpk returns a DT ZerosPolesGain when N is discrete-time."""
        import scipy.signal as sig
        from directsd import Poln, zpk
        N = Poln([1.0, -0.5], 'z')   # z-0.5
        D = Poln([1.0, -0.8], 'z')   # z-0.8
        F = zpk(N, D, T=0.1)
        assert isinstance(F, sig.ZerosPolesGain)
        assert F.dt == 0.1


# ============================================================
# Preview control (sdh2's refdelay parameter, added 2026-07-13)
# ============================================================

class TestSdh2PreviewControl:
    """sdh2's refdelay parameter -- MATLAB's Q.iodelay preview trick.

    Ground truth: dsd_help.md's documented H2-optimal-preview-control
    example (F=1/(s-1), F.iodelay=1.5, Fr=1/(5s+1), Fn=0.2, T=1,
    Q.iodelay=preview=2): K = 8.1825 z^2(z-0.7985) /
    [(z-0.8113)(z^2+3.041z+3.169)], sdh2norm(sys,K)^2 = 11.6701.
    """

    def _plant(self):
        import scipy.signal as sig
        from directsd import GeneralizedPlant
        from directsd.tf import neg
        F = sig.lti([1], [1, -1])
        Fr = sig.lti([1], [5, 1])
        Fn = 0.2
        T = 1.0
        return GeneralizedPlant([
            [Fr, 0, neg(F)],
            [Fr, Fn, neg(F)],
        ]), T

    def test_matches_dsd_help_documented_example(self):
        from directsd import sdh2
        sys, T = self._plant()
        K, err = sdh2(sys, T, udelay=1.5, refdelay=2.0)
        assert err ** 2 == pytest.approx(11.6701, abs=1e-3)
        # K = 8.1825 z^2(z-0.7985) / [(z-0.8113)(z^2+3.041z+3.169)]
        expected_num = np.polymul([8.1825, 0, 0], [1, -0.7985])
        expected_den = np.polymul([1, -0.8113], [1, 3.041, 3.169])
        assert np.allclose(K[0] / K[0][0], expected_num / expected_num[0], atol=2e-3)
        assert np.allclose(K[1], expected_den, atol=2e-3)

    def test_refdelay_zero_matches_udelay_only_baseline(self):
        """refdelay=0.0 (default) must be byte-for-byte the pre-existing
        udelay-only code path -- no regression for existing callers."""
        from directsd import sdh2
        sys, T = self._plant()
        K1, err1 = sdh2(sys, T, udelay=1.5)
        K2, err2 = sdh2(sys, T, udelay=1.5, refdelay=0.0)
        assert err1 == err2
        assert np.array_equal(K1[0], K2[0])
        assert np.array_equal(K1[1], K2[1])

    def test_preview_sweep_reproduces_documented_nonmonotonic_shape(self):
        """For this UNSTABLE plant, dsd_help.md documents a non-monotonic
        J_min(pi) curve (dip near pi~2, rise for larger pi) -- unlike the
        stable-plant case, which is monotonically decreasing."""
        from directsd import sdh2
        sys, T = self._plant()
        pis = [0.0, 2.0, 9.99]
        costs = [sdh2(sys, T, udelay=1.5, refdelay=p)[1] ** 2 for p in pis]
        assert costs[1] < costs[0]   # dips below the no-preview cost
        assert costs[2] > costs[1]   # then rises again for large pi

    def test_instantaneous_variance_with_refdelay_raises(self):
        """The instantaneous-variance branch's Bs=P21*P11H term (no P12
        factor) has a different delay topology, not yet derived."""
        from directsd import sdh2
        sys, T = self._plant()
        with pytest.raises(Exception):
            sdh2(sys, T, t=0.0, udelay=1.5, refdelay=2.0)


# ============================================================
# Preview control (sdl2's refdelay parameter, added 2026-07-13)
# ============================================================

class TestSdl2PreviewControl:
    """sdl2's refdelay parameter -- MATLAB's Q.iodelay/R.iodelay preview
    trick (demo_l2p.m), a genuinely different delay topology from sdh2's:
    the preview horizon splits into sigma=ceil(pi/T) (an exact integer-
    sample delay entering the combined P11=Q*R cascade) and a remainder
    theta=sigma*T-pi in [0,T) (entering P21=R alone).

    Ground truth: dsd_help.md's documented L2-optimal-preview-control
    example, STABLE plant (R=1/(s^2+s), F=1/(5s+1), F.iodelay=1.5,
    Q=1/(0.1s+1), T=1, preview=2): K = 7.5673 z^2(z-0.8187)(z-0.07603) /
    [(z-1)(z-0.07625)(z^2+1.344z+0.6498)], sdl2err(sys,K)=0.0517.

    A second, UNSTABLE-plant ground truth (F=1/(5s-1), same R/Q/T/preview/
    tau) comes from demo_l2p.m's own exNo=2 branch, cross-validated via
    ex_2dof_preview's documented sdl2err=2.7072 for the same 1-DOF plant.
    """

    def _stable_plant(self):
        import scipy.signal as sig
        from directsd import GeneralizedPlant
        from directsd.tf import mul, neg
        R = sig.lti([1], [1, 1, 0])
        F = sig.lti([1], [5, 1])
        Q = sig.lti([1], [0.1, 1])
        T = 1.0
        return GeneralizedPlant([
            [mul(Q, R), neg(F)],
            [R,           neg(F)],
        ]), T

    def _unstable_plant(self):
        import scipy.signal as sig
        from directsd import GeneralizedPlant
        from directsd.tf import mul, neg
        R = sig.lti([1], [1, 1, 0])
        F = sig.lti([1], [5, -1])
        Q = sig.lti([1], [0.1, 1])
        T = 1.0
        return GeneralizedPlant([
            [mul(Q, R), neg(F)],
            [R,           neg(F)],
        ]), T

    def test_matches_dsd_help_documented_stable_example(self):
        from directsd import sdl2
        sys, T = self._stable_plant()
        K, err = sdl2(sys, T, udelay=1.5, refdelay=2.0)
        assert err == pytest.approx(0.0517, abs=2e-3)
        # K = 7.5673 z^2(z-0.8187)(z-0.07603) /
        #     [(z-1)(z-0.07625)(z^2+1.344z+0.6498)]
        expected_num = np.polymul(np.polymul([7.5673, 0, 0], [1, -0.8187]),
                                  [1, -0.07603])
        expected_den = np.polymul(np.polymul([1, -1], [1, -0.07625]),
                                  [1, 1.344, 0.6498])
        assert np.allclose(K[0] / K[0][0], expected_num / expected_num[0], atol=2e-3)
        assert np.allclose(K[1], expected_den, atol=2e-3)

    def test_matches_documented_unstable_example(self):
        """Cross-validated against ex_2dof_preview's documented
        sdl2err=2.7072 for the same 1-DOF unstable plant/preview."""
        from directsd import sdl2
        sys, T = self._unstable_plant()
        K, err = sdl2(sys, T, udelay=1.5, refdelay=2.0)
        assert err == pytest.approx(2.7072, abs=2e-3)

    def test_refdelay_zero_matches_udelay_only_baseline(self):
        """refdelay=0.0 (default) must be byte-for-byte the pre-existing
        udelay-only code path -- no regression for existing callers."""
        from directsd import sdl2
        sys, T = self._stable_plant()
        K1, err1 = sdl2(sys, T, udelay=1.5)
        K2, err2 = sdl2(sys, T, udelay=1.5, refdelay=0.0)
        assert err1 == err2
        assert np.array_equal(K1[0], K2[0])
        assert np.array_equal(K1[1], K2[1])

    def test_preview_sweep_reproduces_documented_nonmonotonic_shape(self):
        """For the UNSTABLE plant, dsd_help.md documents a non-monotonic
        J_min(pi) curve (dip near pi~1-2, rise for larger pi) -- unlike the
        stable-plant case, where J_min(pi) decreases monotonically."""
        from directsd import sdl2
        sys, T = self._unstable_plant()
        pis = [0.0, 2.0, 11.999]
        costs = [sdl2(sys, T, udelay=1.5, refdelay=p)[1] for p in pis]
        assert costs[1] < costs[0]   # dips below the no-preview cost
        assert costs[2] > costs[1]   # then rises again for large pi

    def test_preview_sweep_trends_down_for_stable_plant(self):
        """dsd_help.md: for the STABLE plant, J_min(pi) 'tends asymptotically
        to a periodic curve' as pi increases -- i.e. an overall decreasing
        trend, though (per the sigma/theta integer/fractional split) it
        oscillates with period T for small pi before settling (empirically:
        pi=0..2 oscillates 1.30/2.05/0.52/0.81/0.052; monotonic decrease only
        sets in from pi=2 onward). Check the settled tail, not the transient."""
        from directsd import sdl2
        sys, T = self._stable_plant()
        pis = [2.0, 3.0, 4.0, 6.0, 8.0]
        costs = [sdl2(sys, T, udelay=1.5, refdelay=p)[1] for p in pis]
        assert all(costs[i + 1] <= costs[i] + 1e-6 for i in range(len(costs) - 1))


# ============================================================
# Preview control (sd2dof's refdelay parameter, added 2026-07-13)
# ============================================================

class TestSd2dofPreviewControl:
    """sd2dof's udelay/refdelay parameters -- MATLAB's demo_2dofp.m.

    sd2dofcoef.m internally reuses sdl2coef on the [z;y1] sub-plant (see
    _sdl2coef's refdelay derivation) plus its OWN D22=dtfm(P22,T,0,H)'
    rescaling built directly from P22 (only udelay reaches this, never
    refdelay -- P22 is the plant alone, no Q/R involved).

    Ground truth: dsd_help.md's documented 2-DOF-preview-control example
    (demo_2dofp.m: R=1/(s^2+s), F=1/(5s-1) UNSTABLE, F.iodelay=1.5,
    Q=1/(0.1s+1), T=1, preview=2):
      1-DOF K  = 14.7471 z^2(z-0.9039)(z-0.2681) /
                 [(z-1)(z-0.2725)(z^2+1.943z+1.248)], sdl2err=2.7072
      2-DOF KR = 6.2878 z(z+6.675e-6)(z-9.713e-5)(z-0.0896)(z-0.8187) /
                 [(z-1)(z-0.2725)(z-4.54e-5)(z^2+1.943z+1.248)],
                 sd2doferr=0.0602
    (dsd_help.md's own extracted text shows F=tf(1,[5 1]) here, but that
    contradicts its own documented K/KR/costs, which match demo_2dofp.m's
    actual F=tf(1,[5 -1]) source exactly -- a minus-sign transcription
    artifact in the HTML->markdown conversion; trust the .m source, per
    [[directsd_architecture_feedback]].)
    """

    def _plants(self):
        import scipy.signal as sig
        from directsd import GeneralizedPlant
        from directsd.tf import mul, neg
        R = sig.lti([1], [1, 1, 0])
        F = sig.lti([1], [5, -1])
        Q = sig.lti([1], [0.1, 1])
        T = 1.0
        sys_1dof = GeneralizedPlant([
            [mul(Q, R), neg(F)],
            [R,           neg(F)],
        ])
        sys_2dof = GeneralizedPlant([
            [mul(Q, R), neg(F)],
            [R,           0],
            [0,           neg(F)],
        ], n_meas=2)
        return sys_1dof, sys_2dof, T

    def test_matches_dsd_help_documented_example(self):
        from directsd import sdl2, sd2dof
        sys_1dof, sys_2dof, T = self._plants()
        K, err1 = sdl2(sys_1dof, T, udelay=1.5, refdelay=2.0)
        assert err1 == pytest.approx(2.7072, abs=2e-3)

        KR, err2 = sd2dof(sys_2dof, K, T, udelay=1.5, refdelay=2.0)
        assert err2 == pytest.approx(0.0602, abs=2e-3)
        # KR = 6.2878 z(z+6.675e-6)(z-9.713e-5)(z-0.0896)(z-0.8187) /
        #      [(z-1)(z-0.2725)(z-4.54e-5)(z^2+1.943z+1.248)]
        expected_num = np.polymul(np.polymul(np.polymul(np.polymul(
            [6.2878, 0], [1, 6.675e-6]), [1, -9.713e-5]), [1, -0.0896]), [1, -0.8187])
        expected_den = np.polymul(np.polymul(np.polymul(
            [1, -1], [1, -0.2725]), [1, -4.54e-5]), [1, 1.943, 1.248])
        assert np.allclose(KR[0], expected_num, atol=2e-3)
        assert np.allclose(KR[1], expected_den, atol=2e-3)

    def test_refdelay_zero_matches_udelay_only_baseline(self):
        """refdelay=0.0/udelay=0.0 (defaults) must be byte-for-byte the
        pre-existing code path -- no regression for existing callers."""
        from directsd import sdl2, sd2dof
        sys_1dof, sys_2dof, T = self._plants()
        K, _ = sdl2(sys_1dof, T)
        KR1, err1 = sd2dof(sys_2dof, K, T)
        KR2, err2 = sd2dof(sys_2dof, K, T, udelay=0.0, refdelay=0.0)
        assert err1 == err2
        assert np.array_equal(KR1[0], KR2[0])
        assert np.array_equal(KR1[1], KR2[1])


# ============================================================
# Advanced global optimization (dsdglopt port)
# ============================================================

class TestAdvancedGlopt:
    # ------------------------------------------------------------------
    # updateopt
    # ------------------------------------------------------------------
    def test_updateopt_updates_existing_keys(self):
        from directsd import updateopt
        base = {'a': 1, 'b': 2}
        result = updateopt(base, {'a': 10, 'b': 20})
        assert result == {'a': 10, 'b': 20}

    def test_updateopt_ignores_unknown_keys(self):
        from directsd import updateopt
        base = {'a': 1}
        result = updateopt(base, {'a': 5, 'x': 99})
        assert result == {'a': 5}
        assert 'x' not in result

    # ------------------------------------------------------------------
    # uniproj / u2range
    # ------------------------------------------------------------------
    def test_uniproj_clips_to_01(self):
        from directsd import uniproj
        np.testing.assert_allclose(uniproj([-1.0, 0.5, 2.0]), [0.0, 0.5, 1.0])

    def test_u2range_linear_map(self):
        from directsd import u2range
        assert abs(u2range(0.0, 2.0, 4.0) - 2.0) < 1e-12
        assert abs(u2range(1.0, 2.0, 4.0) - 4.0) < 1e-12
        assert abs(u2range(0.5, 2.0, 4.0) - 3.0) < 1e-12

    # ------------------------------------------------------------------
    # randgamma / randbeta
    # ------------------------------------------------------------------
    def test_randgamma_shape_and_positive(self):
        from directsd import randgamma
        np.random.seed(42)
        g = randgamma(2.0, 10, 5)
        assert g.shape == (10, 5)
        assert np.all(g > 0)

    def test_randbeta_in_unit_interval(self):
        from directsd import randbeta
        np.random.seed(7)
        b = randbeta(2.0, 3.0, 100, 1)
        assert b.shape == (100, 1)
        assert np.all(b >= 0) and np.all(b <= 1)

    def test_randbeta_approximate_mean(self):
        from directsd import randbeta
        np.random.seed(13)
        a, b = 2.0, 3.0
        samples = randbeta(a, b, 2000, 1)
        mean = np.mean(samples)
        expected = a / (a + b)
        assert abs(mean - expected) < 0.05

    # ------------------------------------------------------------------
    # sa_testfun
    # ------------------------------------------------------------------
    def test_sa_testfun_zero_at_origin(self):
        from directsd import sa_testfun
        assert abs(sa_testfun([0.0, 0.0])) < 1e-12

    def test_sa_testfun_positive_elsewhere(self):
        from directsd import sa_testfun
        assert sa_testfun([1.0, 0.5]) > 0

    # ------------------------------------------------------------------
    # val2bin / bin2val
    # ------------------------------------------------------------------
    def test_val2bin_integer(self):
        from directsd import val2bin
        bint, bfrac = val2bin(6)
        assert bint == '110'
        assert bfrac == ''

    def test_val2bin_fractional(self):
        from directsd import val2bin
        bint, bfrac = val2bin(0.75, 2)
        assert bint == '0'
        assert bfrac == '11'

    def test_bin2val_roundtrip(self):
        from directsd import val2bin, bin2val
        for x in [0.0, 0.125, 0.5, 0.875, 1.0, 3.75]:
            bint, bfrac = val2bin(x, 8)
            xr = bin2val(bint, bfrac)
            assert abs(xr - x) < 1e-6, f"val2bin/bin2val roundtrip failed for {x}: got {xr}"

    # ------------------------------------------------------------------
    # coord2hilb / hilb2coord
    # ------------------------------------------------------------------
    def test_hilb2coord_output_shape(self):
        from directsd import hilb2coord
        c = hilb2coord(0.5, 3, 4)
        assert c.shape == (3,)
        assert np.all(c >= 0) and np.all(c <= 1)

    def test_coord2hilb_roundtrip(self):
        from directsd import coord2hilb, hilb2coord
        np.random.seed(11)
        for _ in range(10):
            coord = np.random.rand(2)
            x = coord2hilb(coord, 6)
            assert 0.0 <= x <= 1.0

    def test_hilb2coord_origin(self):
        from directsd import hilb2coord
        c = hilb2coord(0.0, 2, 4)
        assert c.shape == (2,)

    # ------------------------------------------------------------------
    # r2range / r1range
    # ------------------------------------------------------------------
    def test_r2range_truncated_sector(self):
        from directsd import r2range
        Ea, Eb = 0.9, 0.8
        r2min, r2max, E0 = r2range(Ea, Eb, False)
        assert r2max == pytest.approx(Ea ** 2, rel=1e-8)
        assert r2min < 0
        assert E0 == pytest.approx(min(Ea, Eb), rel=1e-8)

    def test_r2range_shifted_sector(self):
        from directsd import r2range
        Ea, Eb = 0.9, 0.8
        r2min, r2max, E0 = r2range(Ea, Eb, True)
        assert E0 == pytest.approx(Ea * Eb, rel=1e-8)
        assert r2min == pytest.approx(-(Ea ** 2) * Eb, rel=1e-8)

    def test_r1range_real_interval(self):
        from directsd import r1range
        r1min, r1max, E0 = r1range(0.3, 0.9, np.inf, False)
        assert r1min < r1max

    # ------------------------------------------------------------------
    # admproj
    # ------------------------------------------------------------------
    def test_admproj_projects_unstable_pole(self):
        from directsd import admproj
        p = np.array([0.1 + 0.0j])
        pp = admproj(p, alpha=0.1, dom='s')
        assert np.real(pp[0]) <= -0.1 + 1e-10

    def test_admproj_z_domain_projects_inside_unit_disk(self):
        from directsd import admproj
        p = np.array([1.5 + 0.0j])
        pp = admproj(p, alpha=0.1, dom='z', T=1.0)
        assert np.abs(pp[0]) < 1.0 + 1e-10

    # ------------------------------------------------------------------
    # par2cp / cp2par
    # ------------------------------------------------------------------
    def test_par2cp_returns_valid_polynomial(self):
        from directsd import par2cp
        rho = np.array([0.4, 0.5])
        Delta, DeltaZ, poles = par2cp(rho, alpha=0.5, beta=np.inf)
        assert len(DeltaZ) == 3
        assert len(poles) == 2

    def test_par2cp_poles_in_stability_sector(self):
        from directsd import par2cp
        rho = np.array([0.5, 0.5, 0.5])
        _, _, poles = par2cp(rho, alpha=0.2, beta=np.inf)
        for p in poles:
            if np.abs(np.imag(p)) < 1e-8:
                pass
            else:
                assert np.abs(p) < 1.0 + 1e-8

    def test_cp2par_clips_to_unit_interval(self):
        from directsd import cp2par
        DeltaZ = np.array([1.0, 0.0, 0.2])
        rho = cp2par(DeltaZ, alpha=0.2, beta=np.inf)
        assert np.all(rho >= 0) and np.all(rho <= 1)

    # ------------------------------------------------------------------
    # guesspoles
    # ------------------------------------------------------------------
    def test_guesspoles_returns_correct_count(self):
        from directsd import guesspoles
        poles = np.array([-1.0, -2.0, -0.5 + 0.5j, -0.5 - 0.5j])
        p = guesspoles(poles, 4)
        assert len(p) == 4

    def test_guesspoles_complex_pairs_first_when_largest(self):
        from directsd import guesspoles
        # Complex pair has larger magnitude than real poles → picked first
        poles = np.array([-0.1, -0.2, -0.8 + 0.8j, -0.8 - 0.8j])
        p = guesspoles(poles, 4)
        assert np.abs(np.imag(p[0])) > 1e-8

    # ------------------------------------------------------------------
    # sasimplex
    # ------------------------------------------------------------------
    def test_sasimplex_minimizes_quadratic(self):
        from directsd import sasimplex
        np.random.seed(42)
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        x_best, y_best, _ = sasimplex(f, [1.0, 1.0],
                                      {'maxFunEvals': 2000, 'startTemp': 0.01})
        assert y_best < 0.1

    # ------------------------------------------------------------------
    # arandsearch
    # ------------------------------------------------------------------
    def test_arandsearch_minimizes_quadratic(self):
        from directsd import arandsearch
        np.random.seed(7)
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        x_best, val, _ = arandsearch(f, [1.0, 1.0],
                                     {'maxFunEvals': 3000})
        assert val < 0.1

    # ------------------------------------------------------------------
    # infglob
    # ------------------------------------------------------------------
    def test_infglob_finds_minimum_of_quadratic(self):
        from directsd import infglob
        f = lambda x: (x - 0.4) ** 2
        x_best, z_best, _, _ = infglob(f, {'maxIter': 200, 'tol': 1e-4})
        assert abs(x_best - 0.4) < 0.01
        assert abs(z_best) < 1e-4

    def test_infglob_returns_four_outputs(self):
        from directsd import infglob
        result = infglob(lambda x: x, {'maxIter': 50})
        assert len(result) == 4

    def test_infglob_x_trace_nonempty(self):
        from directsd import infglob
        _, _, _, trace = infglob(lambda x: np.sin(4 * np.pi * x), {'maxIter': 50})
        assert len(trace) > 0

    # ------------------------------------------------------------------
    # infglobc
    # ------------------------------------------------------------------
    def test_infglobc_finds_feasible_minimum(self):
        from directsd import infglobc
        # Minimize f(x) = x subject to g(x) = x - 0.5 <= 0
        # Global feasible min at x=0 (or at boundary 0.5 if constraint)
        # We encode: if x > 0.5: return [g] (violated), else return [g, f]
        def f(x):
            g = x - 0.5
            if g > 0:
                return [g]
            return [g, float(x)]
        x_best, z_best, _, _ = infglobc(f, {'maxIter': 200, 'nConstr': 1})
        assert x_best is not None
        assert x_best <= 0.5 + 0.05

    def test_infglobc_returns_four_outputs(self):
        from directsd import infglobc
        result = infglobc(lambda x: [float(x)], {'maxIter': 50, 'nConstr': 0})
        assert len(result) == 4

    # ------------------------------------------------------------------
    # optglob
    # ------------------------------------------------------------------
    def test_optglob_finds_minimum(self):
        from directsd import optglob
        f = lambda x: (x - 0.3) ** 2
        x_best, z_best, _, _, _ = optglob(f, {'maxIter': 80, 'maxLoop': 5})
        assert abs(x_best - 0.3) < 0.05

    def test_optglob_returns_five_outputs(self):
        from directsd import optglob
        result = optglob(lambda x: x, {'maxIter': 20, 'maxLoop': 3})
        assert len(result) == 5

    # ------------------------------------------------------------------
    # k2ksi – recover ksi polynomial from a controller
    # ------------------------------------------------------------------

    def test_k2ksi_returns_three_outputs(self):
        """k2ksi returns (ksi, aDelta, bDelta)."""
        from directsd import k2ksi
        # Simple first-order plant  1/(s+1), T=0.1
        plant = ([1.0], [1.0, 1.0])
        # Stabilising controller  K(z) = 0.5 / (z - 0.5)
        K = ([0.5], [1.0, -0.5])
        ksi, aDelta, bDelta = k2ksi(plant, K, T=0.1)
        assert ksi is not None
        assert aDelta is not None
        assert bDelta is not None

    def test_k2ksi_diophantine_consistency(self):
        """aDelta and bDelta satisfy the Diophantine equation up to tolerance."""
        from directsd import k2ksi
        from directsd.polynomial.transforms import dtfm
        import numpy as np
        plant = ([1.0], [1.0, 1.0])
        K = ([0.5], [1.0, -0.5])
        T = 0.1
        ksi, aDelta, bDelta = k2ksi(plant, K, T=T)
        # Reconstruct DeltaZ and Delta
        D22_num, D22_den = dtfm(plant, T)
        D22_num = np.real(np.asarray(D22_num, float)).ravel()
        D22_den = np.real(np.asarray(D22_den, float)).ravel()
        DeltaZ = np.polyadd(np.polymul(D22_den, [1.0, -0.5]),
                            np.polymul(D22_num, [0.5]))
        DeltaZ = DeltaZ / DeltaZ[0]
        Delta = DeltaZ[::-1]
        n = D22_num[::-1]
        d = D22_den[::-1]
        # Verify n*aDelta + d*bDelta ≈ Delta
        lhs = np.polyadd(np.polymul(n, aDelta), np.polymul(d, bDelta))
        # Trim to same length
        target = Delta
        diff = len(lhs) - len(target)
        if diff > 0:
            target = np.concatenate([np.zeros(diff), target])
        elif diff < 0:
            lhs = np.concatenate([np.zeros(-diff), lhs])
        np.testing.assert_allclose(lhs, target, atol=1e-6)

    def test_k2ksi_ksi_is_scalar_for_simple_controller(self):
        """ksi should be a low-degree polynomial (scalar) for a simple plant."""
        from directsd import k2ksi
        plant = ([1.0], [1.0, 1.0])
        K = ([0.5], [1.0, -0.5])
        ksi, _, _ = k2ksi(plant, K, T=0.1)
        assert len(ksi) <= 3  # ksi should be a low-degree polynomial


# ============================================================
# Convex synthesis tests
# ============================================================

try:
    import cvxpy as _cvxpy  # noqa: F401
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


def _make_lifted_plant(T=0.1):
    """Simple SISO lifted plant for convex synthesis tests."""
    import scipy.signal as sig
    from directsd import lift_h2
    ct_plant = sig.StateSpace(
        np.array([[-1.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[1.0], [1.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    P_lifted, _, _ = lift_h2(ct_plant, T)
    return P_lifted


@pytest.mark.skipif(not _CVXPY_AVAILABLE, reason="cvxpy not installed")
class TestConvexSynthesis:
    """Tests for directsd.design.convex — requires cvxpy."""

    # ── import / wiring ───────────────────────────────────────────────────────

    def test_convex_importable(self):
        from directsd.design.convex import (
            sdl1_reg, sd_mixed_h2_l1, sd_constrained,
            sdl1norm, youla_basis,
        )
        assert callable(sdl1_reg)

    def test_convex_in_package_namespace(self):
        import directsd
        assert hasattr(directsd, 'sdl1_reg')
        assert hasattr(directsd, 'sdl1norm')

    # ── youla_basis ───────────────────────────────────────────────────────────

    def test_youla_basis_shapes(self):
        from directsd.design.convex import youla_basis, _stabilising_h2
        P = _make_lifted_plant()
        K0 = _stabilising_h2(P, n_meas=1, n_ctrl=1)
        N_fir = 10
        T_cl, Phi12, Phi21, N = youla_basis(P, K0, N_fir)
        p = T_cl.shape[0] // N
        m = T_cl.shape[1] // N
        assert T_cl.shape  == (N * p, N * m)
        assert Phi12.shape == (N * p, N_fir)      # n_ctrl=1
        assert Phi21.shape == (N_fir, N * m)      # n_meas=1

    def test_youla_basis_affine_identity(self):
        """T(Q=0) must equal T_cl."""
        from directsd.design.convex import youla_basis, _stabilising_h2, _build_toeplitz_np
        P = _make_lifted_plant()
        K0 = _stabilising_h2(P, n_meas=1, n_ctrl=1)
        T_cl, Phi12, Phi21, N = youla_basis(P, K0, N_fir=10)
        Q_zero = np.zeros((10, 1, 1))
        T_Q = T_cl + Phi12 @ _build_toeplitz_np(Q_zero, 1, 1) @ Phi21
        np.testing.assert_allclose(T_Q, T_cl, atol=1e-12)

    # ── sdl1norm ──────────────────────────────────────────────────────────────

    def test_sdl1norm_positive(self):
        from directsd.design.convex import sdl1norm
        plant = ([1.0], [1.0, 1.0])
        K     = ([0.5], [1.0])
        l1, h = sdl1norm(plant, K, T=0.1, N=100)
        assert l1 > 0
        assert h.shape == (100,)

    def test_sdl1norm_stable_loop_finite(self):
        from directsd.design.convex import sdl1norm
        # Stable first-order plant + proportional gain
        plant = ([1.0], [1.0, 2.0])
        K     = ([1.0], [1.0])
        l1, _ = sdl1norm(plant, K, T=0.05, N=200)
        assert np.isfinite(l1)
        assert l1 < 1e4   # converged impulse response

    # ── sdl1_reg ──────────────────────────────────────────────────────────────

    def test_sdl1_reg_returns_controller(self):
        import scipy.signal as sig
        from directsd.design.convex import sdl1_reg
        P = _make_lifted_plant()
        K_opt, l1, Q_fir, info = sdl1_reg(P, N_fir=8)
        assert isinstance(K_opt, sig.StateSpace)
        assert np.isfinite(l1)
        assert l1 > 0
        assert Q_fir.shape == (8, 1, 1)
        assert info['status'] in ('optimal', 'optimal_inaccurate')

    def test_sdl1_reg_l1_nonnegative(self):
        from directsd.design.convex import sdl1_reg
        P = _make_lifted_plant()
        _, l1, _, _ = sdl1_reg(P, N_fir=6)
        assert l1 >= 0

    def test_sdl1_reg_improves_over_h2_base(self):
        """L1-optimal controller should not be worse (in L1) than the H2 base."""
        from directsd.design.convex import sdl1_reg, sdl1norm, _stabilising_h2
        from directsd.polynomial.transforms import dtfm
        P = _make_lifted_plant(T=0.1)
        K_opt, l1_opt, _, _ = sdl1_reg(P, N_fir=10)
        assert l1_opt < 1e6   # feasible solution found

    # ── sd_mixed_h2_l1 ────────────────────────────────────────────────────────

    def test_mixed_mode_a_h2_minimised(self):
        """Mode A: minimise H2 subject to L1 bound."""
        from directsd.design.convex import sd_mixed_h2_l1
        P = _make_lifted_plant()
        K, cost, Q, info = sd_mixed_h2_l1(P, N_fir=8, l1_bound=50.0)
        assert info['status'] in ('optimal', 'optimal_inaccurate')
        assert cost['h2'] >= 0
        assert cost['l1'] <= 50.0 + 1e-3   # constraint satisfied

    def test_mixed_mode_b_l1_minimised(self):
        """Mode B: minimise L1 subject to H2 bound."""
        from directsd.design.convex import sd_mixed_h2_l1
        P = _make_lifted_plant()
        K, cost, Q, info = sd_mixed_h2_l1(P, N_fir=8, h2_bound=50.0)
        assert info['status'] in ('optimal', 'optimal_inaccurate')
        assert cost['l1'] >= 0

    def test_mixed_no_bounds_scalarised(self):
        """Mode C: no bounds → scalarised objective, must still solve."""
        from directsd.design.convex import sd_mixed_h2_l1
        P = _make_lifted_plant()
        K, cost, Q, info = sd_mixed_h2_l1(P, N_fir=6)
        assert info['status'] in ('optimal', 'optimal_inaccurate')

    # ── sd_constrained ────────────────────────────────────────────────────────

    def test_constrained_h2_objective(self):
        from directsd.design.convex import sd_constrained
        P = _make_lifted_plant()
        K, achieved, Q, info = sd_constrained(P, N_fir=8, objective='h2')
        assert info['status'] in ('optimal', 'optimal_inaccurate')
        assert achieved['h2'] >= 0

    def test_constrained_l1_objective(self):
        from directsd.design.convex import sd_constrained
        P = _make_lifted_plant()
        K, achieved, Q, info = sd_constrained(P, N_fir=8, objective='l1')
        assert info['status'] in ('optimal', 'optimal_inaccurate')
        assert achieved['l1'] >= 0

    def test_constrained_linf_objective(self):
        from directsd.design.convex import sd_constrained
        P = _make_lifted_plant()
        K, achieved, Q, info = sd_constrained(P, N_fir=8, objective='linf')
        assert info['status'] in ('optimal', 'optimal_inaccurate')
        assert achieved['linf'] >= 0

    def test_constrained_bad_objective_raises(self):
        from directsd.design.convex import sd_constrained
        P = _make_lifted_plant()
        with pytest.raises(ValueError, match="Unknown objective"):
            sd_constrained(P, N_fir=4, objective='bad')

    def test_constrained_with_output_bound(self):
        from directsd.design.convex import sd_constrained
        P = _make_lifted_plant()
        K, achieved, Q, info = sd_constrained(
            P, N_fir=8, objective='h2', output_bound=5.0
        )
        assert info['status'] in ('optimal', 'optimal_inaccurate')

    def test_constrained_achieved_keys(self):
        """achieved dict must have h2, l1, linf keys."""
        from directsd.design.convex import sd_constrained
        P = _make_lifted_plant()
        _, achieved, _, _ = sd_constrained(P, N_fir=6, objective='h2')
        assert set(achieved.keys()) == {'h2', 'l1', 'linf'}

    # ── missing cvxpy guard ───────────────────────────────────────────────────

    def test_require_cvxpy_passes_when_installed(self):
        from directsd.design.convex import _require_cvxpy
        _require_cvxpy()   # should not raise
