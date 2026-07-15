"""
Minimal-realization (pole/zero and state cancellation) utilities for DirectSD.

Port of MATLAB's ``@zpk/minreal.m`` and ``@zpk/minreals.m`` — both revised by
K. Polyakov from the original MathWorks Control System Toolbox ``minreal``.
MATLAB exposes these as ``minreal(sys, tol)`` / ``minreals(sys, tol)``, whose
behavior dispatches on the class of ``sys`` for the former (Kalman
decomposition for state-space models, root cancellation for zpk/transfer-
function models). Python has no such class-based dispatch for plain ndarrays
and scipy objects, so :class:`Minreal` groups all three operations under one
namespace instead of leaving them as differently-named free functions:

    Minreal.ss(...)            — state-space (Kalman decomposition)
    Minreal.tf(...)            — zpk/tf, general (conjugate-pair symmetry)
    Minreal.tf_symmetric(...)  — zpk/tf, symmetric spectral density
                                  (reciprocal-pair symmetry; port of minreals.m)
"""

import numpy as np


class Minreal:
    """Namespace for DirectSD's minimal-realization operations."""

    @staticmethod
    def _ctrl_staircase(A, B_ctrl, B_full, C_full, tol):
        """
        Reduce to the controllable part from B_ctrl using the Rosenbrock staircase.

        First step uses B_ctrl to find directly-reachable states; subsequent steps
        use the coupling block A21 (lower-left of partitioned A) to find additionally
        reachable states via the system dynamics.

        B_full and C_full are transformed consistently.

        Returns (A_new, B_ctrl_new, B_full_new, C_full_new, nc).
        """
        import scipy.linalg as la
        n, m = A.shape[0], B_ctrl.shape[1]
        if n == 0:
            return A, B_ctrl, B_full, C_full, 0

        # Absolute threshold: based on scale of original B and A to avoid dividing by ~0
        sys_scale = max(np.linalg.norm(B_ctrl, 'fro'), np.linalg.norm(A, 'fro'), 1e-300)
        abs_tol = tol * sys_scale

        nc = 0
        nc_prev = 0
        Aw = A.copy()
        Bctrl = B_ctrl.copy()
        Bfull = B_full.copy()
        Cfull = C_full.copy()

        for step in range(n):
            if step == 0:
                Msub = Bctrl      # first step: use B directly
            else:
                Msub = Aw[nc:, nc_prev:nc]   # subsequent steps: coupling block A21

            nr = Msub.shape[0]
            mc = Msub.shape[1]
            if nr == 0 or mc == 0:
                break

            # SVD guarantees singular values are sorted descending, so the first rk
            # left singular vectors always span the full column space of Msub —
            # no zero can appear between two nonzero values as happens with non-pivoting QR.
            U2, sv, _ = la.svd(Msub, full_matrices=True)   # U2: (nr × nr) orthogonal
            rk = int(np.sum(sv > abs_tol))
            # Sign-normalise for platform determinism: make largest-magnitude element positive.
            for k in range(rk):
                idx = np.argmax(np.abs(U2[:, k]))
                if U2[idx, k] < 0:
                    U2[:, k] = -U2[:, k]
            Q2 = U2
            if rk == 0:
                break

            Qfull = np.eye(n)
            Qfull[nc:, nc:] = Q2.T
            Aw    = Qfull @ Aw @ Qfull.T
            Bctrl = Qfull @ Bctrl
            Bfull = Qfull @ Bfull
            Cfull = Cfull @ Qfull.T

            nc_prev = nc
            nc = nc + rk
            if nc >= n:
                break

        return Aw, Bctrl, Bfull, Cfull, nc

    @staticmethod
    def ss(A, B=None, C=None, D=None, tol=None):
        """
        Minimal realization of a state-space system via Rosenbrock-staircase Kalman
        decomposition (controllability then observability, each via a sequence of
        SVDs on successive coupling blocks rather than one large Krylov matrix).

        Dispatches on its own input instead of requiring a differently-named
        function per calling convention:

            Minreal.ss(A, B, C, D)   -> (A_min, B_min, C_min, D)   raw ndarrays
            Minreal.ss(sys_ss)       -> StateSpace                 scipy object

        Detected by whether `B` is given: a bare `StateSpace`-like object (anything
        exposing `.A/.B/.C/.D`) is passed as the sole positional argument, and the
        reduced result comes back in the same form it went in.

        The staircase algorithm (`_ctrl_staircase`) is preferred over a one-shot
        Krylov-chain SVD (`[B, AB, A^2 B, ...]` built as one matrix and SVD'd once):
        matrix powers amplify ill-conditioning as `A` develops close/repeated
        eigenvalues, where the staircase's one-layer-at-a-time reduction stays
        well-conditioned.

        Port of MATLAB regular.m's very first step (``sys = minreal(ss(sys),...)``),
        which the Python `h2reg`/`hinfreg` were missing entirely. Removes
        uncontrollable and unobservable state directions before any Riccati-based
        synthesis is attempted.

        Uses controllability/observability *matrix rank* (not the Gramian/Lyapunov
        approach `scipy` favors elsewhere) specifically because it works when `A`
        has eigenvalues on the imaginary axis / at z=1 (marginally stable modes,
        e.g. plant integrators) — Lyapunov-equation-based Gramians require `A` to
        be Hurwitz/Schur and would raise or return garbage there.
        """
        as_ss = B is None
        if as_ss:
            sys_ss = A
            A, B, C, D = sys_ss.A, sys_ss.B, sys_ss.C, sys_ss.D

        n = A.shape[0]
        if n == 0:
            return sys_ss if as_ss else (A, B, C, D)
        if tol is None:
            tol = n * np.finfo(float).eps * max(np.linalg.norm(A), 1.0)

        m, p = B.shape[1], C.shape[0]

        def _empty():
            return np.zeros((0, 0)), np.zeros((0, m)), np.zeros((p, 0)), D

        # Step 1: controllability from B. B_full = C.T (kept for C recovery through
        # the same orthogonal transform); C_full = empty (nothing else to carry).
        Aw, Bw, Cw_t, _, nc = Minreal._ctrl_staircase(A, B, C.T, np.zeros((0, n)), tol)
        if nc == 0:
            A_min, B_min, C_min, D = _empty()
        else:
            if nc < n:
                Aw   = Aw[:nc, :nc]
                Bw   = Bw[:nc, :]
                Cw_t = Cw_t[:nc, :]

            # Step 2: observability from C (dual controllability on A.T).
            # B_full dummy (nc×0); C_full = Bw.T (m×nc, nc cols matching A.T's dim).
            Ao_t, Co_t_red, _, Bw_t_cfull, no = Minreal._ctrl_staircase(
                Aw.T, Cw_t, np.zeros((nc, 0)), Bw.T, tol
            )
            if no == 0:
                A_min, B_min, C_min, D = _empty()
            else:
                if no < nc:
                    Ao_t       = Ao_t[:no, :no]
                    Co_t_red   = Co_t_red[:no, :]
                    Bw_t_cfull = Bw_t_cfull[:, :no]
                A_min, B_min, C_min = Ao_t.T, Bw_t_cfull.T, Co_t_red.T

        if as_ss:
            import scipy.signal as sig
            return sig.StateSpace(A_min, B_min, C_min, D)
        return A_min, B_min, C_min, D

    @staticmethod
    def _reducezp(z, p, tol, im_tol=1e-9):
        """
        Cancel matching pairs of zeros (z) and poles (p) within relative tolerance
        `tol`, preserving complex-conjugate symmetry (assumes real coefficients —
        i.e. every non-real root has a conjugate partner in the same array).

        Port of MATLAB `@zpk/minreal.m`'s private `REDUCEZP`. Naive nearest-root
        matching deletes whichever pair it matches first — but when a
        complex-conjugate *pair* of zeros can only be matched against a single
        real pole (or vice versa), blindly deleting one side breaks the
        real-coefficient symmetry of the result. MATLAB's approach instead
        *redistributes* the residual mismatch into a nearby surviving root —
        replacing it with a weighted average `(matched + 2*target)/3` — and
        re-queues the leftover half for another cancellation attempt, rather
        than just discarding it.

        Returns
        -------
        zr, pr : ndarray (complex)
            Surviving zeros and poles after cancellation.
        """
        z = np.asarray(z, dtype=complex)
        pr = list(np.asarray(p, dtype=complex))

        def _closest(zm, lst):
            if not lst:
                return None, None
            d = [abs(zm - s) for s in lst]
            imin = int(np.argmin(d))
            return d[imin], imin

        def _pop_closest_to(target, lst):
            d = [abs(target - s) for s in lst]
            imin = int(np.argmin(d))
            return lst.pop(imin)

        # Represent each conjugate pair by its Im>0 member; the Im<0 partner is
        # implicit (reconstructed at the end) and never separately tracked here —
        # matching MATLAB's `z(imag(z)>0,:)` / `z(imag(z)==0,:)` split.
        ccz = [zi for zi in z if zi.imag > im_tol]
        sz = [zi for zi in z if abs(zi.imag) <= im_tol]

        # --- Pass 1: complex-conjugate zero pairs, cancelled first so that any
        # real-pole "downgrade" fallback feeds cleanly into pass 2. ---
        ccz_keep = []
        for zm in ccz:
            dmin, imin = _closest(zm, pr)
            if dmin is not None and dmin < tol * (1.0 + abs(zm)):
                pm = pr[imin]
                if abs(pm.imag) > im_tol:
                    # pm is complex: cancel (zm,pm) AND their conjugates together.
                    pr.pop(imin)
                    _pop_closest_to(np.conj(pm), pr)
                else:
                    # pm is real: only half the pair can cancel against it —
                    # downgrade the pair to one new real zero, re-queued below.
                    sz.append((pm + 2.0 * zm.real) / 3.0)
                    pr.pop(imin)
            else:
                ccz_keep.append(zm)
        ccz = ccz_keep

        # --- Pass 2: simple (real) zeros. ---
        sz_keep = []
        for zm in sz:
            dmin, imin = _closest(zm, pr)
            if dmin is not None and dmin < tol * (1.0 + abs(zm)):
                pm = pr[imin]
                if abs(pm.imag) > im_tol:
                    # pm is complex: cancelling only its conjugate would break
                    # symmetry — replace the conjugate partner's position instead
                    # of deleting it, absorbing the residual mismatch there.
                    pr.pop(imin)
                    d = [abs(np.conj(pm) - s) for s in pr]
                    if d:
                        icjg = int(np.argmin(d))
                        pr[icjg] = (zm + 2.0 * pm.real) / 3.0
                else:
                    pr.pop(imin)
            else:
                sz_keep.append(zm)
        sz = sz_keep

        zr = []
        for zc in ccz:
            zr.append(zc)
            zr.append(np.conj(zc))
        zr.extend(sz)

        return np.array(zr, dtype=complex), np.array(pr, dtype=complex)

    @staticmethod
    def tf(num, den, tol=0.01):
        """
        Cancel common roots in (num, den) — minreal for raw numpy coefficient
        arrays representing a transfer function's numerator/denominator.

        Uses `_reducezp` (port of MATLAB `@zpk/minreal.m`'s `REDUCEZP`) rather
        than naive first-match root deletion.
        """
        # Strip negligible leading coefficients (MATLAB striplz): the gain is
        # taken from num[0]/den[0] below, so a padded leading 0.0 (e.g. from
        # np.polyadd aligning different-degree operands) would otherwise zero
        # the entire numerator (e.g. E from _ztrm_self_adjoint coming back
        # identically 0 for a 2-output P11, gutting every AHinf design that
        # reached _polhinf).
        num = np.atleast_1d(np.asarray(num, float))
        den = np.atleast_1d(np.asarray(den, float))
        _nmax = np.max(np.abs(num)) if num.size else 0.0
        _dmax = np.max(np.abs(den)) if den.size else 0.0
        if _nmax == 0.0:
            return np.array([0.0]), np.array([1.0])
        while len(num) > 1 and abs(num[0]) <= 1e-12 * _nmax:
            num = num[1:]
        while len(den) > 1 and abs(den[0]) <= 1e-12 * _dmax:
            den = den[1:]
        rn = np.roots(num) if len(num) > 1 else np.array([], dtype=complex)
        rd = np.roots(den) if len(den) > 1 else np.array([], dtype=complex)
        zr, pr = Minreal._reducezp(rn, rd, tol)
        sn = float(np.real(num[0])); sd = float(np.real(den[0]))
        nn = np.real(np.poly(zr)).astype(float) * sn if len(zr) > 0 else np.array([sn])
        nd = np.real(np.poly(pr)).astype(float) * sd if len(pr) > 0 else np.array([sd])
        sc = nd[0] if abs(nd[0]) > 1e-30 else 1.0
        return nn / sc, nd / sc

    @staticmethod
    def _remove(r, x, tol=1e-8):
        """
        Remove, from `r`, the nearest-matching entry to each element of `x`
        in turn. Port of MATLAB `@zpk/private/remove.m`.

        When the target `x[i]` is real but its nearest match in `r` is
        complex, every entry of `r` is snapped to its real part before the
        removal — MATLAB's own fix-up (`r(cc) = real(r(cc))` over a full
        index permutation is equivalent to `r = real(r)`), presumably valid
        because by this point in `minreals`' cancellation loop `r` is
        expected to already be real-symmetric and any remaining imaginary
        part is numerical noise to be cleaned up, not genuine structure.
        """
        r = list(np.asarray(r, dtype=complex))
        for xi in x:
            if len(r) < 1:
                raise ValueError("No more elements to remove")
            d = [abs(ri - xi) for ri in r]
            k = int(np.argmin(d))
            if d[k] > tol:
                raise ValueError(f"Unable to find a term to remove: err={d[k]:g}")
            if np.imag(xi) == 0 and np.imag(r[k]) != 0:
                r = [np.real(ri) for ri in r]
            r.pop(k)
        return np.array(r, dtype=complex)

    @staticmethod
    def tf_symmetric(num, den, ftype='d', tol=None):
        """
        Minimal realization of a *symmetric* rational function (a spectral
        density / para-Hermitian quasipolynomial ratio), preserving
        reciprocal-pair symmetry (`z`, `1/z` for `ftype='d'`/`'z'`) or
        negative-pair symmetry (`s`, `-s` for `ftype='s'`) under cancellation.

        Port of MATLAB `@zpk/minreals.m`. Distinct from `Minreal.tf`
        (`@zpk/minreal.m`'s `REDUCEZP`, which only preserves real-coefficient
        conjugate-pair symmetry): here zeros and poles must *each*
        independently already be symmetric (raises if not — "the function is
        not symmetric"), and cancellation removes a pole/zero pair *together
        with each side's own symmetric partner*, so the result stays
        symmetric too.

        Used by MATLAB's `zterm.m` (`Z = E - B*B~/A` construction),
        `sdh2coef.m` (`A = A0*A1*A1'`-type products), `sdhinferr.m`,
        `quaderr.m`, `abe2std.m` — all quadratic-form / spectral-density
        constructions in the H2/H∞ polynomial design pipeline.

        Parameters
        ----------
        num, den : ndarray
            Numerator/denominator coefficients (descending powers).
        ftype : {'d', 's'}
            'd' (default) for discrete-time (reciprocal pairs); 's' for
            continuous-time (negative pairs).
        tol : float, optional
            Relative tolerance; default `sqrt(eps)`.

        Returns
        -------
        num_min, den_min : ndarray
        """
        from directsd.polynomial.poln import _extrpair

        if tol is None:
            tol = np.sqrt(np.finfo(float).eps)

        zz = np.roots(num) if len(num) > 1 else np.array([], dtype=complex)
        pp = np.roots(den) if len(den) > 1 else np.array([], dtype=complex)
        sn = float(np.real(num[0])) if len(num) > 0 else 1.0
        sd = float(np.real(den[0])) if len(den) > 0 else 1.0

        zs, z_rem, z0 = _extrpair(zz, ftype, tol)
        ps, p_rem, p0 = _extrpair(pp, ftype, tol)
        if z_rem.size > 0 or p_rem.size > 0:
            raise ValueError("The function is not symmetric")

        c0 = min(z0, p0)
        z0 -= c0
        p0 -= c0

        if ftype == 's':
            if z0 % 2 != 0 or p0 % 2 != 0:
                raise ValueError("Incorrect zeros or poles at the origin")
        else:
            if len(zs) + z0 != len(ps) + p0:
                raise ValueError("Incorrect zeros or poles at the origin")

        zs_l, ps_l = list(zs), list(ps)
        modified = False
        i = 0
        while i < len(ps_l):
            A = ps_l[i]
            tolA = max(tol * abs(A), tol)
            if len(zs_l) < 1:
                break
            d = [abs(z - A) for z in zs_l]
            if min(d) < tolA:
                modified = True
                targets = (
                    [A, np.conj(A)]
                    if abs(np.imag(A)) > np.finfo(float).eps
                    else [A]
                )
                zs_l = list(Minreal._remove(zs_l, targets, 100 * tolA))
                ps_l = list(Minreal._remove(ps_l, targets, 100 * tolA))
            else:
                i += 1

        if not modified:
            return np.asarray(num, dtype=float), np.asarray(den, dtype=float)

        zs_arr, ps_arr = np.array(zs_l, dtype=complex), np.array(ps_l, dtype=complex)
        if ftype == 's':
            zz_new = np.concatenate([zs_arr, -zs_arr, np.zeros(z0 // 2, dtype=complex)])
            pp_new = np.concatenate([ps_arr, -ps_arr, np.zeros(p0 // 2, dtype=complex)])
        else:
            z_recip = 1.0 / zs_arr if zs_arr.size > 0 else zs_arr
            p_recip = 1.0 / ps_arr if ps_arr.size > 0 else ps_arr
            zz_new = np.concatenate([zs_arr, z_recip, np.zeros(z0, dtype=complex)])
            pp_new = np.concatenate([ps_arr, p_recip, np.zeros(p0, dtype=complex)])

        num_min = (np.real(np.poly(zz_new)).astype(float) * sn
                   if zz_new.size > 0 else np.array([sn]))
        den_min = (np.real(np.poly(pp_new)).astype(float) * sd
                   if pp_new.size > 0 else np.array([sd]))
        return num_min, den_min
