"""
DirectSD - Direct Sampled-Data Control Systems Toolbox
Python port of the MATLAB DirectSD Toolbox v3.0 by Konstantin Polyakov.

Modules:
    polynomial  - Polynomial class and operations
    linalg      - Linear algebra routines
    design      - Polynomial/H2/Hinf controller design
    analysis    - Sampled-data system analysis (norms, margins)
    sspace      - State-space design methods
    glopt       - Global optimization routines
"""

__version__ = "0.2.0"
__author__ = "Python port. Original MATLAB toolbox by Konstantin Polyakov (1995-2006)."

# Generalized plant
from directsd.sspace.plant import GeneralizedPlant

# Polynomial objects and operations
from directsd.polynomial.poln import (
    Poln,
    s_var, z_var, q_var, p_var, d_var,
)
from directsd.polynomial.operations import (
    compat, deg, gcd, coprime, triple,
    factor, striplz, recip, vec, delzero,
)
from directsd.polynomial.diophantine import dioph, dioph2
from directsd.polynomial.spectral import sfactor, sfactfft
from directsd.polynomial.transforms import ztrm, dtfm

# Linear algebra
from directsd.linalg.matrices import toep, hank, givens, house, lyap, dlyap
from directsd.linalg.linsys import linsys

# Analysis
from directsd.analysis.norms import (
    sdh2norm, sdhinorm, dinfnorm, dahinorm, sdfreq,
)
from directsd.analysis.charpol import charpol, sdmargin
from directsd.analysis.errors import quaderr, sdl2err, sd2doferr

# Design
from directsd.design.polynomial import (
    ch2, sdh2, sdl2,
    sdahinf, sdahinorm,
    sdh2hinf,
    sdtrhinf, sdtrhinferr,
    modsdh2, modsdl2,
    dhinf, sd2dof, split2dof, polquad, polhinf, whquad, ssquad, psigain, polopth2, dtfm2,
)
from directsd.sspace.design import (
    h2reg, hinfreg, sdh2reg, sdhinfreg, sdfast, separss,
    sdh2simple, sdgh2mod, sdnorm, sdsim,
)

# Global optimization
from directsd.glopt.optimize import (
    neldermead, simanneal, randsearch, crandsearch,
    sector, banana,
)

__all__ = [
    # Generalized plant
    'GeneralizedPlant',
    # Polynomial
    'Poln', 's_var', 'z_var', 'q_var', 'p_var', 'd_var',
    'compat', 'deg', 'gcd', 'coprime', 'triple',
    'factor', 'striplz', 'recip', 'vec', 'delzero', 'zpk',
    'dioph', 'dioph2',
    'sfactor', 'sfactfft',
    'ztrm', 'dtfm',
    # LinAlg
    'toep', 'hank', 'givens', 'house', 'lyap', 'dlyap',
    'linsys',
    # Analysis
    'charpol', 'sdmargin',
    'sdh2norm', 'sdhinorm', 'dinfnorm', 'dahinorm',
    'quaderr', 'sdl2err', 'sd2doferr',
    # Design
    'ch2', 'sdh2', 'sdl2',
    'sdahinf', 'sdahinorm',
    'sdh2hinf',
    'sdtrhinf', 'sdtrhinferr',
    'modsdh2', 'modsdl2',
    'dhinf', 'sd2dof', 'split2dof', 'polquad', 'polhinf', 'whquad', 'ssquad', 'psigain', 'polopth2', 'dtfm2',
    'h2reg', 'hinfreg', 'sdh2reg', 'sdhinfreg', 'sdfast', 'separss',
    'sdh2simple', 'sdgh2mod', 'sdnorm', 'sdsim',
    'sdfreq', 'diophsys', 'diophsys2',
    # Convex synthesis (available when cvxpy is installed)
    'sdl1_reg', 'sd_mixed_h2_l1', 'sd_constrained',
    'sdl1norm', 'youla_basis',
    # Optimization
    'neldermead', 'simanneal', 'randsearch', 'crandsearch',
    'sector', 'banana',
    # Advanced glopt
    'updateopt', 'uniproj', 'u2range',
    'randbeta', 'randgamma', 'sa_testfun',
    'val2bin', 'bin2val', 'coord2hilb', 'hilb2coord',
    'r2range', 'r1range', 'admproj', 'par2cp', 'cp2par', 'guesspoles',
    'k2ksi', 'go_par2k', 'f_sdh2p', 'f_sdl2p', 'go_sdh2p', 'go_sdl2p',
    'sasimplex', 'arandsearch',
    'infglob', 'infglobc', 'optglob', 'optglobc',
]

# Utilities
from directsd.polynomial.utils import (
    bilintr, improper, tf2nd, separtf, sumzpk, bilinss, zpk,
)
from directsd.polynomial.diophantine import diophsys, diophsys2

# Lifting (first-class module)
from directsd.design.lifting import lift_h2, lift_l2, compute_gamma

# Additional norm functions
from directsd.analysis.norms import h2norm_ct, hinfnorm_ct

# Enhanced optimization
from directsd.glopt.optimize import dual_annealing

# Convex synthesis (optional — requires cvxpy)
try:
    from directsd.design.convex import (
        sdl1_reg, sd_mixed_h2_l1, sd_constrained,
        sdl1norm, youla_basis,
    )
    _CONVEX_AVAILABLE = True
except ImportError:
    _CONVEX_AVAILABLE = False

# Advanced global optimization variants (dsdglopt port)
from directsd.glopt.advanced import (
    updateopt, uniproj, u2range,
    randbeta, randgamma, sa_testfun,
    val2bin, bin2val, coord2hilb, hilb2coord,
    r2range, r1range, admproj, par2cp, cp2par, guesspoles,
    k2ksi, go_par2k, f_sdh2p, f_sdl2p, go_sdh2p, go_sdl2p,
    sasimplex, arandsearch,
    infglob, infglobc, optglob, optglobc,
)
