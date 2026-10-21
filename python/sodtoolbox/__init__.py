"""SOD Toolbox: smooth orthogonal decomposition for modal identification and reduced-order modelling.

Modules
    core       configuration SOD, state SOD, lagged and scaled variants, boundary-corrected fit (csod)
    recognize  mode recognition: DCT spectra, spectral consistency, circularity pairing, banded rescue
    damping    plane-wise damping fit and the damping identity
    tracking   windowed SOD for slowly varying (trend) records
    rom        bases (LNM, MTA, POD, SOD, complex modes), Galerkin and energy-Galerkin ROMs, exact simulation
    residual   linear-core-plus-residual bases for nonlinear ROMs
    smd        smooth mode decomposition (spatial dual of SOD) and spatial derivatives of full-field data
    systems    small benchmark chains (linear and Duffing) for examples and tests
    plotting   LaTeX figure style with the MATLAB or NLD palette
"""
from .core import *          # noqa: F401,F403
from .recognize import *     # noqa: F401,F403
from .damping import *       # noqa: F401,F403
from .tracking import *      # noqa: F401,F403
from .rom import *           # noqa: F401,F403
from .residual import *      # noqa: F401,F403
from .smd import *           # noqa: F401,F403
from . import systems, plotting  # noqa: F401

__version__ = '0.3.0'
