from warnings import warn

try:
    from .bin.gp import logL as GP_logL
except ImportError: # pragma: no cover
    warn("Did not find compiled code for gp.logL, falling back to python")
    from .src.gp_py import logL as GP_logL

from .src.gp_py import msd2C_ss0 as GP_msd2C_ss0
from .src.gp_py import msd2C_ss1 as GP_msd2C_ss1
from .src.gp_py import BadCovarianceError as GP_BadCovarianceError
