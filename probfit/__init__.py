"""
probfit - Cost function builder. For fitting distributions.

* Code: https://github.com/scikit-hep/probfit
* Docs: http://probfit.readthedocs.io
"""


from ._libstat import integrate1d
from .costfunc import BinnedChi2, BinnedLH, Chi2Regression, SimultaneousFit, UnbinnedLH
from .decorator import *
from .functor import AddPdf, AddPdfNorm, BlindFunc, Convolve, Extended, Normalized
from .funcutil import *
from .oneshot import *
from .pdf import (
    HistogramPdf,
    Polynomial,
    argus,
    cauchy,
    cruijff,
    crystalball,
    doublecrystalball,
    doublegaussian,
    exponential,
    gaussian,
    johnsonSU,
    linear,
    novosibirsk,
    poly2,
    poly3,
    rtv_breitwigner,
    ugaussian,
)
from .plotting import *
from .statutil import *
from .toy import gen_toy, gen_toyn
from .util import *
from .version import __version__

__all__ = [
    "AddPdfNorm",
    "AddPdf",
    "BinnedChi2",
    "BinnedLH",
    "BlindFunc",
    "Chi2Regression",
    "Convolve",
    "Extended",
    "Normalized",
    "Polynomial",
    "HistogramPdf",
    "UnbinnedLH",
    "argus",
    "cruijff",
    "cauchy",
    "rtv_breitwigner",
    "crystalball",
    "doublecrystalball",
    "describe",
    "doublegaussian",
    "johnsonSU",
    "draw_compare",
    "draw_compare_hist",
    "draw_pdf",
    "draw_pdf_with_edges",
    "extended",
    "fwhm_f",
    "gaussian",
    "gen_toy",
    "gen_toyn",
    "integrate1d",
    "linear",
    "normalized",
    "novosibirsk",
    "poly2",
    "poly3",
    "SimultaneousFit",
    "try_binlh",
    "try_chi2",
    "try_uml",
    "ugaussian",
    "exponential",
    "__version__",
]
