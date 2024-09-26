__project__ = "pyCellPhenoX"
# These version placeholders are updated during build by poetry-dynamic-versioning
__version__ = "0.1.2"
__version_tuple__ = (0, 1, 2)
__license__ = "MIT License"
__author__ = "pyCellPhenoX Contributors"

# Import key classes and functions
from .CellPhenoX import CellPhenoX
from .marker_discovery import marker_discovery
from .nonnegativeMatrixFactorization import nonnegativeMatrixFactorization
from .preprocessing import preprocessing
from .principalComponentAnalysis import principalComponentAnalysis  # Fixed spelling
