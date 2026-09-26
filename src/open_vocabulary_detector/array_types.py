"""
Shared NumPy array type aliases for detection outputs.
"""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
