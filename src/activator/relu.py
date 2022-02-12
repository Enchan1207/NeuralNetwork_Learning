#
# ReLU関数
#

from typing import Optional

import numpy as np
from numpy import ndarray

from . import Activator


class Relu(Activator):
    """ReLU関数
    """

    def __init__(self) -> None:
        self._mask: Optional[ndarray] = None

    def forward(self, x: ndarray) -> ndarray:
        mask = (x <= 0)
        y = x.copy()
        y[mask] = 0

        self._mask = mask
        return y

    def backward(self, dout: ndarray) -> ndarray:
        if self._mask is None:
            raise ValueError("Please call forward() at least once before call backward().")

        result = dout.copy()
        result[self._mask] = 0
        return result
