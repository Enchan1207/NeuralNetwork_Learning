#
# Sigmoid関数
#

from typing import Optional

import numpy as np
from numpy import ndarray

from . import Activator


class Sigmoid(Activator):

    """Sigmoid関数
    """

    def __init__(self) -> None:
        self._y: Optional[ndarray] = None

    def forward(self, x: ndarray) -> ndarray:
        y = 1.0 / (1.0 + np.exp(-x))
        self._y = y
        return y

    def backward(self, dout: ndarray) -> ndarray:
        if self._y is None:
            raise ValueError("Please call forward() at least once before call backward().")

        dx: ndarray = dout * self._y * (1.0 - self._y)
        return dx
