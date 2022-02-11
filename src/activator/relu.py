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
        self._y: Optional[ndarray] = None

    def forward(self, x: ndarray) -> ndarray:
        y: ndarray = np.vectorize(lambda n: 1 if n > 0 else 0)(x > 0)

        self._y = y
        return y

    def backward(self, dout: ndarray) -> ndarray:
        if self._y is None:
            raise ValueError("Please call forward() at least once before call backward().")

        result: ndarray = self._y * dout
        return result
