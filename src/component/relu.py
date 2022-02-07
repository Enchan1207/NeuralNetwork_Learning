#
# ReLUコンポーネント
#

from typing import Optional
import numpy as np
from numpy import ndarray


class Relu:
    """ReLU関数コンポーネント
    """

    def __init__(self) -> None:
        self._y: Optional[ndarray] = None

    def forward(self, x: ndarray) -> ndarray:
        """順伝播を計算します.

        Args:
            x (ndarray): 入力

        Returns:
            ndarray: 出力
        """

        y = np.vectorize(lambda n: 1 if n > 0 else 0)(x > 0)

        self._y = y
        return y

    def backward(self, dout: ndarray) -> ndarray:
        """逆伝播を計算します.

        Args:
            dout (ndarray): 出力の変化量

        Returns:
            ndarray: 入力の変化量

        Raises:
            ValueError: 初期化後、一度もforwardを呼び出さずに呼び出した場合.
        """
        if self._y is None:
            raise ValueError("Please call forward() at least once before call backward().")

        return self._y * dout
