#
# Affineコンポーネント
#

from typing import Optional, Tuple

import numpy as np
from numpy import ndarray


class Affine:
    """Affineコンポーネント
    """

    def __init__(self, w: ndarray, b: ndarray) -> None:
        """重みとバイアスを設定してコンポーネントを生成します.

        Args:
            w (ndarray): 重み
            b (ndarray): バイアス
        """

        self.w: ndarray = w
        self.b: ndarray = b

        self._x: Optional[ndarray] = None

    def forward(self, x: ndarray) -> ndarray:
        """順伝播を計算します.

        Args:
            x (ndarray): 入力

        Returns:
            ndarray: 出力

        Raises:
            ValueError: 不正な引数が入力された場合.
        """

        self._x = x

        return np.dot(x, self.w) + self.b

    def backward(self, dout: ndarray) -> ndarray:
        """逆伝播を計算します.

        Args:
            dout (ndarray): 出力の変化量

        Returns:
            ndarray: 入力の変化量

        Raises:
            ValueError: 初期化後、一度もforwardを呼び出さずに呼び出した場合.
        """

        if self._x is None:
            raise ValueError("Please call forward() at least once before call backward().")

        dx = np.dot(dout, self.w.T)
        dw = np.dot(self._x.T, dout)
        db = np.sum(dout, axis=0)

        return dx
