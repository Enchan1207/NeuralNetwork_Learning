#
# 加算コンポーネント
#
from typing import Tuple

from numpy import ndarray


class Adder:

    """加算コンポーネント
    """

    def forward(self, x: ndarray, y: ndarray) -> ndarray:
        """順伝播を計算します.

        Args:
            x (ndarray):入力1
            y (ndarray): 入力2

        Returns:
            ndarray: 順伝播演算結果
        """

        return x + y

    def backward(self, dout: ndarray) -> Tuple[ndarray, ndarray]:
        """逆伝播を計算します.

        Args:
            dout (ndarray): 出力の変化量

        Returns:
            Tuple[ndarray]: 入力の変化量
        """

        return (dout, dout)
