#
# 乗算コンポーネント
#
from typing import Optional, Tuple

from numpy import ndarray


class Multiplexer:

    """乗算コンポーネント
    """

    def __init__(self) -> None:
        self._x: Optional[ndarray] = None
        self._y: Optional[ndarray] = None

    def forward(self, x: ndarray, y: ndarray) -> ndarray:
        """順伝播を計算します.

        Args:
            x (ndarray):入力1
            y (ndarray): 入力2

        Returns:
            ndarray: 順伝播演算結果
        """

        # backwardで使うので保存しておく
        self._x = x
        self._y = y

        return x * y

    def backward(self, dout: ndarray) -> Tuple[ndarray, ndarray]:
        """逆伝播を計算します.

        Args:
            dout (ndarray): 出力の変化量

        Returns:
            Tuple[ndarray]: 入力の変化量

        Raises:
            ValueError: 初期化後、一度もforwardを呼び出さずに呼び出した場合.
        """

        if self._x is None or self._y is None:
            raise ValueError("Please call forward() at least once before call backward().")

        # それぞれ、forwardで入力された量*doutが返る
        dx = self._y * dout
        dy = self._x * dout

        return (dx, dy)
