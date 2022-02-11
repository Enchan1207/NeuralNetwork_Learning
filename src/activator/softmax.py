#
# Softmax関数
#

from typing import Optional

import numpy as np
from numpy import ndarray

from . import Activator


class Softmax(Activator):
    """Softmax関数
    """

    def __init__(self) -> None:
        self._y: Optional[ndarray] = None

    def forward(self, x: ndarray) -> ndarray:
        # ミニバッチ処理の場合multirows(複数入力の縦連)になるので, np.maxおよびnp.sumは各行ごとに計算する必要がある.
        # axis=1にするとnp.arrayを横に挟んで潰すように合計を計算してくれるが,
        # np.array.shape = (3,)みたいな形(ミニバッチでないとき)の配列を突っ込むとエラーになるので axis=-1を指定する.
        # keepdimsは合計を計算する前のnp.arrayの形状を保つフラグ.

        # exp(x)の計算
        c = np.max(x, axis=-1, keepdims=True)  # exp爆発回避
        exp_x = np.exp(x - c)

        # 横方向の合計値を計算
        sum_x = np.sum(x - c, axis=-1, keepdims=True)

        y: ndarray = exp_x / sum_x
        self._y = y

        return y

    def backward(self, dout: ndarray) -> ndarray:
        if self._y is None:
            raise ValueError("Please call forward() at least once before call backward().")

        # Softmax単体の逆伝播は, 配列のi番目について y_i * (dout_i - sum(y*dout))
        # なので(doutはバッチでないものとする), これのバッチ対応版を考えなきゃならない.

        # y*doutまでは定数、合計値は各行ごと(各バッチごと)
        sum_y = np.sum(self._y * dout, axis=-1, keepdims=True)

        # あとは普通に計算するだけ.
        result: ndarray = self._y * (dout - sum_y)

        return result
