#
# 交差エントロピー誤差
#
from typing import Optional, Union

import numpy as np
from numpy import ndarray

from . import LossFunction


class CrossEntropyError(LossFunction):
    """交差エントロピー誤差
    """

    def __init__(self) -> None:
        self._y: Optional[ndarray] = None
        self._t: Optional[ndarray] = None

    def forward(self, y: ndarray, t: ndarray) -> ndarray:

        self._y = y
        self._t = t

        # ミニバッチ処理では交差エントロピー誤差はデータごとではなく平均値を計算して用いるため,
        # 少なくとも1行のデータに変換する ((N,)を(1,N)にする)
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        batch_size = y.shape[0]

        # one-hot表現の場合は, 教師データを正解ラベルのインデックスに変換する.
        # [[0, 1, 0].
        #  [0, 0, 1]]
        # を投げると, [1, 2]が返る(各行の最大値インデックスとなる).
        if t.size == y.size:
            t = t.argmax(axis=1)

        # 誤差を計算する.
        # tは, 各行の最大値インデックス つまり「各バッチについて(各行について)教師データの最大値は何列目か」を表す.
        # y[np.arange[batch_size], t]により, 各バッチについて「正解」にあたる部分のyの値を持ってくる.
        # (np.array[行, 列]とするとスライスを取得できる)
        cell = np.log(y[np.arange(batch_size), t] + 1E-7)  # 1E-7は log(0)=NaN の回避

        result: ndarray = -np.sum(cell) / batch_size
        return result

    def backward(self, dout: Union[ndarray, float]) -> ndarray:
        if self._t is None or self._y is None:
            raise ValueError("Please call forward() at least once before call backward().")

        # 交差エントロピー誤差単体の逆伝播は
        # -t_i/y_i * dout
        result: ndarray = -(self._t / self._y) * dout

        return result
