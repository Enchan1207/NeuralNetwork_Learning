#
# NNの各層を表すクラス
#
from __future__ import annotations

from typing import Optional, Tuple, Type

import numpy as np
from numpy import ndarray

from src.activator import Activator
from src.layerdiff import LayerDifferencial


class Layer:
    """NNの各層を表すクラス
    """

    def __init__(self, w: ndarray, b: ndarray, activator: Type[Activator]) -> None:
        """重みとバイアス, 活性化関数を設定してレイヤを初期化します.

        Args:
            w (ndarray): 重み
            b (ndarray): バイアス
            activator (Type[Activator]): 活性化関数
        """

        self.w: ndarray = w
        self.b: ndarray = b
        self.activator = activator()

        self._x: Optional[ndarray] = None

    @staticmethod
    def create_by(shape: Tuple[int, int], activator: Type[Activator]) -> Layer:
        """形状と活性化関数を指定して初期状態のレイヤを生成します.

        Args:
            shape (Tuple[int, int]): 形状
            activator (Type[Activator]): 活性化関数

        Returns:
            Layer: 形状パラメータをもとに生成されたレイヤ

        Note:
            初期状態のレイヤでは、重みはガウス分布*0.01, バイアスはゼロの状態になっています.
        """

        return Layer(
            np.random.randn(*shape) * 0.01,
            np.zeros(shape[1]),
            activator
        )

    @property
    def shape(self) -> Tuple[int, int]:
        """レイヤの形状を返します.

        Returns:
            Tuple[int, int]: 形状(入力ニューロン数, 出力ニューロン数)
        """
        shape_ = self.w.shape
        try:
            row, col = shape_
        except ValueError:
            row, col = 1, shape_[0]

        return (row, col)

    def __str__(self) -> str:
        return f"Layer(in: {self.shape[0]} out: {self.shape[1]}, activator: {self.activator.__class__.__name__})"

    def forward(self, x: ndarray, pass_activator: bool = False) -> ndarray:
        """レイヤにデータを投入し, 結果を返します.

        Args:
            x (ndarray): 入力
            pass_activator (bool, optional): 活性化関数を通さない場合はTrueに設定します.

        Returns:
            ndarray: 出力
        """

        self._x = x

        result: ndarray = np.dot(x, self.w) + self.b

        if not pass_activator:
            result = self.activator.forward(result)

        return result

    def backward(self, dout: ndarray, pass_activator: bool = False) -> LayerDifferencial:
        """このレイヤの逆伝播を計算します.

        Args:
            dout (ndarray): 出力の変化量
            pass_activator (bool, optional): 活性化関数を通さない場合はTrueに設定します.

        Returns:
            LayerDifferencial: 入力の変化量

        Raises:
            ValueError: 初期化後, 一度もforwardを呼び出さずに呼び出した場合.
        """

        if self._x is None:
            raise ValueError("Please call forward() at least once before call backward().")

        # 活性化関数に通して
        if not pass_activator:
            dout = self.activator.backward(dout)

        # 各変化量を返す
        dx = np.dot(dout, self.w.T)
        dw = np.dot(self._x.T, dout)
        db = np.sum(dout, axis=0)

        return LayerDifferencial(dx, dw, db)
