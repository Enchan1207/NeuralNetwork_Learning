#
# 基底クラス
#

import numpy as np
from numpy import ndarray
from abc import ABCMeta, abstractmethod


class Activator(metaclass=ABCMeta):
    """活性化関数の基底クラス.
    """

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        """順伝播を計算します.

        Args:
            x (ndarray): 入力

        Returns:
            ndarray: 出力
        """
        raise NotImplementedError()

    @abstractmethod
    def backward(self, dout: ndarray) -> ndarray:
        """逆伝播を計算します.

        Args:
            dout (ndarray): 出力の変化量

        Returns:
            ndarray: 入力の変化量

        Raises:
            ValueError: 初期化後一度もforwardを呼び出さずに実行した場合.
        """
        raise NotImplementedError()
