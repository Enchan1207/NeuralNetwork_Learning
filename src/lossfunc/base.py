#
# 損失関数の基底クラス
#

from abc import ABCMeta, abstractmethod

from numpy import ndarray


class LossFunction(metaclass=ABCMeta):
    """損失関数の基底クラス.
    """

    @abstractmethod
    def forward(self, y: ndarray, t: ndarray) -> ndarray:
        """訓練データおよび教師データから, 損失関数の順伝播を計算します.

        Args:
            x (ndarray): 訓練データ
            t (ndarray): 教師データ

        Returns:
            ndarray: 損失関数の順伝播
        """
        raise NotImplementedError()

    @abstractmethod
    def backward(self, dout: ndarray) -> ndarray:
        """出力の変化量から, 損失関数の逆伝播を計算します.

        Args:
            dout (ndarray): 出力の変化量

        Returns:
            ndarray: 損失関数の逆伝播

        Raises:
            ValueError: 初期化後一度もforwardを呼び出さずに実行した場合.
        """
        raise NotImplementedError()
