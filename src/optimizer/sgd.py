#
# SGD
#
from typing import List, Tuple

from src.layer import Layer
from src.layerdiff import LayerDifferencial

from . import Optimizer


class SGD(Optimizer):
    """SGD(確率的勾配降下法)
    """

    def __init__(self, learn_rate: float) -> None:
        """学習率を指定してオプティマイザを初期化します.

        Args:
            learn_rate (float): 学習率
        """
        self.lr = learn_rate

    def update(self, diffs: List[Tuple[Layer, LayerDifferencial]]):
        for layer, diff in diffs:
            layer.w -= self.lr * diff.dw
            layer.b -= self.lr * diff.db
