#
# 学習オプティマイザの基底クラス
#

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from src.layer import Layer
from src.layerdiff import LayerDifferencial


class Optimizer(metaclass=ABCMeta):
    """学習オプティマイザの基底クラス
    """

    @abstractmethod
    def update(self, diffs: List[Tuple[Layer, LayerDifferencial]]):
        """レイヤとその微分値から最適化を行います.

        Args:
            diffs (List[Tuple[Layer, LayerDifferencial]]): 入力

        Note:
            この関数は副作用を伴います.
        """
