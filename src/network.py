#
# ニューラルネットワーク
#

from typing import List, Optional, Tuple, Union

from numpy import ndarray

from src.activator import Activator
from src.layer import Layer
from src.layerdiff import LayerDifferencial
from src.lossfunc import CrossEntropyError, LossFunction


class NeuralNetwork:
    """ニューラルネットワーク
    """

    def __init__(self,
                 layers: Optional[List[Layer]] = None,
                 loss_func: Optional[LossFunction] = None) -> None:
        """レイヤと損失関数を指定してニューラルネットワークを初期化します.

        Args:
            layers (Optional[List[Layer]]): レイヤ
            loss_func (Optional[LossFunction]): 損失関数
        """
        self.layers: List[Layer] = layers or []
        self.loss_func = loss_func or CrossEntropyError()

    def addlayer(self, layer: Layer, index: Optional[int] = None):
        """ネットワークの指定位置にレイヤを追加します.

        Args:
            layer (Layer): 追加するレイヤ
            index (Optional[int]): 追加対象の位置.
        """

        if index is None:
            self.layers.append(layer)
        else:
            self.layers.insert(index, layer)

    def removelayer(self, index: Optional[int] = None):
        """ネットワークの指定位置にあるレイヤを削除します.

        Args:
            index (Optional[int]): 削除するレイヤの位置
        """
        if index is None:
            self.layers.pop()
        else:
            self.layers.pop(index)

    def predict(self, x: ndarray) -> ndarray:
        """入力を投入し、推論を行います.

        Args:
            x (ndarray): 入力

        Returns:
            ndarray: 推論結果
        """

        # 各レイヤにxを入力して連鎖する
        result = x
        for layer in self.layers:
            # Softmax関数は通過しない
            is_softmax: bool = isinstance(layer.activator, Activator)
            result = layer.forward(result, is_softmax)

        return result

    def loss(self, x: ndarray, t: ndarray) -> ndarray:
        """訓練データおよび教師データを投入し、損失関数を計算します.

        Args:
            x (ndarray): 訓練データ
            t (ndarray): 教師データ

        Returns:
            ndarray: 損失関数の計算結果
        """

        # 各レイヤにxを入力して連鎖する
        result = x
        for layer in self.layers:
            result = layer.forward(result)

        # 交差エントロピー誤差を計算する
        loss = self.loss_func.forward(result, t)

        return loss

    def gradient(self, x: ndarray, t: ndarray) -> List[Tuple[Layer, LayerDifferencial]]:
        """訓練データおよび教師データを投入し、各レイヤの勾配を計算します.

        Args:
            x (ndarray): 訓練データ
            t (ndarray): 教師データ

        Returns:
            List[Tuple[Layer, LayerDifferencial]]: 勾配の計算結果.
        """

        # まずはforwardを回して、レイヤの状態を設定する
        self.loss(x, t)

        dout: Union[float, ndarray] = 1.0

        # 損失関数の逆伝播を求め、
        loss_back = self.loss_func.backward(dout)

        # 次にレイヤのbackwardを回して勾配とする
        differencials: List[Tuple[Layer, LayerDifferencial]] = []
        result = loss_back
        for i in range(len(self.layers) - 1, 0, -1):  # レイヤ数~0で回すため
            layer = self.layers[i]
            layer_diff = layer.backward(result)
            result = layer_diff.dx

            differencials.append((layer, layer_diff))

        return differencials
