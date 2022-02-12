#
# ニューラルネットワーク
#

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from src import layer

from src.activator import Activator, Relu, Softmax
from src.layer import Layer
from src.layerdiff import LayerDifferencial
from src.lossfunc import CrossEntropyError


class NeuralNetwork:
    """ニューラルネットワーク
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """入出力のニューロン数を指定してニューラルネットワークを初期化します.

        Args:
            input_size  (int): 入力層ニューロン数
            output_size (int): 出力層ニューロン数

        Note:
            初期化時に自動でレイヤが追加され、最低1層のNNが生成されます.
        """
        self.loss_func = CrossEntropyError()

        # 初期レイヤの設定
        initial_layer = Layer.create_by((input_size, output_size), Softmax())
        self.layers = [initial_layer]
        self.input_size = input_size
        self.output_size = output_size

    def add_layer(self, neuron_size: int, activator: Activator, index: Optional[int] = None):
        """ネットワークの指定位置に指定ニューロン数のレイヤを追加します.

        Args:
            neuron_size (int): 追加するレイヤのニューロン数
            activator (Activator): 追加するレイヤの活性化関数
            index (Optional[int]): レイヤの追加位置. 指定のない場合は出力層の直前に追加されます.

        Raises:
            IndexError: 追加位置に不正な値が渡された場合.

        Note:
            既存のNN構成は破壊されます(再学習が必要になります).
        """

        # 追加先を特定
        index = index if index is not None else len(self.layers) - 1
        if index < 0 or index >= len(self.layers):
            raise IndexError(f"Invalid index: {index} Please specify 0 ~ (layer count - 1).")

        # 1. 新しいレイヤを追加する. 行数は前のレイヤの出力数またはNNの入力数とする.
        new_input_size = self.layers[index - 1].shape[1] if index > 0 else self.input_size
        new_layer = Layer.create_by((new_input_size, neuron_size), activator)
        self.layers.insert(index, new_layer)

        # 2. 次のレイヤの入力数を修正する.
        _, next_output = self.layers[index + 1].shape
        new_next_shape = (neuron_size, next_output)
        self.layers[index + 1] = Layer.create_by(new_next_shape, self.layers[index + 1].activator)

    def remove_layer(self, index: Optional[int] = None):
        """ネットワークの指定位置にあるレイヤを削除します.

        Args:
            index (Optional[int]): 削除するレイヤの位置.指定のない場合は出力層の直前が削除されます.

        Raises:
            IndexError: 追加位置に不正な値が渡された場合.

        Note:
            既存のNN構成は破壊されます(再学習が必要になります).
        """
        raise NotImplementedError()

        # 追加先を特定し、前後のレイヤインデックスを取得
        index = index or len(self.layers) - 1

        if index < 0 or index > len(self.layers):
            raise IndexError(f"Invalid index: {index} Please specify 0 ~ layer count.")

        # 既存のレイヤを削除
        self.layers.pop(index)

    def __str__(self) -> str:
        network_info: str = f"NeuralNetwork(layer: {len(self.layers)}, loss: {self.loss_func.__class__.__name__})"
        layers_info = "\n".join([f"\t{str(layer)}" for layer in self.layers])

        return f"{network_info}\n{layers_info}"

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
