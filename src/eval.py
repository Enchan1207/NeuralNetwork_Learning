#
# NNの評価を行う
#
import numpy as np
from numpy import ndarray
from .network import NeuralNetwork


def evaluate_accuracy(network: NeuralNetwork, x: ndarray, t: ndarray) -> float:
    """NNに対してテストデータを投入し、精度評価を行います.

    Args:
        network (NeuralNetwork): 対象となるニューラルネットワーク
        x (ndarray): テストデータ
        t (ndarray): テスト教師データ

    Returns:
        float: 精度
    """
    y = network.predict(x)

    y = np.argmax(y, axis=1)
    if t.ndim != 1:
        t = np.argmax(t, axis=1)

    result: float = np.sum(y == t) / float(x.shape[0])
    return result
