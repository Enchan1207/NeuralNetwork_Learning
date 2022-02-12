#
# NeuralNetwork_Learning
#
import sys
from typing import List

import numpy as np
from numpy import ndarray
from src.activator import Relu
from src.mnist_loader import get_datamodel
from src.network import NeuralNetwork
from src.optimizer import SGD


def main(args: List[str]) -> int:
    (x_train, t_train), (x_test, t_test) = get_datamodel()

    print("訓練データ:")
    print(f"\t画像:{x_train.shape}\n\t教師:{t_train.shape}")
    print("テストデータ:")
    print(f"\t画像:{x_test.shape}\n\t教師:{t_test.shape}")

    # ネットワーク生成
    network = NeuralNetwork(784, 10)
    network.add_layer(100, Relu)
    print(network)

    # オプティマイザ生成
    lr = 0.1
    optimizer = SGD(lr)

    # 学習開始!
    batch_size = 100
    train_size = x_train.shape[0]
    step_count = 10000
    for step in range(step_count):
        print(f"Step {step}:")

        # ミニバッチの生成
        batch_indices = np.random.choice(train_size, batch_size)
        x_batch: ndarray = x_train[batch_indices]
        t_batch: ndarray = t_train[batch_indices]

        gradient = network.gradient(x_batch, t_batch)
        optimizer.update(gradient)

    print(network)

    return 0


if __name__ == "__main__":
    result = 0
    try:
        result = main(sys.argv) or 0
    except KeyboardInterrupt:
        print("Ctrl+C")
        exit(result)
