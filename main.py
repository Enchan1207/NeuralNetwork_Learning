#
# NeuralNetwork_Learning
#
import sys
from typing import List
from src.mnist_loader import get_datamodel


def main(args: List[str]) -> int:
    (x_train, t_train), (x_test, t_test) = get_datamodel()

    print("訓練データ:")
    print(f"\t画像:{x_train.shape}\n\t教師:{t_train.shape}")
    print("テストデータ:")
    print(f"\t画像:{x_test.shape}\n\t教師:{t_test.shape}")

    return 0


if __name__ == "__main__":
    result = 0
    try:
        result = main(sys.argv) or 0
    except KeyboardInterrupt:
        print("Ctrl+C")
        exit(result)
