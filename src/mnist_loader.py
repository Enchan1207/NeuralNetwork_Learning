#
# MNISTデータベースローダ
#

from numpy import ndarray
from typing import Tuple
import os
import sys
from dotenv import load_dotenv

# dotenvからパスを読み込む
load_dotenv()
load_dotenv(".local.env", override=True)

# sys.pathに追加し
mnist_db_path = os.getenv("OFFICIAL_REPO_PATH")
if mnist_db_path is None:
    raise ImportError("Please specify MNIST dataset path at .env or .local.env")
sys.path.append(mnist_db_path)

# 読み込み!
print(f"Load MNIST dataset from {mnist_db_path}...")
try:
    from dataset.mnist import load_mnist  # type: ignore
except ModuleNotFoundError as e:
    print("Failed to load MNIST dataset")


def get_datamodel(**kwargs) -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]:
    """MNISTデータベースを読み込みます.

    Returns:
        Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]: 訓練用画像データ,教師データ : テスト用画像データ,教師データのタプル.
    """

    # オプションはkwargsに従う
    kwargs['normalize'] = True
    kwargs['one_hot_label'] = True

    test_datas: Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]] = load_mnist(**kwargs)
    return test_datas
