#
# Softmax + 交差エントロピー誤差コンポーネント
#

import numpy as np
from numpy import ndarray


class SoftmaxWithLoss:
    """Softmax + 交差エントロピー誤差コンポーネント
    """

    def __init__(self) -> None:
        self._y: ndarray = None
        self._t: ndarray = None

    def forward(self, x: ndarray, t: ndarray) -> ndarray:
        """順伝播を計算します.

        Args:
            x (ndarray): 入力
            t (ndarray): 教師データ

        Returns:
            ndarray: 出力

        Raises:
            ValueError: 不正な引数が入力された場合.
        """

        self._t = t

        # Softmaxをくぐらせて
        softmax_x = self._softmax(x)
        self._y = softmax_x

        # 交差エントロピー誤差をくぐらせる
        ce_error_x = self._cross_entropy_error(softmax_x, t)

        return ce_error_x

    def backward(self, dout: ndarray) -> ndarray:
        """逆伝播を計算します.

        Args:
            dout (ndarray): 出力の変化量

        Returns:
            ndarray: 入力の変化量

        Raises:
            ValueError: 初期化後, 一度もforwardを呼び出さずに呼び出した場合.
        """

        if self._y is None or self._t is None:
            raise ValueError("Please call forward() at least once before call backward().")

        batch_size = self._t.shape[0]
        return (self._y - self._t) / batch_size

    def _softmax(self, x: ndarray) -> ndarray:

        # ミニバッチ処理の場合multirows(複数入力の縦連)になるので, np.maxおよびnp.sumは各行ごとに計算する必要がある.
        # axis=1にするとnp.arrayを横に挟んで潰すように合計を計算してくれるが,
        # np.array.shape = (3,)みたいな形(ミニバッチでないとき)の配列を突っ込むとエラーになるので axis=-1を指定する.
        # keepdimsは合計を計算する前のnp.arrayの形状を保つフラグ.

        # exp(x)の計算
        c = np.max(x, axis=-1, keepdims=True)  # exp爆発回避
        exp_x = np.exp(x - c)

        # 横方向の合計値を計算
        sum_x = np.sum(x - c, axis=-1, keepdims=True)

        return exp_x / sum_x

    def _cross_entropy_error(self, y: ndarray, t: ndarray) -> ndarray:

        # ミニバッチ処理では交差エントロピー誤差はデータごとではなく平均値を計算して用いるため,
        # 少なくとも1行のデータに変換する ((N,)を(1,N)にする)
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        batch_size = y.shape[0]

        # one-hot表現の場合は, 教師データを正解ラベルのインデックスに変換する.
        # [[0, 1, 0].
        #  [0, 0, 1]]
        # を投げると, [1, 2]が返る(各行の最大値インデックスとなる).
        if t.size == y.size:
            t = t.argmax(axis=1)

        # 誤差を計算する.
        # tは, 各行の最大値インデックス つまり「各バッチについて(各行について)教師データの最大値は何列目か」を表す.
        # y[np.arange[batch_size], t]により, 各バッチについて「正解」にあたる部分のyの値を持ってくる.
        # (np.array[行, 列]とするとスライスを取得できる)
        cell = t * np.log(y[np.arange(batch_size), t] + 1E-7)  # 1E-7は log(0)=NaN の回避
        return -np.sum(cell) / batch_size
