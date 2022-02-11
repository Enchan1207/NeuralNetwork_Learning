#
# レイヤの微分結果を扱うクラス
#

from dataclasses import dataclass

from numpy import ndarray


@dataclass
class LayerDifferencial:

    dx: ndarray
    dw: ndarray
    db: ndarray
