"""信号族。

每个信号接受收盘价面板 (date × code)，返回 **分数矩阵** 同形状：
分数越高越看多，NaN 表示当日该股票不可用。

时序纪律：第 t 行的分数只允许使用 ≤ t 的数据。所有实现都通过对
shift/rolling 的显式使用来保证这一点，并由 tests/test_signals.py 的
未来函数检测覆盖。
"""

from signal_lab.signals.base import SIGNAL_REGISTRY, SignalSpec, build_scores
from signal_lab.signals import momentum, pairs, reversal, volatility

__all__ = [
    "SIGNAL_REGISTRY",
    "SignalSpec",
    "build_scores",
    "momentum",
    "pairs",
    "reversal",
    "volatility",
]
