"""低波动异象信号。

经济逻辑
--------
低波动股票的风险调整收益长期高于高波动股票，与"高风险高收益"的
教科书结论相悖。解释包括：杠杆约束使投资者被迫追逐高波动股、
彩票偏好、以及基准挂钩的资金流。A 股同样有此记录。

该信号与反转/动量相关性低，是一个有用的分散化来源。

分数定义
--------
    score = −过去 lookback 日已实现波动率

分数越高越看多 —— 即波动越低的股票得分越高。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.signals.base import register


def low_vol_scores(close: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """已实现波动率的相反数。仅使用 ≤ t 的价格。"""
    log_ret = np.log(close / close.shift(1))
    realized = log_ret.rolling(lookback, min_periods=max(lookback // 2, 5)).std()
    scores = -realized
    return scores.replace([np.inf, -np.inf], np.nan)


def register_all() -> None:
    register(
        "volatility",
        low_vol_scores,
        description="低波动异象：做多低波动组，做空高波动组",
    )
