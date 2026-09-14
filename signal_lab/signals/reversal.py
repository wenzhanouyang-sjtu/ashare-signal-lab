"""截面反转信号。

经济逻辑
--------
短期内跌幅较大的股票倾向于在随后数日反弹。成因包括：流动性提供者要求
的补偿、投资者过度反应后的修正、以及被迫卖出造成的价格压力。
A 股散户成交占比高、卖空受限，该效应在文献中记录得较为稳健。

分数定义
--------
    score = −(过去 lookback 日对数收益)

分数越高越看多 —— 即跌得越多的股票得分越高。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.signals.base import register


def reversal_scores(close: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """过去 lookback 日收益的相反数。仅使用 ≤ t 的价格。"""
    log_ret = np.log(close / close.shift(lookback))
    scores = -log_ret
    return scores.replace([np.inf, -np.inf], np.nan)


def register_all() -> None:
    register(
        "reversal",
        reversal_scores,
        description="截面反转：做多近期跌幅最大的一组，做空涨幅最大的一组",
    )
