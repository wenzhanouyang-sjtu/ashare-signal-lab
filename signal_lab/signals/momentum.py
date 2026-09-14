"""截面动量信号。

经济逻辑
--------
中期赢家倾向继续跑赢（趋势延续、信息扩散缓慢、处置效应）。
与反转信号构成一对竞争假设：短周期（数日）以反转为主，
中长周期（数月）以动量为主，二者的分界正是本项目要检验的问题之一。

关于 skip
---------
标准做法是跳过最近若干个交易日再计算收益。原因：最近 1 个月的收益
带有强烈的短期反转成分，会污染动量信号。skip=0 与 skip=5 的对比
本身就是一组有意义的对照实验。

分数定义
--------
    score = log P(t − skip) − log P(t − skip − lookback)

分数越高越看多。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.signals.base import register


def momentum_scores(
    close: pd.DataFrame, lookback: int = 120, skip: int = 5
) -> pd.DataFrame:
    """过去 lookback 日收益（跳过最近 skip 日）。仅使用 ≤ t 的价格。"""
    end = close.shift(skip)
    start = close.shift(skip + lookback)
    scores = np.log(end / start)
    return scores.replace([np.inf, -np.inf], np.nan)


def register_all() -> None:
    register(
        "momentum",
        momentum_scores,
        description="截面动量：做多中期赢家，做空中期输家",
    )
