"""调仓日程。

为什么调仓频率是一等公民
------------------------
规划阶段的实测：日频调仓配合 15bp 单边成本，年化成本可达 75%，
足以吃掉任何日频信号。**信号是否有边际** 和 **信号能否覆盖交易成本**
是两个独立的问题，而调仓频率决定了后者的答案。

因此调仓日程不是实现细节，而是与信号并列的研究参数。
"""

from __future__ import annotations

import pandas as pd

# 频率别名 -> pandas resample 规则。
# pandas 2.2 起 "M" 被 "ME" 取代，这里同时兼容。
_FREQ_ALIASES = {
    "D": None,          # 每日
    "W": "W",           # 每周（取每期最后一个交易日）
    "M": "ME",
    "ME": "ME",
    "2W": "2W",
}


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "W") -> pd.DatetimeIndex:
    """从交易日历中取出调仓日。

    Parameters
    ----------
    index
        全部交易日。
    freq
        "D" 每日 / "W" 每周 / "M" 每月 / "2W" 每两周。

    Returns
    -------
    调仓日的子集（升序）。每周/每月取该期 **最后一个交易日** ——
    这保证调仓日一定是真实交易日，而非日历上的周末。
    """
    index = pd.DatetimeIndex(index).sort_values()
    rule = _FREQ_ALIASES.get(freq.upper(), _FREQ_ALIASES.get(freq))
    if freq.upper() == "D":
        return index

    if rule is None:
        raise ValueError(f"不支持的调仓频率: {freq!r}（可选 D / W / M / 2W）")

    try:
        grouped = pd.Series(index, index=index).resample(rule).last().dropna()
    except ValueError:
        # 兼容旧版 pandas 的月末别名
        grouped = pd.Series(index, index=index).resample("M").last().dropna()
    return pd.DatetimeIndex(grouped.to_numpy())
