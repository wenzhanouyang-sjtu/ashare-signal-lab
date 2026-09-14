"""多重检验校正。

为什么必须有这一层
------------------
本项目一共回测了 N 个信号参数组合。如果只看"夏普最高的那个"，
即使所有信号都毫无预测力，N 次抽样里的最大值也会是正数 ——
这正是数据窥探（data snooping）的机制。

在 N = 20 次独立检验、每次都是纯噪声的情况下，最高夏普的期望值
约为 1.87 倍标准误，而不是 0。不做校正就直接汇报"最好的那个"，
等于系统性地高估自己。

这里提供两种校正：
    Bonferroni         控制族错误率 (FWER)，最保守
    Benjamini-Hochberg 控制错误发现率 (FDR)，通常更实用
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bonferroni(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Bonferroni 校正：阈值收紧为 alpha / N。控制 FWER。"""
    p = np.asarray(pvalues, dtype=float)
    return p < (alpha / len(p))


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR 校正。

    Returns
    -------
    (reject, adjusted_p)
        reject      布尔数组，True 表示在该 FDR 水平下显著
        adjusted_p  校正后的 p 值（单调化后），可与 alpha 直接比较

    做法：把 p 值升序排列，找到最大的 k 使得 p_(k) ≤ k/N · alpha，
    则前 k 个拒绝。这比 Bonferroni 有更高的检验力。
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    order = np.argsort(p)
    ranked = p[order]

    # 校正后的 p 值，并从大到小取累计最小值以保证单调性
    adjusted_sorted = ranked * n / np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted
    return adjusted <= alpha, adjusted


def summarize(pvalues: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """把原始 p 值与两种校正结果并排放进一张表。"""
    p = pvalues.to_numpy(dtype=float)
    reject_bh, adj_bh = benjamini_hochberg(p, alpha)
    reject_bonf = bonferroni(p, alpha)

    return pd.DataFrame(
        {
            "name": pvalues.index,
            "p_value": p,
            "p_adjusted_bh": adj_bh,
            "significant_raw": p <= alpha,
            "significant_bh": reject_bh,
            "significant_bonferroni": reject_bonf,
        }
    ).sort_values("p_value").reset_index(drop=True)
