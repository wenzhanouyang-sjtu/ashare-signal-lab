"""置换检验：这个夏普是信号带来的，还是时间对齐带来的巧合？

原理
----
零假设 H0：**持仓与未来收益之间没有任何预测关系**。
若 H0 成立，把价格序列整体循环平移任意天数，再让同一个策略在同一套
持仓规则下交易，得到的夏普应与真实夏普同分布。

于是零分布的构造方式是把 **价格面板的行** 循环平移（circular shift），
权重不动 —— 这等价于平移权重，但避免了重新对齐索引的麻烦。

为什么用循环平移而不是随机置换
------------------------------
随机重排收益会同时破坏收益的波动聚集与厚尾，使零分布失真。
循环平移只切断"持仓 ↔ 收益"的对齐关系，完整保留：

    * 每只股票自身的收益序列结构（波动聚集、跳空、涨跌停）
    * 截面相关性（同一日的行被整体移动，截面结构不变）
    * 策略的换手率、持仓集中度、成本量级
    * 涨跌停与停牌的判定（可交易性面板随价格一起平移）

块平移
------
日收益存在自相关时，逐日平移会让相邻日的对齐关系部分保留。
block_size > 1 时平移量取块的整数倍，进一步削弱残余结构。

为什么不用「权重 × 收益」的快速代理
-----------------------------------
早期版本用 w·r 直接算策略收益，看似便宜，但**它和引擎不是同一个策略**：
引擎在调仓日按开盘价成交，而代理假设旧权重吃满全天的收盘到收盘收益。
实测两者在低换手信号上相关 0.99（可用），在高换手信号上只有 0.17
（不可用）—— 用它做检验等于检验另一个策略。

代价是每次置换要重跑一次引擎。实测引擎单次 0.06~0.15 秒，
1000 次置换约 1~2 分钟，完全可接受。**统计量与零分布必须出自同一条
计算路径**，这是这个模块唯一重要的设计决定。

平移的分辨率限制
----------------
可用的不同平移量只有 n_days / block_size 个（本项目约 121 个）。
因此 p 值的最小分辨率约为 1/(n_blocks+1) ≈ 0.008，
低于此值的 p 值无法区分。对本项目（预期 p 值偏大）不构成限制。

**无放回，不是有放回抽样。** 早期实现固定抽 n_permutations=1000 次，
但可选位移只有 121 个，于是同一个零假设样本被反复计入 ——
报出去的"1000 次置换"实际有效样本只有 121 个，是在虚报自由度。
现在改为**遍历全部不重复位移**（n_permutations 超过可用位移数时，
按等间距取子集，仍是确定性的无放回抽样）。
这既更正确，也更快：121 次引擎调用代替 1000 次。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.backtest.engine import run_backtest

TRADING_DAYS = 252


def _sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def _roll_panels(panels: dict[str, pd.DataFrame], shift: int) -> dict[str, pd.DataFrame]:
    """把面板的日期行循环平移 shift 位。截面结构与各序列自身结构均保持不变。"""
    out = {}
    for k, v in panels.items():
        arr = np.roll(v.to_numpy(), shift, axis=0)
        out[k] = pd.DataFrame(arr, index=v.index, columns=v.columns)
    return out


def permutation_test(
    weights: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    cfg,
    *,
    n_permutations: int = 1000,
    block_size: int = 20,
    alternative: str = "greater",
) -> dict:
    """循环平移置换检验（引擎口径，精确）。

    Parameters
    ----------
    weights
        目标权重矩阵（信号生成日，收盘后）。引擎会自行施加 signal_lag。
    panels
        clean.build_panels() 的输出。所有面板会随价格一起平移。
    cfg
        全局配置。
    n_permutations
        置换次数**上限**。可用位移数（n_days / block_size）少于它时全部遍历；
        多于它时按等间距无放回取子集 —— 任何情况下都不重复使用同一个平移量。
    block_size
        循环平移的块长，单位交易日。1 表示逐日平移。
    alternative
        "greater"（默认，检验夏普是否显著为正）或 "two-sided"。

    Returns
    -------
    dict
        sharpe_obs        真实毛夏普（引擎口径）
        sharpe_obs_net    真实净夏普
        sharpe_null_mean  零分布均值
        sharpe_null_std   零分布标准差
        p_value           置换 p 值
        n_permutations    实际执行的置换次数（= 实际用到的不同位移数）
        n_distinct_shifts 位移池大小，决定 p 值分辨率下限 1/(n+1)
        null_distribution 零分布夏普数组（供画图）

    统计量用 **毛收益**：置换检验问的是"信号有没有预测力"，
    成本是确定性的摩擦项，把它放进统计量只会稀释信号本身的检验。
    净夏普另行报告。

    本函数**不需要随机种子**：位移是穷举的，没有随机抽样环节，
    因此结果完全确定、可复现，重跑给出逐位相同的 p 值。
    """
    n_days = len(panels["close"])
    block = max(int(block_size), 1)
    n_blocks = max(n_days // block, 1)

    base = run_backtest(weights, panels, cfg)
    sharpe_obs = _sharpe(base.gross_returns.to_numpy(dtype=float))
    sharpe_obs_net = _sharpe(base.returns.to_numpy(dtype=float))

    # 全部不重复位移。k 从 1 起：k=0 即不平移，那是观测值本身，不属于零分布。
    shift_pool = sorted({k * block % n_days for k in range(1, n_blocks + 1)} - {0})
    if len(shift_pool) > n_permutations:
        # 可用位移多于上限时等间距取子集：确定性、无放回，且覆盖整个位移空间。
        idx = np.linspace(0, len(shift_pool) - 1, n_permutations).round().astype(int)
        shifts = [shift_pool[i] for i in sorted(set(idx.tolist()))]
    else:
        shifts = shift_pool

    null = np.full(len(shifts), np.nan, dtype=float)
    n_valid = 0
    for shift in shifts:
        try:
            res = run_backtest(weights, _roll_panels(panels, shift), cfg)
        except Exception:  # noqa: BLE001 — 个别平移可能造成退化面板，跳过即可
            continue
        s = _sharpe(res.gross_returns.to_numpy(dtype=float))
        if np.isfinite(s):
            null[n_valid] = s
            n_valid += 1

    null = null[:n_valid]
    if n_valid == 0 or not np.isfinite(sharpe_obs):
        return {
            "sharpe_obs": sharpe_obs,
            "sharpe_obs_net": sharpe_obs_net,
            "sharpe_null_mean": np.nan,
            "sharpe_null_std": np.nan,
            "p_value": np.nan,
            "n_permutations": 0,
            "n_distinct_shifts": len(shift_pool),
            "null_distribution": null,
        }

    if alternative == "greater":
        # +1 修正：避免 p = 0 这种"不可能更极端"的过度声明
        p = (int(np.sum(null >= sharpe_obs)) + 1) / (n_valid + 1)
    else:
        p = (int(np.sum(np.abs(null) >= abs(sharpe_obs))) + 1) / (n_valid + 1)

    return {
        "sharpe_obs": sharpe_obs,
        "sharpe_obs_net": sharpe_obs_net,
        "sharpe_null_mean": float(null.mean()),
        "sharpe_null_std": float(null.std(ddof=1)) if n_valid > 1 else np.nan,
        "p_value": float(p),
        "n_permutations": n_valid,
        "n_distinct_shifts": len(shift_pool),
        "null_distribution": null,
    }
