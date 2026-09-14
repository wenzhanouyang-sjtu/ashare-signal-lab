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
可用的不同平移量只有 n_days / block_size 个（约 120 个）。
因此 p 值的最小分辨率约为 1/(n_blocks+1) ≈ 0.008，
低于此值的 p 值无法区分。对本项目（预期 p 值偏大）不构成限制。
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
    seed: int = 20240914,
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
        置换次数。
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
        n_permutations    实际有效置换次数
        n_distinct_shifts 可用的不同平移量个数（决定 p 值分辨率）
        null_distribution 零分布夏普数组（供画图）

    统计量用 **毛收益**：置换检验问的是"信号有没有预测力"，
    成本是确定性的摩擦项，把它放进统计量只会稀释信号本身的检验。
    净夏普另行报告。
    """
    n_days = len(panels["close"])
    block = max(int(block_size), 1)
    n_blocks = max(n_days // block, 1)

    base = run_backtest(weights, panels, cfg)
    sharpe_obs = _sharpe(base.gross_returns.to_numpy(dtype=float))
    sharpe_obs_net = _sharpe(base.returns.to_numpy(dtype=float))

    rng = np.random.default_rng(seed)
    null = np.full(n_permutations, np.nan, dtype=float)
    n_valid = 0
    for i in range(n_permutations):
        shift = int(rng.integers(1, n_blocks + 1)) * block % n_days
        if shift == 0:
            continue
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
            "n_distinct_shifts": n_blocks,
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
        "n_distinct_shifts": n_blocks,
        "null_distribution": null,
    }
