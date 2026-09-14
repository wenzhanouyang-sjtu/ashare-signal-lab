"""统计套利（配对交易）信号。

方法论
------
1. 在每个调仓日，用滚动窗口估计 **全部两两配对** 的对数价格回归系数 beta
   与相关系数。这一步用矩阵运算一次算完，避免对 4.5 万对逐一循环。
2. 用相关系数做 **预筛选** —— 只保留走势高度同步的配对。这既是效率考量，
   也是方法论要求：协整的前提是两个标的受共同的因子驱动。
3. 对候选配对计算价差的 z 分数，取错价最极端的 top_k 个建仓：
   价差偏高 → 做空价差（空 y、多 x）；价差偏低 → 反之。

进场 / 出场缓冲带
------------------
|z| ≥ entry_z 时开仓，|z| ≤ exit_z 时平仓，两者之间 **维持现状**。
若只用单一阈值（|z| 回落就平），持仓会在阈值附近反复开关，制造大量
无效换手 —— 而本项目已经证明成本是净收益的头号杀手。缓冲带把"开仓
需要多极端"与"平仓需要多收敛"解耦，是降低换手的标准做法。

仓位需要跨调仓日维持，因此本函数是 **状态化** 的：按时间顺序遍历调仓日，
用 held 字典记录当前持仓的配对及其 beta。

关于"样本内筛选 vs 样本外表现"
------------------------------
规划阶段的快速探测显示：用样本内协整检验挑出的配对，其样本外表现
并不优于随机挑选（z 值为负）。本框架的作用正是把这件事放到完整的
样本外 + 多重检验框架下重新检验，而不是假设它成立。

分数定义
--------
本族不返回"分数"，而是直接返回 **目标权重** —— 因为配对交易的仓位
天然是成对构建的，无法拆成单只股票的横截面排序。
故 weight_builder = "direct"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.signals.base import register


def estimate_halflife(spread: np.ndarray) -> float:
    """用 AR(1) 估计均值回复半衰期（OU 过程）。

    拟合 Δs_t = a + b·s_{t−1}，则半衰期 = −ln2 / ln(1+b)。
    b ≥ 0（无回复）或 b ≤ −1（振荡）时返回 inf / nan。
    """
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 20:
        return float("nan")
    ds = np.diff(s)
    lag = s[:-1]
    X = np.column_stack([np.ones(len(lag)), lag])
    try:
        coef, *_ = np.linalg.lstsq(X, ds, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    b = coef[1]
    if b >= 0 or b <= -1:
        return float("inf") if b >= 0 else float("nan")
    return float(-np.log(2.0) / np.log(1.0 + b))


def _pair_z(
    values: np.ndarray,
    t: int,
    zscore_window: int,
    pairs: list[tuple[int, int, float]],
) -> np.ndarray:
    """给定 (i, j, beta) 列表，算每个配对在 t 日的价差 z 分数。"""
    if not pairs:
        return np.empty(0)
    gi = np.array([p[0] for p in pairs])
    gj = np.array([p[1] for p in pairs])
    b = np.array([p[2] for p in pairs])
    zw = values[t - zscore_window + 1 : t + 1]
    spreads = zw[:, gi] - b[None, :] * zw[:, gj]          # (zscore_window, n_pairs)
    mu = spreads.mean(axis=0)
    sd = spreads.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (spreads[-1] - mu) / sd


def pairs_weights(
    close: pd.DataFrame,
    *,
    beta_window: int = 60,
    zscore_window: int = 60,
    min_corr: float = 0.7,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    top_k: int = 20,
    gross_exposure: float = 1.0,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """在每个调仓日生成配对交易目标权重。

    只在调仓日有值，其余为 NaN（引擎遇到全 NaN 行不交易，即维持持仓）。
    调仓日若判无可持有配对，则写 **全零行** —— 明确表示目标为空仓，
    这与"没有信号"不同：前者要求平掉现有持仓，后者是维持。
    """
    log_px = np.log(close.replace(0.0, np.nan))
    n_dates, n_codes = log_px.shape
    values = log_px.to_numpy()

    dates = rebalance_dates if rebalance_dates is not None else log_px.index
    out = np.full((n_dates, n_codes), np.nan)
    date_pos = {d: i for i, d in enumerate(log_px.index)}

    min_window = max(beta_window, zscore_window)
    held: dict[tuple[int, int], float] = {}      # (i, j) -> beta
    n_empty = 0

    for d in dates:
        t = date_pos.get(d)
        if t is None or t < min_window:
            continue

        win = values[t - beta_window + 1 : t + 1]           # (beta_window, n)
        ok = np.isfinite(win).all(axis=0)
        idx = np.nonzero(ok)[0]
        if len(idx) < 10:
            continue

        xw = win[:, idx]
        xc = xw - xw.mean(axis=0, keepdims=True)
        cov = (xc.T @ xc) / (beta_window - 1)
        var = np.diag(cov).copy()
        var[var <= 0] = np.nan
        std = np.sqrt(var)

        sub_iu = np.triu_indices(len(idx), k=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov[sub_iu] / (std[sub_iu[0]] * std[sub_iu[1]])
            betas = cov[sub_iu] / var[sub_iu[1]]

        sel = np.isfinite(corr) & (corr >= min_corr) & np.isfinite(betas) & (betas > 0)
        cand = [(int(idx[i]), int(idx[j]), float(b))
                for i, j, b in zip(sub_iu[0][sel], sub_iu[1][sel], betas[sel])]
        cand_set = {(i, j): b for i, j, b in cand}

        # ---- 1) 已持仓：更新 beta，判断是否触及平仓线
        still_held: list[tuple[int, int, float]] = []
        for (i, j), b in held.items():
            b_now = cand_set.get((i, j), b)      # beta 随窗口滚动更新
            still_held.append((i, j, b_now))
        z_held = _pair_z(values, t, zscore_window, still_held)
        keep: dict[tuple[int, int], float] = {}
        for (i, j, b), zz in zip(still_held, z_held):
            if not np.isfinite(zz):
                continue                          # 数据缺失 → 平仓
            if abs(zz) > exit_z:
                keep[(i, j)] = b

        # ---- 2) 新开仓：|z| ≥ entry_z，按极端程度排序补足到 top_k
        z_cand = _pair_z(values, t, zscore_window, cand)
        enters = [
            (abs(zz), i, j, b)
            for (i, j, b), zz in zip(cand, z_cand)
            if np.isfinite(zz) and abs(zz) >= entry_z and (i, j) not in keep
        ]
        enters.sort(reverse=True)
        for _, i, j, b in enters:
            if len(keep) >= top_k:
                break
            keep[(i, j)] = b
        held = keep

        # ---- 3) 写出目标权重（按总敞口归一，含 beta 腿）
        row = np.zeros(n_codes)
        if not keep:
            n_empty += 1
            out[t] = row
            continue

        pairs_list = [(i, j, b) for (i, j), b in keep.items()]
        z_keep = _pair_z(values, t, zscore_window, pairs_list)
        legs = np.zeros(n_codes)
        for (i, j, b), zz in zip(pairs_list, z_keep):
            sgn = -np.sign(zz) if np.isfinite(zz) else 0.0
            if sgn == 0.0:
                continue
            legs[i] += sgn
            legs[j] += -sgn * b
        gross = np.abs(legs).sum()
        if gross <= 0:
            out[t] = row
            continue
        out[t] = legs * (gross_exposure / gross)

    df = pd.DataFrame(out, index=log_px.index, columns=close.columns)
    df.attrs["n_empty_rebalances"] = n_empty
    return df


def register_all() -> None:
    register(
        "pairs",
        pairs_weights,
        weight_builder="direct",
        description="统计套利：在协整配对中交易价差的极端偏离",
    )
