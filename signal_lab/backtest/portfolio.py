"""组合构建：把信号分数转换成目标权重。

三个工具，对应三个不同的实际问题：

  cross_sectional_weights  信号 → 截面多空权重（做多高分位、做空低分位）
  apply_buffer             换手率控制 —— 进入/退出阈值分离，避免临界点反复触发
  vol_target               风险控制 —— 按已实现波动缩放总仓位

关于换手率：规划阶段的实测显示，日频调仓配合 15bp 单边成本，
年化成本可达 75%，足以吃掉任何信号。因此换手率不是事后统计，
而是构建权重时就要主动控制的对象。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_weights(
    scores: pd.DataFrame,
    *,
    quantile: float = 0.2,
    gross_exposure: float = 1.0,
    min_names: int = 10,
    long_only: bool = False,
) -> pd.DataFrame:
    """截面排序多空组合：做多分数最高的一组，做空最低的一组，组内等权。

    Parameters
    ----------
    scores
        信号矩阵 (date × code)，**分数越高越看多**。
    quantile
        每一侧选取的比例（0.2 表示各取前 20%）。
    gross_exposure
        总敞口。多空时多头与空头各占一半；long_only 时全部给多头。
    min_names
        当日有效股票数少于此值时清仓 —— 避免在样本过少时下注。

    Returns
    -------
    目标权重矩阵，与 scores 同形状。
    """
    values = scores.to_numpy()
    # 显式分配可写数组。pandas 3.0 的 Copy-on-Write 下
    # DataFrame.to_numpy() 可能返回只读视图，不可直接赋值。
    out = np.zeros(values.shape, dtype=float)

    for i in range(values.shape[0]):
        row = values[i]
        valid = np.isfinite(row)
        n_valid = int(valid.sum())
        if n_valid < min_names:
            continue

        k = max(int(n_valid * quantile), 1)
        idx = np.nonzero(valid)[0]
        # 按分数升序：最低的 k 个做空，最高的 k 个做多
        order = idx[np.argsort(row[idx], kind="stable")]
        shorts, longs = order[:k], order[-k:]

        if long_only:
            out[i, longs] = gross_exposure / k
        else:
            out[i, longs] = (gross_exposure / 2.0) / k
            out[i, shorts] = -(gross_exposure / 2.0) / k

    return pd.DataFrame(out, index=scores.index, columns=scores.columns)


def apply_buffer(
    target: pd.DataFrame,
    *,
    entry: float = 1.0,
    exit_: float = 0.0,
) -> pd.DataFrame:
    """缓冲带 / 迟滞：减少临界点附近的无效换手。

    当已有持仓时，只有目标权重回落到 exit_ 以下才平仓；
    当空仓时，只有目标权重超过 entry 才建仓。

    做法是把 |target| 映射到 {0, 1} 的持有状态，再乘回目标权重的符号与大小。
    这样对多空对称，也不依赖具体的信号量纲。
    """
    magnitude = target.abs().to_numpy()
    sign = np.sign(target.to_numpy())
    held = np.zeros(magnitude.shape[1], dtype=bool)
    out = np.zeros(magnitude.shape, dtype=float)

    for i in range(magnitude.shape[0]):
        m = np.nan_to_num(magnitude[i], nan=0.0)
        # 已持有的：跌破 exit_ 才平；未持有的：突破 entry 才开
        stay = held & (m > exit_)
        enter = (~held) & (m >= entry)
        held = stay | enter
        out[i] = np.where(held, m, 0.0)

    return pd.DataFrame(
        out * sign, index=target.index, columns=target.columns
    )


def vol_target(
    asset_returns: pd.DataFrame,
    *,
    target_vol: float = 0.10,
    window: int = 60,
    max_leverage: float = 2.0,
    min_leverage: float = 0.0,
) -> pd.DataFrame:
    """波动率目标：按组合已实现波动缩放总仓位。

    返回一个逐日的缩放系数矩阵（广播用），使组合的事前波动接近 target_vol。
    缩放系数取 T 日之前的已实现波动，不含当日 —— 避免引入未来信息。
    """
    # 等权组合的近似：先按截面等权求组合收益，再滚动估计波动
    n_valid = asset_returns.notna().sum(axis=1).replace(0, np.nan)
    port_ret = asset_returns.mean(axis=1)
    realized = port_ret.rolling(window).std() * np.sqrt(252.0)
    scale = (target_vol / realized).clip(lower=min_leverage, upper=max_leverage)
    scale = scale.shift(1)  # 用截至前一日的波动，当日起效
    return pd.DataFrame(
        np.repeat(scale.to_numpy()[:, None], asset_returns.shape[1], axis=1),
        index=asset_returns.index,
        columns=asset_returns.columns,
    ).fillna(1.0)


def inverse_vol_weights(
    scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    window: int = 60,
    quantile: float = 0.2,
    gross_exposure: float = 1.0,
    min_names: int = 10,
) -> pd.DataFrame:
    """先做截面选股，再按波动率倒数分配权重 —— 让各持仓的风险贡献接近。"""
    base = cross_sectional_weights(
        scores, quantile=quantile, gross_exposure=gross_exposure, min_names=min_names
    )
    vol = asset_returns.rolling(window).std().shift(1)
    inv = (1.0 / vol.replace(0.0, np.nan)).to_numpy()

    base_v = base.to_numpy()
    sign = np.sign(base_v)
    raw = np.abs(base_v) * inv
    out = np.zeros_like(raw)

    # 在多空两侧内部各自归一化，保持 base 赋予每侧的总敞口不变
    for i in range(raw.shape[0]):
        for side in (1, -1):
            mask = sign[i] == side
            total = np.nansum(raw[i][mask])
            if total > 0:
                out[i][mask] = raw[i][mask] / total * np.nansum(np.abs(base_v[i][mask]))

    return pd.DataFrame(
        out * sign, index=scores.index, columns=scores.columns
    ).fillna(0.0)
