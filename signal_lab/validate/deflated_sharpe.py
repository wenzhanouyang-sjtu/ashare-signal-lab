"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014)。

要解决的问题
------------
夏普比率的显著性通常用 t 检验判断：t = SR · sqrt(T)。但这个检验假设
**只试了一次**。如果你试了 20 个参数组合再汇报最好的那个，那么即使
全部都是噪声，最大夏普的期望值也不是 0，而是随试验次数增长。

Deflated Sharpe Ratio 把"试验次数"显式地纳入统计量：

    SR₀ = sqrt(Var{SR_n}) · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

SR₀ 是 N 次独立试验下的 **预期最大夏普**（纯噪声情形）。然后检验
观测到的 SR 是否显著高于 SR₀：

    DSR = Φ( (SR − SR₀)·sqrt(T−1) / sqrt(1 − γ₃·SR + (γ₄−1)/4·SR²) )

其中 γ₃ 为偏度、γ₄ 为峰度（非超额）。分母修正了收益分布的非正态性：
负偏、厚尾都会降低 DSR。

注意 SR 与 SR₀ 必须使用 **同一时间尺度**（此处为日频，非年化），
因为 sqrt(T−1) 的缩放依赖于此。
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni 常数


def _sharpe_nonannualized(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1))


def expected_max_sharpe(sharpe_ratios: np.ndarray, n_trials: int | None = None) -> float:
    """N 次独立试验下，纯噪声情形中最大夏普的期望值 SR₀。

    Parameters
    ----------
    sharpe_ratios
        全部试验的夏普（日频、非年化），用于估计跨试验的方差。
    n_trials
        试验次数。默认取 sharpe_ratios 的长度。
    """
    sr = np.asarray(sharpe_ratios, dtype=float)
    sr = sr[np.isfinite(sr)]
    n = int(n_trials if n_trials is not None else len(sr))
    if n < 2 or len(sr) < 2:
        return 0.0
    var_sr = float(np.var(sr, ddof=1))
    if var_sr <= 0:
        return 0.0

    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    all_trial_sharpes: np.ndarray,
    n_trials: int | None = None,
) -> dict:
    """计算单个策略的 Deflated Sharpe Ratio。

    Returns
    -------
    dict
        dsr            被"通缩"后的概率。DSR < 0.95 通常认为不足以
                       在考虑了多重检验后声称有真实技能。
        sharpe_daily   观测到的日频夏普
        sr0            该试验次数下的预期最大夏普（纯噪声基准）
        p_value        单边 p 值（H0: 真实夏普 ≤ SR₀）
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 30:
        return {"dsr": np.nan, "sharpe_daily": np.nan, "sr0": np.nan, "p_value": np.nan}

    sr = _sharpe_nonannualized(r)
    sr0 = expected_max_sharpe(all_trial_sharpes, n_trials)

    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # 非超额峰度，正态为 3

    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom_sq <= 0:
        return {"dsr": np.nan, "sharpe_daily": sr, "sr0": sr0, "p_value": np.nan}

    z = (sr - sr0) * np.sqrt(T - 1.0) / np.sqrt(denom_sq)
    dsr = float(stats.norm.cdf(z))

    return {
        "dsr": dsr,
        "sharpe_daily": sr,
        "sharpe_annual": sr * np.sqrt(252.0),
        "sr0": sr0,
        "sr0_annual": sr0 * np.sqrt(252.0),
        "p_value": float(1.0 - dsr),
        "n_trials": int(n_trials if n_trials is not None else len(all_trial_sharpes)),
        "n_obs": T,
    }
