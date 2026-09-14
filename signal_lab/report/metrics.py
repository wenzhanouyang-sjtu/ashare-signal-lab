"""绩效指标。

约定：无风险利率取 0（A 股研究的常见做法，也便于与文献对比）。
所有年化均按 252 个交易日折算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown_series(equity: pd.Series) -> pd.Series:
    """回撤序列（相对历史最高净值的比例，≤ 0）。"""
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def performance_stats(returns: pd.Series, *, name: str = "") -> dict:
    """把日收益序列汇总成一组绩效指标。

    返回的字典会被直接写进 summary.csv，因此键名保持英文、
    且在 README 中有对应解释。
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return {"name": name, "n_days": 0}

    equity = (1.0 + r).cumprod()
    n = len(r)
    years = n / TRADING_DAYS

    ann_ret = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else np.nan
    mdd = max_drawdown(equity)

    return {
        "name": name,
        "n_days": n,
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": float(ann_ret / abs(mdd)) if mdd < 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
    }


def format_stats(stats: dict) -> str:
    """把指标格式化成一行可读文本。"""
    if not stats.get("n_days"):
        return f"{stats.get('name', ''):<28} （无数据）"
    return (
        f"{stats['name']:<28} "
        f"夏普 {stats['sharpe']:>+6.2f}  "
        f"年化 {stats['annual_return']:>+7.2%}  "
        f"波动 {stats['annual_vol']:>6.2%}  "
        f"回撤 {stats['max_drawdown']:>7.2%}  "
        f"Calmar {stats['calmar']:>+5.2f}"
    )


def summarize_table(results: dict[str, pd.Series], **kwargs) -> pd.DataFrame:
    """把 {名称: 日收益序列} 汇总成 DataFrame。"""
    rows = [performance_stats(r, name=k, **kwargs) for k, r in results.items()]
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
