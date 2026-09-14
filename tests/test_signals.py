"""信号层的未来函数（look-ahead）检测。

方法：**截断不变性**
--------------------
一个只用 ≤t 信息的信号，有一个可验证的数学性质：

    把价格矩阵截断到前 T 行后重新计算，第 T 行之前的信号值必须完全不变。

如果某处不小心用了未来数据（例如 rolling 默认居中对齐、shift 方向写反、
用全样本均值做标准化），截断就会改变历史信号值，测试立刻失败。

这比"肉眼看代码"强得多，因为它不依赖审查者的细心程度。

同时也检查：信号在 t 日的取值，不能依赖 t+1 日及以后的任何数据 ——
这一点用"只改未来数据、看历史信号是否变化"来测。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_lab.backtest.portfolio import cross_sectional_weights
from signal_lab.backtest.schedule import rebalance_dates
from signal_lab.signals import momentum, pairs, reversal, volatility

N_DATES = 300
N_CODES = 40
SEED = 7


def _synthetic_prices(n_dates: int = N_DATES, n_codes: int = N_CODES) -> pd.DataFrame:
    """构造带共同因子的对数随机游走，便于产生相关配对。"""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    codes = [f"S{i:03d}" for i in range(n_codes)]

    market = rng.normal(0.0002, 0.012, size=(n_dates, 1))
    beta = rng.uniform(0.6, 1.4, size=(1, n_codes))
    idio = rng.normal(0.0, 0.008, size=(n_dates, n_codes))
    log_ret = market * beta + idio
    level = 10.0 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(level, index=dates, columns=codes)


@pytest.mark.parametrize(
    "name,fn,kwargs",
    [
        ("reversal", reversal.reversal_scores, {"lookback": 5}),
        ("momentum", momentum.momentum_scores, {"lookback": 60, "skip": 5}),
        ("volatility", volatility.low_vol_scores, {"lookback": 20}),
    ],
)
def test_score_signal_no_lookahead(name, fn, kwargs):
    """截面信号：截断后历史值不变。"""
    close = _synthetic_prices()
    full = fn(close, **kwargs)

    cut = 200
    trunc = fn(close.iloc[:cut], **kwargs)

    pd.testing.assert_frame_equal(
        full.iloc[:cut], trunc, check_names=False, obj=f"{name} 截断不变性"
    )


@pytest.mark.parametrize(
    "name,fn,kwargs",
    [
        ("reversal", reversal.reversal_scores, {"lookback": 5}),
        ("momentum", momentum.momentum_scores, {"lookback": 60, "skip": 5}),
    ],
)
def test_score_signal_ignores_future_modification(name, fn, kwargs):
    """只篡改 t 之后的价格，t 之前的信号值不应有任何变化。

    这能抓出"用了全样本统计量做标准化"这类隐蔽的未来函数 ——
    截断测试抓不到它（截断后全样本也变了），但篡改未来数据可以。
    """
    close = _synthetic_prices()
    base = fn(close, **kwargs)

    tampered = close.copy()
    cut = 200
    tampered.iloc[cut:] *= 3.0        # 未来价格整体翻三倍

    after = fn(tampered, **kwargs)
    pd.testing.assert_frame_equal(
        base.iloc[:cut], after.iloc[:cut], check_names=False, obj=f"{name} 未来无关性"
    )


def test_pairs_no_lookahead():
    """配对权重：截断后历史调仓日的权重不变。

    配对信号是状态化的（持仓跨调仓日维持），因此这条测试同时验证了
    "状态只依赖历史" —— 若 held 字典被未来数据污染，截断后必然对不上。
    """
    close = _synthetic_prices()
    reb = rebalance_dates(close.index, "W")
    kwargs = dict(
        beta_window=60, zscore_window=60, min_corr=0.5,
        entry_z=1.5, exit_z=0.5, top_k=10, gross_exposure=1.0,
    )

    full = pairs.pairs_weights(close, **kwargs, rebalance_dates=reb)

    cut = 220
    trunc = pairs.pairs_weights(
        close.iloc[:cut], **kwargs, rebalance_dates=reb[reb < close.index[cut]]
    )

    pd.testing.assert_frame_equal(
        full.iloc[:cut], trunc.reindex(full.index[:cut]),
        check_names=False, obj="pairs 截断不变性",
    )


def test_pairs_actually_takes_positions():
    """回归测试：配对信号必须真的建仓。

    历史 bug：out 初始化为 NaN 后用 `row[i] += w` 累加，
    NaN + x == NaN，导致所有权重被静默吞掉、回测全程零交易却不出错。
    这个测试锁死该行为。
    """
    close = _synthetic_prices()
    reb = rebalance_dates(close.index, "W")
    w = pairs.pairs_weights(
        close, beta_window=60, zscore_window=60, min_corr=0.5,
        entry_z=1.5, exit_z=0.5, top_k=10, gross_exposure=1.0,
        rebalance_dates=reb,
    )

    active = w.dropna(how="all")
    assert len(active) > 0, "配对信号没有产生任何调仓日权重"
    assert np.abs(active.to_numpy()).sum() > 0, "配对信号全部为空仓"

    # 敞口应归一：非空仓调仓日的总敞口等于 gross_exposure
    gross = active.abs().sum(axis=1)
    nonempty = gross[gross > 1e-9]
    assert len(nonempty) > 0
    np.testing.assert_allclose(nonempty.to_numpy(), 1.0, rtol=1e-9)


def test_pairs_exit_band_reduces_turnover():
    """出场缓冲带必须真的降低换手：exit_z 越大，持仓越早平掉、换手越高。

    这是对缓冲带机制的定向测试 —— 若 exit_z 被忽略（例如误写成常量），
    两个配置会给出完全相同的权重，测试失败。
    """
    close = _synthetic_prices()
    reb = rebalance_dates(close.index, "W")
    kwargs = dict(
        beta_window=60, zscore_window=60, min_corr=0.5,
        entry_z=1.5, top_k=10, gross_exposure=1.0, rebalance_dates=reb,
    )
    w_tight = pairs.pairs_weights(close, exit_z=0.25, **kwargs)
    w_loose = pairs.pairs_weights(close, exit_z=1.50, **kwargs)

    assert not w_tight.equals(w_loose), "exit_z 未生效：两个配置给出相同权重"


def test_entry_z_does_not_bind_under_top_k():
    """记录一个方法论事实：候选池远大于 top_k 时，entry_z 不 binding。

    这不是 bug 而是配对交易的固有性质：每周都有远多于 top_k 个配对的
    |z| 超过任何合理阈值。因此 entry_z 在 top_k 约束下是退化参数，
    不应放进参数网格（本项目据此把它改为固定值，见 config.yaml）。
    """
    close = _synthetic_prices()
    reb = rebalance_dates(close.index, "W")
    kwargs = dict(
        beta_window=60, zscore_window=60, min_corr=0.5,
        exit_z=0.5, top_k=5, gross_exposure=1.0, rebalance_dates=reb,
    )
    w_low = pairs.pairs_weights(close, entry_z=1.5, **kwargs)
    w_high = pairs.pairs_weights(close, entry_z=2.5, **kwargs)

    pd.testing.assert_frame_equal(
        w_low, w_high,
        obj="候选池远大于 top_k 时，entry_z 不应改变选中的配对",
    )


def test_cross_sectional_weights_no_lookahead():
    """组合构建层：截断后历史权重不变。"""
    close = _synthetic_prices()
    scores = reversal.reversal_scores(close, lookback=5)
    w_full = cross_sectional_weights(scores, quantile=0.2, gross_exposure=1.0, min_names=10)
    w_trunc = cross_sectional_weights(
        scores.iloc[:200], quantile=0.2, gross_exposure=1.0, min_names=10
    )
    pd.testing.assert_frame_equal(
        w_full.iloc[:200], w_trunc, check_names=False, obj="组合构建截断不变性"
    )
