"""验证层的测试。

重点锁死两类曾经真实发生过、且不会报错的错误：

1. **置换检验用了错误的持仓**。曾经把「目标权重」（非调仓日为 NaN）当成
   持仓传给置换检验，`nan_to_num` 把 NaN 变成 0，于是检验以为策略在
   79% 的日子里空仓 —— 观测统计量算出来是 -0.59，而引擎的真实毛夏普是
   +0.66，符号都反了。这个 bug 不抛异常，只是悄悄给出错误结论。

2. **置换检验无法识别真信号**。一个不能通过"偷看未来应该显著"这一关的
   检验，它的 p 值没有任何意义。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_lab.backtest.engine import run_backtest
from signal_lab.backtest.portfolio import cross_sectional_weights
from signal_lab.backtest.schedule import rebalance_dates
from signal_lab.validate.multiple_testing import benjamini_hochberg, bonferroni
from signal_lab.validate.permutation import permutation_test
from signal_lab.validate.walkforward import evaluate_selection, walkforward_splits

N_DATES, N_CODES, SEED = 500, 30, 11


class _Cfg(dict):
    """极简配置对象，满足引擎对属性访问的需求。"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(k) from exc

    def path(self, *keys):
        from pathlib import Path

        return Path("/tmp") / Path(*keys)


def _cfg():
    return _Cfg(
        backtest=_Cfg(
            costs=_Cfg(
                commission=0.00025, stamp_tax=0.0005, transfer_fee=0.00001,
                impact_coef=0.10, impact_model="sqrt",
            ),
            execution=_Cfg(
                t_plus_1=True, limit_updown=True, suspension=True,
                signal_lag=1, min_trade_weight=0.001,
            ),
        )
    )


def _panels():
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2021-01-04", periods=N_DATES)
    codes = [f"S{i:02d}" for i in range(N_CODES)]
    ret = rng.normal(0.0003, 0.016, size=(N_DATES, N_CODES))
    close = pd.DataFrame(
        1500.0 * np.exp(np.cumsum(np.log1p(ret), axis=0)), index=dates, columns=codes
    )
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.004, (N_DATES, N_CODES)))
    amount = pd.DataFrame(
        rng.uniform(2e8, 8e8, size=(N_DATES, N_CODES)), index=dates, columns=codes
    )
    return {"open": open_, "close": close, "amount": amount}, close


def _weekly_weights(close: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    reb = rebalance_dates(close.index, "W")
    mask = pd.Series(False, index=close.index)
    mask.loc[mask.index.intersection(reb)] = True
    return cross_sectional_weights(
        scores.where(mask), quantile=0.2, gross_exposure=1.0, min_names=10
    )


# ---------------------------------------------------------------- 置换检验


def test_permutation_detects_lookahead():
    """偷看未来的信号必须被判定为显著。

    这是置换检验的"阳性对照"。若这个测试失败，说明检验没有检出力，
    那么它对真实信号给出的"不显著"结论也不可信。
    """
    panels, close = _panels()
    future = close.pct_change(3).shift(-3)          # 偷看未来 3 日
    w = _weekly_weights(close, future)

    pt = permutation_test(w, panels, _cfg(), n_permutations=30, block_size=10, seed=3)
    assert pt["sharpe_obs"] > 1.0, "偷看未来的信号毛夏普应当很高"
    assert pt["p_value"] < 0.1, f"偷看未来未被判显著，p={pt['p_value']}"
    assert abs(pt["sharpe_null_mean"]) < 1.0, "零分布应当集中在 0 附近"


def test_permutation_null_centered_for_random_signal():
    """纯随机信号：零分布应围绕 0，且观测值不应落在极端分位。"""
    panels, close = _panels()
    rng = np.random.default_rng(5)
    noise = pd.DataFrame(
        rng.normal(size=close.shape), index=close.index, columns=close.columns
    )
    w = _weekly_weights(close, noise)

    pt = permutation_test(w, panels, _cfg(), n_permutations=30, block_size=10, seed=7)
    assert np.isfinite(pt["p_value"])
    assert pt["p_value"] > 0.05, f"随机信号被判显著，p={pt['p_value']}"


def test_permutation_uses_realized_weights():
    """置换检验的观测统计量必须与引擎的毛夏普一致。

    回归测试：曾经传入目标权重（非调仓日为 NaN），nan→0 后检验以为策略
    大部分时间空仓，算出的观测夏普与引擎符号相反。现在接口要求显式传入
    目标权重并**由引擎自行回测**得到观测值，因此二者必然一致。
    """
    panels, close = _panels()
    rng = np.random.default_rng(13)
    scores = pd.DataFrame(
        rng.normal(size=close.shape), index=close.index, columns=close.columns
    )
    w = _weekly_weights(close, scores)

    pt = permutation_test(w, panels, _cfg(), n_permutations=10, block_size=10, seed=1)
    engine = run_backtest(w, panels, _cfg())

    def _sr(x):
        x = np.asarray(x, dtype=float)
        return x.mean() / x.std(ddof=1) * np.sqrt(252)

    assert pt["sharpe_obs"] == pytest.approx(_sr(engine.gross_returns), rel=1e-12)
    assert pt["sharpe_obs_net"] == pytest.approx(_sr(engine.returns), rel=1e-12)


def test_permutation_handles_flat_strategy():
    """从不交易的策略不应崩溃，p 值应为 NaN 或未定义而非抛异常。"""
    panels, close = _panels()
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    pt = permutation_test(w, panels, _cfg(), n_permutations=5, block_size=10, seed=1)
    assert "p_value" in pt


# ---------------------------------------------------------------- Walk-forward


def test_walkforward_splits_are_out_of_sample():
    """每一折的测试集必须在训练集之后，且两者不重叠。"""
    splits = walkforward_splits(2000, n_splits=5, train_days=750, test_days=250)
    assert len(splits) == 5
    for sp in splits:
        assert sp.train_end <= sp.test_start, "训练集与测试集重叠"
        assert sp.test_end > sp.test_start
        assert sp.test_end <= 2000


def test_walkforward_splits_anchored_grows():
    """锚定式训练集起点固定为 0，滚动式起点随折后移。"""
    roll = walkforward_splits(2000, n_splits=4, train_days=600, test_days=200, anchored=False)
    anch = walkforward_splits(2000, n_splits=4, train_days=600, test_days=200, anchored=True)
    assert all(s.train_start == 0 for s in anch)
    assert len({s.train_start for s in roll}) > 1
    assert all(s.train_end - s.train_start == 600 for s in roll)


def test_walkforward_rejects_insufficient_data():
    with pytest.raises(ValueError):
        walkforward_splits(100, n_splits=5, train_days=750, test_days=250)


def test_evaluate_selection_finds_predictive_ranking():
    """构造"样本内排名确实能预测样本外"的数据，秩相关应为正且高。

    这验证 evaluate_selection 本身没有写反 —— 若把秩相关的符号搞反，
    真实的预测力会被报成"没有预测力"。
    """
    rng = np.random.default_rng(3)
    n, k = 1500, 6
    idx = pd.bdate_range("2018-01-01", periods=n)
    # 每个策略有固定的真实均值，样本内外都由它驱动 → 排名应当稳定。
    # 均值间距必须明显大于样本均值的标准误 0.01/sqrt(500) ≈ 0.00045，
    # 否则排名本身就是噪声，测的就不是「有没有写反」而是运气。
    true_mu = np.array([0.0040, 0.0025, 0.0010, 0.0, -0.0015, -0.0030])
    cols = {}
    for j in range(k):
        cols[f"s{j}"] = rng.normal(true_mu[j], 0.01, n)
    ret = pd.DataFrame(cols, index=idx)

    splits = walkforward_splits(n, n_splits=3, train_days=500, test_days=250)
    folds = evaluate_selection(ret, splits)
    assert not folds.empty
    assert folds["rank_ic"].mean() > 0.5, "样本内排名应能预测样本外排名"
    assert (folds["rank_ic"] > 0).all(), "每一折的秩相关都应为正"
    assert (folds["best_in_sample"] == "s0").all()
    assert (folds["best_test_sharpe"] > folds["test_sharpe_of_median"]).all()


# ---------------------------------------------------------------- 多重检验


def test_benjamini_hochberg_matches_known_example():
    """BH 的标准算例，手算可验证。

    阈值是 p_(k) ≤ k/N·α = 0.005k：
        k=1 → 0.005 ≥ 0.001 ✓      k=2 → 0.010 ≥ 0.008 ✓
        k=3 → 0.015 < 0.039 ✗      k≥4 均不满足
    故最大的 k 为 2，只有前两个 p 值被拒绝。
    """
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    reject, adj = benjamini_hochberg(p, alpha=0.05)
    assert reject[:2].all()
    assert not reject[2:].any()
    assert np.all(np.diff(adj) >= -1e-12), "校正后的 p 值必须单调"
    # 与 statsmodels 的 fdr_bh 逐一对拍（见 README 的验证说明）
    from statsmodels.stats.multitest import multipletests

    rej_ref, adj_ref, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(adj, adj_ref, atol=1e-12)
    assert (reject == rej_ref).all()


def test_benjamini_hochberg_is_less_conservative_than_bonferroni():
    p = np.array([0.001, 0.01, 0.02, 0.03, 0.5, 0.7, 0.9])
    assert benjamini_hochberg(p, 0.05)[0].sum() >= bonferroni(p, 0.05).sum()


def test_bonferroni_threshold():
    p = np.array([0.001, 0.02, 0.03, 0.04])
    # 阈值 0.05/4 = 0.0125，只有 0.001 通过
    assert bonferroni(p, 0.05).tolist() == [True, False, False, False]
