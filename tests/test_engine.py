"""回测引擎正确性测试。

这些断言是项目可信度的基础 —— 如果引擎的记账本身是错的，
后面所有关于"信号是否有效"的结论都没有意义。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_lab.backtest.engine import run_backtest
from signal_lab.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_panels(dates, codes, price, *, can_buy=True, can_sell=True, amount=1e9):
    if isinstance(price, (int, float)):
        px = pd.DataFrame(float(price), index=dates, columns=codes)
    else:
        px = price
    return {
        "open": px,
        "close": px,
        "amount": pd.DataFrame(float(amount), index=dates, columns=codes),
        "can_buy": pd.DataFrame(can_buy, index=dates, columns=codes),
        "can_sell": pd.DataFrame(can_sell, index=dates, columns=codes),
    }


def test_cost_identity(cfg):
    """核心恒等式：毛净值 − 实际净值 == 累计成本。

    两条账本执行完全相同的交易，唯一差异是实际账本支付了费用。
    这个恒等式成立，才说明"成本"与"收益"是干净分离的。
    """
    dates = pd.bdate_range("2020-01-01", periods=40)
    codes = ["A", "B"]
    panels = make_panels(dates, codes, 10.0)

    weights = pd.DataFrame(0.0, index=dates, columns=codes)
    for i in range(len(dates)):
        weights.iloc[i, i % 2] = 1.0  # 每天在两个标的间来回切换，制造换手

    res = run_backtest(weights, panels, cfg)

    assert res.n_trades > 0, "应当发生了交易"
    assert res.gross_equity.iloc[-1] == pytest.approx(1.0, abs=1e-10), "价格恒定，毛净值应恒为 1"
    assert res.gross_equity.iloc[-1] - res.equity.iloc[-1] == pytest.approx(
        res.total_cost_abs, abs=1e-12
    )


def test_buy_and_hold(cfg):
    """买入持有：期末净值应等于 末日收盘 / 次日开盘（再扣买入成本）。"""
    dates = pd.bdate_range("2020-01-01", periods=30)
    px = pd.DataFrame({"A": np.linspace(10.0, 20.0, 30)}, index=dates)
    panels = make_panels(dates, ["A"], px)
    weights = pd.DataFrame(1.0, index=dates, columns=["A"])

    res = run_backtest(weights, panels, cfg)

    theoretical = 20.0 / px["A"].iloc[1]
    # 差额应恰好等于买入成本，量级在万分之几
    assert res.equity.iloc[-1] == pytest.approx(theoretical, rel=1e-3)
    assert res.equity.iloc[-1] < theoretical, "净值必须低于无成本理论值"


def test_no_lookahead(cfg):
    """未来函数检测：篡改 t 日之后的价格，不应改变 t 日之前的净值路径。

    这是引擎最关键的一条断言 —— 任何"偷看未来"的实现都会在这里暴露。
    """
    dates = pd.bdate_range("2020-01-01", periods=60)
    codes = ["A", "B"]
    rng = np.random.default_rng(0)
    px = pd.DataFrame(
        10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 2)), axis=0)),
        index=dates,
        columns=codes,
    )
    weights = pd.DataFrame(0.0, index=dates, columns=codes)
    for i in range(len(dates)):
        weights.iloc[i, i % 2] = 1.0

    base = run_backtest(weights, make_panels(dates, codes, px), cfg)

    # 把第 40 天之后的价格整体乘 3
    tampered = px.copy()
    tampered.iloc[40:] *= 3.0
    after = run_backtest(weights, make_panels(dates, codes, tampered), cfg)

    pd.testing.assert_series_equal(
        base.equity.iloc[:40], after.equity.iloc[:40], check_names=False
    )
    assert not np.allclose(base.equity.iloc[-1], after.equity.iloc[-1]), (
        "改动未来价格后最终净值应当变化，否则测试本身失效"
    )


def test_limit_up_blocks_buying(cfg):
    """封涨停时买不进 —— 不是打个折扣，而是完全不成交。"""
    dates = pd.bdate_range("2020-01-01", periods=30)
    px = pd.DataFrame({"A": np.linspace(10.0, 20.0, 30)}, index=dates)
    panels = make_panels(dates, ["A"], px, can_buy=False)
    weights = pd.DataFrame(1.0, index=dates, columns=["A"])

    res = run_backtest(weights, panels, cfg)

    assert res.equity.iloc[-1] == pytest.approx(1.0, abs=1e-12), "从未成交，净值应不变"
    assert res.n_rejected > 0, "应记录到被拒绝的委托"


def test_no_trade_when_target_unchanged(cfg):
    """目标权重不变时不应产生换手与成本。"""
    dates = pd.bdate_range("2020-01-01", periods=30)
    codes = ["A"]
    panels = make_panels(dates, codes, 10.0)
    weights = pd.DataFrame(1.0, index=dates, columns=codes)

    res = run_backtest(weights, panels, cfg)

    # 只在建仓那天有换手，之后持仓不动
    assert res.turnover.iloc[2:].sum() == pytest.approx(0.0, abs=1e-12)
    assert res.n_trades <= 2


def test_costs_reduce_net_returns(cfg):
    """有换手时，净收益必须严格低于毛收益。"""
    dates = pd.bdate_range("2020-01-01", periods=40)
    codes = ["A", "B"]
    rng = np.random.default_rng(1)
    px = pd.DataFrame(
        10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (40, 2)), axis=0)),
        index=dates,
        columns=codes,
    )
    weights = pd.DataFrame(0.0, index=dates, columns=codes)
    for i in range(len(dates)):
        weights.iloc[i, i % 2] = 1.0

    res = run_backtest(weights, make_panels(dates, codes, px), cfg)

    assert res.total_cost_abs > 0
    assert res.equity.iloc[-1] < res.gross_equity.iloc[-1]


def test_t_plus_1_flag_does_not_block_shorts(cfg):
    """回归测试：t_plus_1 不得阻止建立空头。

    曾经的实现用 `sellable = shares` 封顶卖出量。持空头时 shares ≤ 0，
    卖出量被清零，空腿永远建不起来 —— 于是 t_plus_1 这个 flag 实际控制的是
    "能不能做空"，而它在 config.yaml 里默认 true，等于把配对交易偷偷变成了
    单边多头。这个差异不会报错、不会警告，只会让夏普换个数字。

    这里显式关闭与开启 t_plus_1，断言两套配置结果完全一致，且都能建空头。
    """
    import copy

    dates = pd.bdate_range("2020-01-01", periods=20)
    codes = ["A", "B"]
    panels = make_panels(dates, codes, 10.0)

    weights = pd.DataFrame(0.0, index=dates, columns=codes)
    weights["A"] = -0.5          # 持续做空 A
    weights["B"] = 0.5

    cfg_off = copy.deepcopy(cfg)
    cfg_off.backtest.execution.t_plus_1 = False
    cfg_on = copy.deepcopy(cfg)
    cfg_on.backtest.execution.t_plus_1 = True

    res_off = run_backtest(weights, panels, cfg_off)
    res_on = run_backtest(weights, panels, cfg_on)

    assert res_on.weights["A"].min() < -0.1, "t_plus_1=True 时仍应能建立空头"
    pd.testing.assert_series_equal(res_off.equity, res_on.equity)
    assert res_on.meta["t_plus_1"] is True
