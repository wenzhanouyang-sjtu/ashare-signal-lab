"""回测引擎。

设计原则
--------
1. **无未来函数**：t 日收盘产生的目标权重，在 t+lag 日开盘执行。
   引擎内部通过 shift(signal_lag) 强制这一点，调用方无法绕过。
2. **可交易性约束**：涨跌停封板、停牌时委托被拒绝 —— 不是"假设能成交
   再打个折扣"，而是直接不成交，并单独统计被拒绝的委托数。
3. **T+1**：当日买入的股票当日不可卖出。在本引擎"每日一次开盘净额调仓"
   的框架下该约束永不 binding（不存在同日两向成交），详见循环内注释。
4. **成本与收益分离**：维护两条账本，实际账本扣成本、毛账本不扣，
   两者执行完全相同的交易。于是 毛收益 − 净收益 精确等于成本，
   可以干净地回答"这个策略是被成本吃掉的，还是本来就没边际"。

记账单位为"元"，初始净值 1.0。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signal_lab.backtest.costs import CostModel


@dataclass
class BacktestResult:
    """回测输出。所有序列均以交易日为索引。"""

    equity: pd.Series              # 实际净值曲线（已扣成本）
    gross_equity: pd.Series        # 毛净值曲线（不扣成本）
    returns: pd.Series             # 日净收益
    gross_returns: pd.Series       # 日毛收益
    turnover: pd.Series            # 日单边换手率（成交金额 / 净值）
    cost_series: pd.Series         # 日成本（占净值比例）
    cost_abs: pd.Series            # 日成本（元，相对初始净值 1.0）
    weights: pd.DataFrame          # 每日收盘后的持仓权重
    n_trades: int = 0              # 成交笔数
    n_rejected: int = 0            # 因涨跌停/停牌被拒绝的委托笔数
    meta: dict = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """累计成本占初始净值的比例（求和口径，用于快速比较）。"""
        return float(self.cost_series.sum())

    @property
    def total_cost_abs(self) -> float:
        """累计成本的绝对金额。

        恒等式：gross_equity[-1] - equity[-1] == total_cost_abs
        （两条账本持仓完全相同，唯一差异就是已支付的费用）
        该恒等式是引擎正确性的核心断言，见 tests/test_engine.py。
        """
        return float(self.cost_abs.sum())

    @property
    def annual_turnover(self) -> float:
        """年化单边换手率。"""
        n_days = max(len(self.turnover), 1)
        return float(self.turnover.sum()) * 252.0 / n_days


def run_backtest(
    target_weights: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    cfg,
) -> BacktestResult:
    """执行回测。

    Parameters
    ----------
    target_weights
        目标权重矩阵 (date × code)，索引为 **信号生成日**（收盘后）。
        引擎按 cfg.backtest.execution.signal_lag 自动后移执行。
    panels
        clean.build_panels() 的输出。需含 open / close，
        可选 amount / can_buy / can_sell。
    cfg
        全局配置。
    """
    open_px = panels["open"]
    close_px = panels["close"]
    amount = panels.get("amount")
    can_buy = panels.get("can_buy")
    can_sell = panels.get("can_sell")

    dates = close_px.index
    codes = close_px.columns
    n_days, n_codes = len(dates), len(codes)

    lag = int(cfg.backtest.execution.signal_lag)
    t_plus_1 = bool(cfg.backtest.execution.t_plus_1)
    cost_model = CostModel.from_config(cfg)

    # 强制信号滞后：t 日收盘产生的信号，在 t+lag 日开盘执行
    exec_weights = (
        target_weights.reindex(index=dates, columns=codes).astype(float).shift(lag)
    )

    # 停牌期间价格缺失，估值时用前值结转（持仓被冻结，不产生盈亏）
    open_v = open_px.ffill().to_numpy()
    close_v = close_px.ffill().to_numpy()
    amount_v = amount.to_numpy() if amount is not None else None
    can_buy_v = can_buy.fillna(False).astype(bool).to_numpy() if can_buy is not None else None
    can_sell_v = can_sell.fillna(False).astype(bool).to_numpy() if can_sell is not None else None
    w_target = exec_weights.to_numpy()

    shares = np.zeros(n_codes)
    cash = 1.0          # 实际账本
    g_cash = 1.0        # 毛账本（不扣成本）

    equity_arr = np.zeros(n_days)
    gross_arr = np.zeros(n_days)
    turnover_arr = np.zeros(n_days)
    cost_arr = np.zeros(n_days)
    cost_abs_arr = np.zeros(n_days)
    weight_arr = np.zeros((n_days, n_codes))

    n_trades = 0
    n_rejected = 0

    for i in range(n_days):
        px_open = open_v[i]
        valid = np.isfinite(px_open) & (px_open > 0)

        # ---------------------------------------------------------- 开盘调仓
        if np.any(np.isfinite(w_target[i])) and np.any(valid):
            equity_open = cash + float(np.sum(shares * np.where(valid, px_open, 0.0)))

            w = np.nan_to_num(w_target[i], nan=0.0)
            target_value = w * equity_open
            target_shares = np.where(valid, target_value / np.where(valid, px_open, 1.0), 0.0)

            # 不可交易的股票保持原有持仓（委托被拒绝）
            delta = target_shares - shares
            blocked = ~valid.copy()
            if can_buy_v is not None:
                blocked |= (delta > 0) & ~can_buy_v[i]
            if can_sell_v is not None:
                blocked |= (delta < 0) & ~can_sell_v[i]
            n_rejected += int(np.sum(blocked & (np.abs(delta) > 1e-9)))
            delta = np.where(blocked, 0.0, delta)

            # 无交易带：偏离目标不足 min_trade_weight 的委托直接跳过。
            # 这不是"拒绝"，而是主动不做 —— 否则维持精确仓位会持续制造
            # 微小交易，把换手率与成本推高到失真。
            min_w = float(cfg.backtest.execution.get("min_trade_weight", 0.0) or 0.0)
            if min_w > 0 and equity_open > 0:
                trade_w = np.abs(delta) * np.where(valid, px_open, 0.0) / equity_open
                delta = np.where(trade_w < min_w, 0.0, delta)

            # T+1：当日买入的股票当日不可卖出。
            #
            # 本引擎每日只在开盘执行 **一次净额调仓**：对任一股票，delta 非买
            # 即卖，不存在同日先买后卖。于是"当日买入的股数"在执行前恒为零，
            # 这条约束在本框架下 **数学上永不 binding** —— 这是框架的推论，
            # 不是省略。cfg 里的 t_plus_1 因此只作为口径记录写入 result.meta。
            #
            # ⚠ 此处原先的实现是 `sellable = shares`，以当前持仓数封顶卖出量。
            # 那不是 T+1，而是 **禁止卖空**：持空头时 shares ≤ 0，任何卖出都被
            # 清零，配对交易的空腿永远建不起来 —— t_plus_1 一开一关会跑出两套
            # 完全不同的结果，而配置里它默认是 true。已修正。
            # 回归测试：tests/test_engine.py::test_t_plus_1_flag_does_not_block_shorts
            #
            # 若未来改为日内多批次调仓，必须在此处用真正的 bought_today 掩码
            # 实现该约束，而不是恢复旧写法。
            _ = t_plus_1

            traded_value = np.abs(delta) * np.where(valid, px_open, 0.0)
            active = traded_value > 1e-12

            if np.any(active):
                total_cost = 0.0
                for k in np.nonzero(active)[0]:
                    amt = None
                    if amount_v is not None:
                        a = amount_v[i, k]
                        amt = float(a) if np.isfinite(a) else None
                    total_cost += cost_model.cost_of(
                        traded_value[k], is_sell=bool(delta[k] < 0), daily_amount=amt
                    )

                cash -= float(np.sum(delta * np.where(valid, px_open, 0.0)))
                g_cash -= float(np.sum(delta * np.where(valid, px_open, 0.0)))
                cash -= total_cost
                shares = shares + delta

                n_trades += int(active.sum())
                turnover_arr[i] = float(traded_value.sum()) / max(equity_open, 1e-9)
                cost_arr[i] = total_cost / max(equity_open, 1e-9)
                cost_abs_arr[i] = total_cost

        # ---------------------------------------------------------- 收盘估值
        px_close = np.where(np.isfinite(close_v[i]), close_v[i], 0.0)
        holdings_value = float(np.sum(shares * px_close))
        equity_arr[i] = cash + holdings_value
        gross_arr[i] = g_cash + holdings_value

        denom = equity_arr[i] if equity_arr[i] > 1e-9 else 1.0
        weight_arr[i] = shares * px_close / denom

    equity = pd.Series(equity_arr, index=dates, name="equity")
    gross_equity = pd.Series(gross_arr, index=dates, name="gross_equity")
    returns = equity.pct_change().fillna(0.0)
    gross_returns = gross_equity.pct_change().fillna(0.0)

    return BacktestResult(
        equity=equity,
        gross_equity=gross_equity,
        returns=returns,
        gross_returns=gross_returns,
        turnover=pd.Series(turnover_arr, index=dates, name="turnover"),
        cost_series=pd.Series(cost_arr, index=dates, name="cost"),
        cost_abs=pd.Series(cost_abs_arr, index=dates, name="cost_abs"),
        weights=pd.DataFrame(weight_arr, index=dates, columns=codes),
        n_trades=n_trades,
        n_rejected=n_rejected,
        meta={
            "signal_lag": lag,
            "t_plus_1": t_plus_1,
            "round_trip_bp": round(cost_model.round_trip_bp(), 2),
        },
    )
