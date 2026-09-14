"""数据清洗与可交易性标记。

回测最常见的失真来源，是假设所有价格随时都能成交。A 股有三条硬约束：

  1. **涨跌停板** —— 封板时无法成交（买单排不上、卖单砸不动）
  2. **停牌**       —— 停牌期间完全不可交易，复牌常常跳空
  3. **T+1**       —— 当日买入的股票当日不可卖出

本模块负责把前两条标记成显式的布尔列，供回测引擎消费。T+1 约束在引擎层实现。

关于涨跌停幅度的判定：limit 价格按交易所规则四舍五入到 0.01 元，因此
"涨幅恰好 10%" 对低价股并不成立（例如 3.33 元涨停价是 3.66 元，涨幅 9.91%）。
这里直接按四舍五入后的涨跌停价比较，而不是比较百分比，避免误判。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.data.symbols import board_of

# 判定"封板"的容差（元）。涨跌停价保留两位小数，取一个远小于 0.01 的值即可。
_LIMIT_TOL = 1e-4


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """基础清洗：类型、排序、去重、异常值处理。

    不做前向填充 —— 停牌期间的价格缺失是有信息量的，由 mark_tradeable 标记。
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["code", "date"]).drop_duplicates(["code", "date"])

    # 负价格或零价格属于数据错误，置为缺失
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out.loc[out[col] <= 0, col] = np.nan

    # high/low 与 close 的逻辑一致性校验（复权后偶有不一致，做温和修正）
    if {"high", "low", "close"}.issubset(out.columns):
        out["high"] = out[["high", "close", "open"]].max(axis=1)
        out["low"] = out[["low", "close", "open"]].min(axis=1)

    return out.reset_index(drop=True)


def limit_ratio(code: str, cfg) -> float:
    """按板块返回涨跌停幅度。"""
    board = board_of(code)
    limits = cfg.backtest.price_limits
    # 北交所 ±30% 未在配置中单列时退回 default
    return float(limits.get(board, limits.get("default", 0.10)))


def mark_tradeable(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """为每只股票标记可交易性。

    新增列：
        prev_close   前收盘价（停牌期间沿用最后成交价）
        suspended    停牌（已上市但当日无成交记录）
        limit_up     收盘封涨停（无法买入）
        limit_down   收盘封跌停（无法卖出）
        can_buy      次日开盘可否买入
        can_sell     次日开盘可否卖出
        ret          当日对数收益（停牌日为 0）

    关于停牌的表示 —— 容易踩的坑
    -----------------------------
    数据源在停牌日 **根本不返回该股票的行**，而不是返回一行成交量为 0 的记录。
    因此"成交量 == 0 即停牌"这条判据永远不会触发（实测 0 次命中），
    停牌会静默地表现成"这只股票那天不存在"，与"尚未上市"无法区分。

    正确做法是把面板补全到「已上市 × 全部交易日」的完整网格：
    某只股票从首次出现之日起视为已上市，此后凡是没有数据的交易日即为停牌。
    这样"未上市"（不该有仓位）与"停牌"（有仓位但卖不掉）才被分开，
    两者对回测的含义完全不同。

    补全后价格列为 NaN、涨跌停标记为 False、can_buy/can_sell 为 False，
    引擎据此外推：停牌期间持仓被冻结，复牌首日的收益自然包含跳空。
    """
    out = df.sort_values(["code", "date"]).copy()

    # ---- 补全网格：把「已上市但当日无数据」的日子显式补成停牌行
    present = pd.crosstab(out["date"], out["code"]) > 0     # (date, code) 是否有记录
    listed = present.cummax(axis=0)                          # 首次出现后一直为 True
    suspended_panel = listed & ~present

    if suspended_panel.to_numpy().any():
        full_index = pd.MultiIndex.from_product(
            [present.index, present.columns], names=["date", "code"]
        )
        target = listed.stack()                              # 只保留已上市的格子
        target = target[target].index
        out = (
            out.set_index(["date", "code"])
            .reindex(target)
            .reset_index()
        )
        out["code"] = out["code"].astype(str)

    g = out.groupby("code", group_keys=False)

    # 停牌日收益记为 0（持仓被冻结，不产生盈亏）；
    # 复牌首日的收益相对最后成交价计算，因此包含复牌跳空。
    out["prev_close"] = g["close"].ffill().groupby(out["code"]).shift(1)
    out["ret"] = (
        np.log(out["close"] / out["prev_close"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    sup = suspended_panel.stack()
    out["suspended"] = out.set_index(["date", "code"]).index.map(sup).fillna(False).to_numpy()

    # 涨跌停：按四舍五入后的实际涨跌停价比较，而不是比较百分比
    ratio = out["code"].astype(str).map(lambda c: limit_ratio(c, cfg))
    pc = out["prev_close"]
    limit_up_price = (pc * (1 + ratio)).round(2)
    limit_down_price = (pc * (1 - ratio)).round(2)
    has_prev = pc.notna() & (pc > 0)
    has_px = out["close"].notna()

    out["limit_up"] = has_prev & has_px & (out["close"] >= limit_up_price - _LIMIT_TOL)
    out["limit_down"] = has_prev & has_px & (out["close"] <= limit_down_price + _LIMIT_TOL)

    # 可交易性：停牌不可交易；封板时对应方向不可成交
    out["can_buy"] = ~out["suspended"] & ~out["limit_up"]
    out["can_sell"] = ~out["suspended"] & ~out["limit_down"]

    if not cfg.backtest.execution.limit_updown:
        out["can_buy"] = ~out["suspended"]
        out["can_sell"] = ~out["suspended"]
    if not cfg.backtest.execution.suspension:
        out["can_buy"] = True
        out["can_sell"] = True

    return out.sort_values(["date", "code"]).reset_index(drop=True)


def to_panel(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """长表转宽表：index=date, columns=code。

    缺失值 **不做填充** —— 保持 NaN，让下游明确区分"无数据"与"价格不变"。
    """
    panel = df.pivot_table(index="date", columns="code", values=field, aggfunc="last")
    return panel.sort_index()


_NUMERIC_FIELDS = ("open", "high", "low", "close", "volume", "amount", "turnover")
_BOOL_FIELDS = ("can_buy", "can_sell", "suspended")


def build_panels(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """一次性构建回测需要的所有面板。

    布尔面板必须显式转成 bool —— pivot 后缺失值（股票尚未上市或无数据）
    会把整列变成 float，导致引擎里的位运算报错，同时也会模糊
    "缺失" 与 "False" 的区别。这里缺失一律视为不可交易。
    """
    panels: dict[str, pd.DataFrame] = {}
    for field in _NUMERIC_FIELDS + _BOOL_FIELDS:
        if field not in df.columns:
            continue
        panel = to_panel(df, field)
        if field in _BOOL_FIELDS:
            panel = panel.fillna(False).astype(bool)
        panels[field] = panel
    return panels
