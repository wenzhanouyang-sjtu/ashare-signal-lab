"""A股交易成本模型。

多数个人回测把成本设成一个拍脑袋的固定 bp，这是失真的主要来源之一。
这里按 A 股实际的费用结构拆开计算：

  佣金      双边，万 2.5（券商默认档，机构更低）
  印花税    仅卖出单边，2023-08-28 起由 0.1% 降至 0.05%
  过户费    双边，十万分之一（沪市原按面值收，现统一按成交金额）
  冲击成本  与参与率（成交额占当日总成交额之比）非线性相关

冲击成本采用平方根律：cost ∝ 系数 × sqrt(参与率) × 成交金额。
这是业内常用的近似（Barra / Almgren 类模型的一阶简化），
比固定 bp 更接近实际 —— 大额交易在小盘股上的冲击远超标称费率。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    """把 config.yaml 里的费率参数封装成可调用的成本函数。"""

    commission: float = 0.00025
    stamp_tax: float = 0.0005
    transfer_fee: float = 0.00001
    impact_coef: float = 0.10
    impact_model: str = "sqrt"

    @classmethod
    def from_config(cls, cfg) -> "CostModel":
        c = cfg.backtest.costs
        return cls(
            commission=float(c.commission),
            stamp_tax=float(c.stamp_tax),
            transfer_fee=float(c.transfer_fee),
            impact_coef=float(c.impact_coef),
            impact_model=str(c.impact_model),
        )

    def fixed_rate(self, is_sell: bool) -> float:
        """与成交金额成正比的费率部分（佣金 + 过户费 + 卖出印花税）。"""
        rate = self.commission + self.transfer_fee
        if is_sell:
            rate += self.stamp_tax
        return rate

    def impact_rate(self, value: float, daily_amount: float | None) -> float:
        """冲击成本率（占成交金额的比例）。

        daily_amount 缺失或为 0 时，退回到只用固定费率 ——
        不臆造冲击，因为无成交量本身意味着该笔交易不该发生。
        """
        if not daily_amount or daily_amount <= 0 or value <= 0:
            return 0.0
        participation = min(value / daily_amount, 1.0)
        if self.impact_model == "sqrt":
            return self.impact_coef * np.sqrt(participation)
        return self.impact_coef * participation

    def cost_of(
        self,
        value: float,
        *,
        is_sell: bool,
        daily_amount: float | None = None,
    ) -> float:
        """单笔交易成本（元）。value 为成交金额，取正数。"""
        value = abs(float(value))
        if value == 0:
            return 0.0
        return value * (self.fixed_rate(is_sell) + self.impact_rate(value, daily_amount))

    def round_trip_bp(self) -> float:
        """一买一卖的固定费率（基点），用于快速估算成本量级。"""
        return (self.fixed_rate(False) + self.fixed_rate(True)) * 1e4
