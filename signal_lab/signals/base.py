"""信号层的统一接口与登记表。

设计意图：把"信号参数网格"变成可枚举的对象。每个 (信号族, 参数组合)
是一个 SignalSpec，全部枚举出来的数量就是多重检验的 **试验次数** ——
这是 FDR 校正与 Deflated Sharpe 的输入，必须显式可数。

如果试验次数不可数，就无法判断一个漂亮的夏普是真实的还是搜出来的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

# 信号函数签名：(close, **params) -> 分数矩阵
ScoreFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class SignalSpec:
    """一个具体的 (信号族, 参数组合)。"""

    family: str
    params: dict
    fn: ScoreFn
    weight_builder: str = "cross_sectional"  # 权重构建方式

    @property
    def name(self) -> str:
        """可读标识，例如 'reversal[lookback=5]'。"""
        if not self.params:
            return self.family
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.family}[{inner}]"

    def scores(self, close: pd.DataFrame) -> pd.DataFrame:
        return self.fn(close, **self.params)


# 信号族 -> (函数, 参数网格的来源键, 权重构建方式)
SIGNAL_REGISTRY: dict[str, dict] = {}


def register(
    family: str,
    fn: ScoreFn,
    *,
    weight_builder: str = "cross_sectional",
    description: str = "",
) -> None:
    SIGNAL_REGISTRY[family] = {
        "fn": fn,
        "weight_builder": weight_builder,
        "description": description,
    }


def enumerate_specs(cfg) -> list[SignalSpec]:
    """把 config.yaml 中启用的信号族展开成全部参数组合。

    返回列表的长度即多重检验的试验次数，会被写入结果文件。
    """
    # 延迟导入，避免循环依赖
    from signal_lab.signals import momentum, pairs, reversal, volatility

    _ensure_registered(momentum, pairs, reversal, volatility)

    specs: list[SignalSpec] = []
    sig_cfg = cfg.signals

    for family, entry in SIGNAL_REGISTRY.items():
        fam_cfg = sig_cfg.get(family)
        if not fam_cfg or not fam_cfg.get("enabled", False):
            continue

        grid = _param_grid(family, fam_cfg)
        for params in grid:
            specs.append(
                SignalSpec(
                    family=family,
                    params=params,
                    fn=entry["fn"],
                    weight_builder=entry["weight_builder"],
                )
            )
    return specs


def _param_grid(family: str, fam_cfg) -> list[dict]:
    """把 *_grid 形式的配置展开成笛卡尔积；无网格则返回单组参数。"""
    import itertools

    keys = [k for k in fam_cfg if k.endswith("_grid")]
    if not keys:
        return [{}]

    names = [k[: -len("_grid")] for k in keys]
    value_lists = [list(fam_cfg[k]) for k in keys]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def _ensure_registered(*modules) -> None:
    if SIGNAL_REGISTRY:
        return
    for mod in modules:
        mod.register_all()


def build_scores(spec: SignalSpec, close: pd.DataFrame) -> pd.DataFrame:
    return spec.scores(close)
