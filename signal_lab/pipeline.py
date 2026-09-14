"""回测主流程：把「信号 → 目标权重 → 回测 → 统计」串成一条可复用的流水线。

为什么单独抽一层
----------------
`scripts/02_run_signals.py` 与 `scripts/03_validate.py` 需要 **完全相同** 的
权重构建逻辑。若两处各写一份，验证层就可能在校验一个与回测不同的策略 ——
这是最隐蔽也最致命的一类错误（验证看起来通过了，但对象不是被发现的那个）。

因此权重构建、回测执行、统计口径全部集中在 `run_spec()` 里，脚本只负责
调度与展示。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from signal_lab.backtest.engine import BacktestResult, run_backtest
from signal_lab.backtest.portfolio import cross_sectional_weights
from signal_lab.backtest.schedule import rebalance_dates
from signal_lab.data.clean import build_panels
from signal_lab.report.metrics import performance_stats
from signal_lab.signals.base import SignalSpec


def load_panels(cfg) -> dict[str, pd.DataFrame]:
    """读取清洗后的价格数据并转成面板。"""
    path = cfg.path("data", "cache_dir") / "prices_clean.parquet"
    if not path.exists():
        raise SystemExit(f"找不到清洗后的数据: {path}\n请先运行 scripts/01_fetch_data.py")
    return build_panels(pd.read_parquet(path))


def build_target_weights(spec: SignalSpec, close: pd.DataFrame, cfg, reb_dates) -> pd.DataFrame:
    """把信号转成目标权重。两条路径：直接给权重（配对），或截面排序。"""
    if spec.weight_builder == "direct":
        # 非网格参数（窗口长度、top_k、出场阈值…）一律从 config.yaml 取，
        # 网格参数由 SignalSpec 提供 —— 保证 config.yaml 是唯一的参数来源。
        fam_cfg = dict(cfg.signals.get(spec.family, {}))
        base = {
            k: v
            for k, v in fam_cfg.items()
            if not k.endswith("_grid") and k not in ("enabled", "description")
        }
        base.setdefault("gross_exposure", float(cfg.backtest.gross_exposure))
        base.update(spec.params)
        return spec.fn(close, **base, rebalance_dates=reb_dates)

    scores = spec.scores(close)
    # 只在调仓日留下信号，其余置 NaN —— 引擎遇到全 NaN 行不交易，即维持持仓
    mask = pd.Series(False, index=close.index)
    mask.loc[mask.index.intersection(reb_dates)] = True
    scores = scores.where(mask)

    return cross_sectional_weights(
        scores,
        quantile=float(cfg.backtest.quantile),
        gross_exposure=float(cfg.backtest.gross_exposure),
        min_names=int(cfg.backtest.min_names),
    )


@dataclass
class SpecRun:
    """一个信号组合的完整运行结果。"""

    spec: SignalSpec
    name: str
    weights: pd.DataFrame        # 引擎的 **实际每日持仓**（稠密，无 NaN）
    target_weights: pd.DataFrame # 信号给出的目标权重（仅调仓日有值，余为 NaN）
    result: BacktestResult
    net: pd.Series       # 从首次建仓日起算的净收益
    gross: pd.Series
    stats: dict

    @property
    def start(self) -> pd.Timestamp:
        return self.net.index[0]


def run_spec(
    spec: SignalSpec,
    panels: dict[str, pd.DataFrame],
    cfg,
    reb_dates,
    *,
    name: str | None = None,
) -> SpecRun:
    """构建权重、执行回测、汇总统计。

    统计从 **首次建仓日** 起算：信号需要预热窗口（如 250 日动量）才能产生
    第一个仓位，把预热期的零收益算进夏普会稀释分母、人为改变结论。
    """
    close = panels["close"]
    weights = build_target_weights(spec, close, cfg, reb_dates)
    res = run_backtest(weights, panels, cfg)

    active = res.weights.abs().sum(axis=1) > 1e-9
    start = active.idxmax() if active.any() else res.returns.index[0]
    net = res.returns.loc[start:]
    gross = res.gross_returns.loc[start:]

    stats = performance_stats(net, name=name or spec.name)
    stats_gross = performance_stats(gross, name=name or spec.name)
    stats.update(
        {
            "family": spec.family,
            "params": str(spec.params),
            "sharpe_gross": stats_gross["sharpe"],
            "annual_return_gross": stats_gross["annual_return"],
            "annual_turnover": float(
                res.turnover.loc[start:].sum() * 252 / max(len(net), 1)
            ),
            "total_cost": float(res.cost_abs.loc[start:].sum()),
            "n_trades": res.n_trades,
            "n_rejected": res.n_rejected,
        }
    )

    return SpecRun(
        spec=spec,
        name=name or spec.name,
        # 实际持仓取自引擎，而非目标权重：目标权重在非调仓日为 NaN（"维持"），
        # 若下游把它当成 0（"空仓"）就会严重低估敞口。任何基于持仓的
        # 事后分析（置换检验、归因、图表）都必须用实际持仓。
        weights=res.weights,
        target_weights=weights,
        result=res,
        net=net,
        gross=gross,
        stats=stats,
    )


def default_rebalance_dates(panels: dict[str, pd.DataFrame], cfg) -> pd.DatetimeIndex:
    return rebalance_dates(panels["close"].index, str(cfg.backtest.rebalance))


__all__ = [
    "SpecRun",
    "build_target_weights",
    "default_rebalance_dates",
    "load_panels",
    "run_spec",
]
