#!/usr/bin/env python3
"""步骤 2：枚举全部信号参数组合，逐一回测。

用法::

    python scripts/02_run_signals.py                    # 全量
    python scripts/02_run_signals.py --family reversal  # 只跑某一族
    python scripts/02_run_signals.py --freq D           # 改用日频调仓（对照实验）

输出::

    results/daily_returns.parquet   每个信号组合的日净收益
    results/daily_gross_returns.parquet  毛收益（未扣成本）
    results/summary_all.csv         绩效汇总（含换手率与成本）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_lab.config import load_config  # noqa: E402
from signal_lab.pipeline import (  # noqa: E402
    default_rebalance_dates,
    load_panels,
    run_spec,
)
from signal_lab.signals.base import enumerate_specs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="运行全部信号回测")
    ap.add_argument("--family", default=None, help="只运行指定信号族")
    ap.add_argument("--freq", default=None, help="覆盖调仓频率 (D/W/M)")
    args = ap.parse_args()

    cfg = load_config()
    freq = args.freq or str(cfg.backtest.rebalance)

    panels = load_panels(cfg)
    close = panels["close"]
    reb = default_rebalance_dates(panels, cfg)

    print(f"数据: {close.shape[1]} 只 × {close.shape[0]} 个交易日")
    print(f"区间: {close.index[0].date()} ~ {close.index[-1].date()}")
    print(f"调仓频率: {freq} — 共 {len(reb)} 个调仓日\n")

    specs = enumerate_specs(cfg)
    if args.family:
        specs = [s for s in specs if s.family == args.family]
    if not specs:
        raise SystemExit("没有启用任何信号，请检查 config.yaml 的 signals 段")

    print(f"待检验信号组合: {len(specs)} 个")
    print("（该数字即多重检验的试验次数，将用于 FDR 校正与 Deflated Sharpe）\n")

    net_returns: dict[str, pd.Series] = {}
    gross_returns: dict[str, pd.Series] = {}
    rows: list[dict] = []

    for i, spec in enumerate(specs, 1):
        t0 = time.time()
        name = spec.name if not args.freq else f"{spec.name}|{freq}"
        try:
            run = run_spec(spec, panels, cfg, reb, name=name)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:>3}/{len(specs)}] {name:<34} 失败: {type(exc).__name__}: {exc}")
            continue

        net_returns[name] = run.net
        gross_returns[name] = run.gross
        stats = dict(run.stats)
        stats["runtime_s"] = round(time.time() - t0, 1)
        rows.append(stats)

        print(
            f"[{i:>3}/{len(specs)}] {name:<34} "
            f"净夏普 {stats['sharpe']:>+6.2f} | 毛夏普 {stats['sharpe_gross']:>+6.2f} | "
            f"换手 {stats['annual_turnover']:>6.1f}x | 成本 {stats['total_cost']:>6.2%} "
            f"({time.time() - t0:.0f}s)"
        )

    if not rows:
        raise SystemExit("所有信号均失败，请检查数据与配置")

    results_dir = cfg.path("report", "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    summary.to_csv(results_dir / "summary_all.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(net_returns).to_parquet(results_dir / "daily_returns.parquet")
    pd.DataFrame(gross_returns).to_parquet(results_dir / "daily_gross_returns.parquet")
    summary.to_json(
        results_dir / "summary_all.json", orient="records", force_ascii=False, indent=2
    )

    print(f"\n{'=' * 78}")
    print("净夏普排名前 10:")
    print(f"{'=' * 78}")
    cols = ["name", "sharpe", "sharpe_gross", "annual_return", "max_drawdown", "annual_turnover"]
    print(summary[cols].head(10).to_string(index=False))

    positive = int((summary["sharpe"] > 0).sum())
    print(f"\n净夏普 > 0 的组合: {positive}/{len(summary)}")
    print(f"结果已写入 {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
