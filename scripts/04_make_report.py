#!/usr/bin/env python3
"""步骤 4：生成图表与结论摘要。

用法::

    python scripts/04_make_report.py

前置：已运行 01（数据）、02（回测）、03（验证）。

输出::

    results/figures/*.png     图表
    results/SUMMARY.md        自动生成的结论摘要（数字均来自实际结果文件）

为什么摘要要自动生成
--------------------
README 里的数字如果是手抄的，改一次参数就会过期，而且没人会发现。
这里把结论段落也变成产物：改配置 → 重跑 → 摘要自动更新。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_lab.backtest.costs import CostModel  # noqa: E402
from signal_lab.backtest.engine import run_backtest  # noqa: E402
from signal_lab.config import load_config  # noqa: E402
from signal_lab.pipeline import (  # noqa: E402
    build_target_weights,
    default_rebalance_dates,
    load_panels,
)
from signal_lab.report import figures  # noqa: E402
from signal_lab.report.metrics import performance_stats  # noqa: E402
from signal_lab.signals.base import enumerate_specs  # noqa: E402

TRADING_DAYS = 252


def _scaled_cfg(cfg, multiplier: float):
    """按倍数缩放全部费率参数，得到一条成本敏感性曲线。"""
    new = copy.deepcopy(cfg)
    costs = new["backtest"]["costs"]
    for key in ("commission", "stamp_tax", "transfer_fee", "impact_coef"):
        costs[key] = float(costs[key]) * multiplier
    return new


def cost_sensitivity(
    cfg,
    panels,
    reb,
    names: list[str],
    *,
    multipliers: list[float],
) -> dict[str, pd.DataFrame]:
    """在若干成本档位下重跑指定信号，得到净夏普随成本的变化。"""
    specs = {s.name: s for s in enumerate_specs(cfg)}
    out: dict[str, pd.DataFrame] = {}

    for nm in names:
        spec = specs.get(nm)
        if spec is None:
            continue
        weights = build_target_weights(spec, panels["close"], cfg, reb)
        rows = []
        for m in multipliers:
            c = _scaled_cfg(cfg, m)
            res = run_backtest(weights, panels, c)
            active = res.weights.abs().sum(axis=1) > 1e-9
            start = active.idxmax() if active.any() else res.returns.index[0]
            stats = performance_stats(res.returns.loc[start:], name=nm)
            rows.append(
                {
                    "cost_multiplier": m,
                    "round_trip_bp": CostModel.from_config(c).round_trip_bp(),
                    "sharpe": stats["sharpe"],
                }
            )
        out[nm] = pd.DataFrame(rows).set_index("round_trip_bp")
    return out


def equal_weight_benchmark(panels) -> pd.Series:
    """股票池等权买入持有的日收益。

    注意：股票池是 **当前** 成分股，存在幸存者偏差（见 README）。
    这里用它只是给策略一个量级参照，不是严格的业绩基准。
    """
    r = panels["close"].pct_change()
    return r.mean(axis=1, skipna=True).fillna(0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成图表与结论摘要")
    ap.add_argument("--skip-sensitivity", action="store_true", help="跳过成本敏感性扫描")
    args = ap.parse_args()

    cfg = load_config()
    results_dir = cfg.path("report", "results_dir")
    fig_dir = cfg.path("report", "figures_dir")

    summary = pd.read_csv(results_dir / "summary_all.csv")
    net = pd.read_parquet(results_dir / "daily_returns.parquet")
    gross = pd.read_parquet(results_dir / "daily_gross_returns.parquet")

    print(f"读取结果: {len(summary)} 个信号组合")

    panels = load_panels(cfg)
    reb = default_rebalance_dates(panels, cfg)
    bench = equal_weight_benchmark(panels)

    made: list[Path] = []
    made.append(figures.equity_curves(net, summary, bench, fig_dir))
    print(f"  ✓ {made[-1].name}")

    made.append(figures.gross_vs_net_sharpe(summary, fig_dir))
    print(f"  ✓ {made[-1].name}")

    made.append(figures.turnover_vs_sharpe(summary, fig_dir))
    print(f"  ✓ {made[-1].name}")

    made.append(figures.drawdown(net, summary, fig_dir))
    print(f"  ✓ {made[-1].name}")

    made.append(figures.cost_drag(net, gross, summary, fig_dir))
    print(f"  ✓ {made[-1].name}")

    # 参数热力图：对每个有 ≥2 个网格参数的族各画一张
    for fam in figures.FAMILY_ORDER:
        try:
            made.append(figures.parameter_heatmap(summary, fam, fig_dir))
            print(f"  ✓ {made[-1].name}")
        except (ValueError, KeyError) as exc:
            print(f"  – 跳过 {fam} 热力图: {exc}")

    # Walk-forward（需要 03 的产物）
    wf_path = results_dir / "walkforward.csv"
    if wf_path.exists():
        folds = pd.read_csv(wf_path)
        if not folds.empty:
            made.append(figures.walkforward(folds, fig_dir))
            print(f"  ✓ {made[-1].name}")

    # 置换零分布（需要 03 的产物）
    null_path = results_dir / "null_distributions.parquet"
    if null_path.exists():
        nulls = pd.read_parquet(null_path)
        made.append(figures.null_distributions(nulls, summary, fig_dir))
        print(f"  ✓ {made[-1].name}")

    # 成本敏感性
    sens = {}
    if not args.skip_sensitivity:
        picks = figures.best_per_family(summary)
        mults = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        print(f"\n成本敏感性扫描: {len(picks)} 个信号 × {len(mults)} 个成本档位")
        sens = cost_sensitivity(cfg, panels, reb, list(picks.values()), multipliers=mults)
        if sens:
            made.append(figures.cost_sensitivity(sens, fig_dir))
            print(f"  ✓ {made[-1].name}")

    # ------------------------------------------------------------ 摘要
    valid_path = results_dir / "validation.csv"
    validation = pd.read_csv(valid_path) if valid_path.exists() else None

    lines: list[str] = []
    lines.append("# 结果摘要（自动生成）\n")
    lines.append(
        "本文件由 `scripts/04_make_report.py` 自动生成，"
        "所有数字来自 `results/` 下的实际产物，请勿手工编辑。\n"
    )

    lines.append("\n## 样本与规模\n")
    lines.append(f"- 股票池：沪深300 成分股（当前成分，共 {panels['close'].shape[1]} 只）")
    lines.append(
        f"- 区间：{panels['close'].index[0].date()} ~ {panels['close'].index[-1].date()}"
        f"（{len(panels['close'])} 个交易日）"
    )
    lines.append(f"- 调仓频率：{cfg.backtest.rebalance}，共 {len(reb)} 个调仓日")
    lines.append(f"- 信号参数组合（= 多重检验的试验次数 N）：**{len(summary)}**")

    lines.append("\n## 全部结果（按净夏普排序）\n")
    cols = ["name", "sharpe", "sharpe_gross", "annual_return", "annual_turnover", "total_cost"]
    if validation is not None and "p_value" in validation.columns:
        merged = summary.merge(
            validation[["name", "p_value", "p_adjusted_bh", "dsr"]], on="name", how="left"
        )
        cols += ["p_value", "p_adjusted_bh", "dsr"]
    else:
        merged = summary
    lines.append(merged.sort_values("sharpe", ascending=False)[cols].to_markdown(index=False))

    lines.append("\n## 成本侵蚀\n")
    hi = summary[summary["annual_turnover"] > 40]
    lo = summary[summary["annual_turnover"] < 20]
    if not hi.empty:
        lines.append(
            f"- 高换手组（>40x/年，{len(hi)} 个）：年化换手 "
            f"{hi['annual_turnover'].mean():.1f}x，十年累计成本 "
            f"{hi['total_cost'].mean() * 100:.1f}%，"
            f"毛夏普均值 {hi['sharpe_gross'].mean():+.2f} → 净夏普均值 {hi['sharpe'].mean():+.2f}"
        )
    if not lo.empty:
        lines.append(
            f"- 低换手组（<20x/年，{len(lo)} 个）：年化换手 "
            f"{lo['annual_turnover'].mean():.1f}x，十年累计成本 "
            f"{lo['total_cost'].mean() * 100:.1f}%，"
            f"毛夏普均值 {lo['sharpe_gross'].mean():+.2f} → 净夏普均值 {lo['sharpe'].mean():+.2f}"
        )

    if sens:
        lines.append("\n各信号族的盈亏平衡成本（净夏普归零时的往返固定成本，不含冲击成本）：\n")
        rows = []
        for nm, df in sens.items():
            s = df["sharpe"].to_numpy(dtype=float)
            x = df.index.to_numpy(dtype=float)
            be = np.nan
            cross = np.nonzero(np.diff(np.sign(s)) != 0)[0]
            if len(cross):
                i = cross[0]
                if s[i + 1] != s[i]:
                    be = x[i] + (x[i + 1] - x[i]) * (0.0 - s[i]) / (s[i + 1] - s[i])
            # 没有盈亏平衡点有两种截然不同的原因，直接写出来。
            # 裸的 nan 会被读成"算错了"，而这两件事恰恰是本项目最想说明的。
            if not np.isnan(be):
                note = ""
            elif s[0] <= 0:
                note = "零成本下已为负 —— 不存在盈亏平衡点"
            else:
                note = f"扫描范围内始终为正（上限 {x[-1]:.1f}bp）"
            rows.append({"signal": nm, "break_even_round_trip_bp": be,
                         "sharpe_at_zero_cost": s[0], "说明": note})
        lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"))

    if validation is not None and "p_value" in validation.columns:
        lines.append("\n## 显著性检验\n")
        n = len(validation)
        lines.append(f"- 置换检验原始 p ≤ {cfg.validation.multiple_testing.alpha}："
                     f"{int((validation['p_value'] <= cfg.validation.multiple_testing.alpha).sum())} / {n}")
        lines.append(f"- Benjamini-Hochberg FDR 校正后显著：{int(validation['significant_bh'].sum())} / {n}")
        lines.append(f"- Bonferroni 校正后显著：{int(validation['significant_bonferroni'].sum())} / {n}")
        lines.append(f"- Deflated Sharpe > 0.95：{int((validation['dsr'] > 0.95).sum())} / {n}")

        # 置换检验的有效自由度：位移池大小决定 p 值分辨率下限。
        # 不写出这一行，读者会以为 p 值可以取到任意小。
        meta_path = results_dir / "validation_summary.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("n_distinct_shifts"):
                lines.append(
                    f"- 置换位移池：**{meta['n_distinct_shifts']}** 个不重复位移，"
                    f"p 值最小分辨率 {meta['p_value_resolution']:.4f}"
                    f"（穷举无放回，非「抽 1000 次」）"
                )

        best = validation.sort_values("sharpe_net", ascending=False).iloc[0]
        lines.append(
            f"\n最优信号 **{best['name']}**：净夏普 {best['sharpe_net']:+.3f}，"
            f"而 N={n} 次试验下的噪声上限 SR₀ = {best['sr0_annual']:+.3f}（年化）。"
        )
        lines.append(
            "→ 结论："
            + ("**超过**噪声上限。" if best["sharpe_net"] > best["sr0_annual"]
               else "**未超过**噪声上限 —— 这个夏普可以被「试了很多次」本身解释。")
        )

    wf_path = results_dir / "walkforward.csv"
    if wf_path.exists():
        folds = pd.read_csv(wf_path)
        if not folds.empty:
            lines.append("\n## Walk-forward 样本外\n")
            lines.append(folds.to_markdown(index=False, floatfmt=".3f"))
            lines.append(
                f"\n- 样本内最优夏普均值 {folds['best_train_sharpe'].mean():+.3f} "
                f"→ 样本外 {folds['best_test_sharpe'].mean():+.3f}"
            )
            lines.append(f"- 样本内排名 → 样本外排名的秩相关均值：{folds['rank_ic'].mean():+.3f}")

    lines.append("\n## 图表\n")
    for p in made:
        lines.append(f"- `figures/{p.name}`")

    (results_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n共生成 {len(made)} 张图 → {fig_dir}/")
    print(f"结论摘要 → {results_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
