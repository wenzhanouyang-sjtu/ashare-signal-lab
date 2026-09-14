#!/usr/bin/env python3
"""步骤 3：验证层 —— 判断漂亮的回测结果里有多少是运气。

四道关卡，各针对一类具体的自欺方式：

    置换检验     把持仓与收益的时间对齐关系打乱，还能有这个夏普吗？
    Walk-forward 在历史上挑最好的，在未来是否仍然最好？
    FDR 校正     N 次检验中控制假阳性比例
    Deflated SR  "试了很多次之后，最好那次"本身就偏高

用法::

    python scripts/03_validate.py                 # 全量
    python scripts/03_validate.py --permutations 200   # 快速试跑

输出::

    results/validation.csv          每个信号的 p 值、FDR、DSR
    results/walkforward.csv         逐折样本内外表现
    results/null_distributions.parquet  置换零分布（供画图）
    results/validation_summary.json 汇总结论
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_lab.config import load_config  # noqa: E402
from signal_lab.pipeline import (  # noqa: E402
    default_rebalance_dates,
    load_panels,
    run_spec,
)
from signal_lab.signals.base import enumerate_specs  # noqa: E402
from signal_lab.validate.deflated_sharpe import deflated_sharpe_ratio  # noqa: E402
from signal_lab.validate.multiple_testing import benjamini_hochberg, bonferroni  # noqa: E402
from signal_lab.validate.permutation import permutation_test  # noqa: E402
from signal_lab.validate.walkforward import (  # noqa: E402
    evaluate_selection,
    summarize_selection,
    walkforward_splits,
)

TRADING_DAYS = 252


def main() -> int:
    ap = argparse.ArgumentParser(description="验证层：置换检验 + 样本外 + 多重检验校正")
    ap.add_argument("--permutations", type=int, default=None, help="覆盖置换次数")
    ap.add_argument("--skip-permutation", action="store_true", help="跳过置换检验（快速）")
    args = ap.parse_args()

    cfg = load_config()
    panels = load_panels(cfg)
    close = panels["close"]
    reb = default_rebalance_dates(panels, cfg)

    n_perm = args.permutations or int(cfg.validation.permutation.n_permutations)
    block = int(cfg.validation.permutation.block_size)
    alpha = float(cfg.validation.multiple_testing.alpha)

    specs = enumerate_specs(cfg)
    n_trials = len(specs)

    print(f"数据: {close.shape[1]} 只 × {close.shape[0]} 个交易日")
    print(f"试验次数 N = {n_trials}（多重检验校正的输入）")
    print(f"置换检验: 位移池 {close.shape[0] // block} 个，上限 {n_perm} 次，块长 {block}\n")


    rows: list[dict] = []
    net_returns: dict[str, pd.Series] = {}
    nulls: dict[str, np.ndarray] = {}
    n_perm_actual = 0          # 实际执行的置换次数（位移穷举，可能少于上限）
    n_shifts_pool = 0          # 位移池大小，决定 p 值分辨率下限

    for i, spec in enumerate(specs, 1):
        t0 = time.time()
        try:
            run = run_spec(spec, panels, cfg, reb, name=spec.name)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:>3}/{n_trials}] {spec.name:<34} 失败: {type(exc).__name__}: {exc}")
            continue

        net_returns[spec.name] = run.net

        rec = {
            "name": spec.name,
            "family": spec.family,
            "sharpe_net": float(run.stats["sharpe"]),
            "sharpe_gross": float(run.stats["sharpe_gross"]),
            "annual_turnover": float(run.stats["annual_turnover"]),
            "total_cost": float(run.stats["total_cost"]),
            "n_obs": int(len(run.net)),
        }

        if not args.skip_permutation:
            pt = permutation_test(
                run.target_weights, panels, cfg,
                n_permutations=n_perm, block_size=block,
            )
            rec.update(
                {
                    "sharpe_perm_obs": pt["sharpe_obs"],
                    "sharpe_perm_obs_net": pt["sharpe_obs_net"],
                    "perm_null_mean": pt["sharpe_null_mean"],
                    "perm_null_std": pt["sharpe_null_std"],
                    "p_value": pt["p_value"],
                }
            )
            nulls[spec.name] = pt["null_distribution"]
            n_perm_actual = max(n_perm_actual, int(pt["n_permutations"]))
            n_shifts_pool = max(n_shifts_pool, int(pt["n_distinct_shifts"]))

        rows.append(rec)

        extra = ""
        if "p_value" in rec:
            extra = f" | 置换 p {rec['p_value']:.3f}"
        print(
            f"[{i:>3}/{n_trials}] {spec.name:<34} "
            f"净夏普 {rec['sharpe_net']:>+6.2f} | 毛夏普 {rec['sharpe_gross']:>+6.2f}{extra} "
            f"({time.time() - t0:.0f}s)"
        )

    if not rows:
        raise SystemExit("所有信号均失败")

    table = pd.DataFrame(rows)

    # ---------------------------------------------------------- 多重检验校正
    if "p_value" in table.columns:
        p = table["p_value"].to_numpy(dtype=float)
        reject_bh, adj_bh = benjamini_hochberg(p, alpha)
        table["p_adjusted_bh"] = adj_bh
        table["significant_bh"] = reject_bh
        table["significant_bonferroni"] = bonferroni(p, alpha)

        # Deflated Sharpe：用全部试验的夏普估计跨试验方差，再对"试了 N 次"罚一次
        # 注意 DSR 要求日频（非年化）夏普
        trial_sharpes = table["sharpe_net"].to_numpy(dtype=float) / np.sqrt(TRADING_DAYS)
        dsr_rows = []
        for _, r in table.iterrows():
            dsr = deflated_sharpe_ratio(
                net_returns[r["name"]].to_numpy(dtype=float),
                trial_sharpes,
                n_trials=n_trials,
            )
            dsr_rows.append(dsr)
        dsr_df = pd.DataFrame(dsr_rows)
        table["sr0_annual"] = dsr_df["sr0_annual"]
        table["dsr"] = dsr_df["dsr"]

    table = table.sort_values("sharpe_net", ascending=False).reset_index(drop=True)

    results_dir = cfg.path("report", "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "validation.csv", index=False, encoding="utf-8-sig")

    if nulls:
        pd.DataFrame(nulls).to_parquet(results_dir / "null_distributions.parquet")

    # ---------------------------------------------------------- Walk-forward
    ret_df = pd.DataFrame(net_returns)
    wf_cfg = cfg.validation.walkforward
    splits = walkforward_splits(
        len(ret_df),
        n_splits=int(wf_cfg.n_splits),
        train_days=int(wf_cfg.train_days),
        test_days=int(wf_cfg.test_days),
        anchored=bool(wf_cfg.anchored),
    )
    folds = evaluate_selection(ret_df, splits)
    wf_summary = summarize_selection(folds)
    folds.to_csv(results_dir / "walkforward.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- 汇总结论
    summary = {
        "n_trials": n_trials,
        "n_signals_backtested": int(len(table)),
        "n_permutations": 0 if args.skip_permutation else n_perm_actual,
        "n_permutations_requested": 0 if args.skip_permutation else n_perm,
        "n_distinct_shifts": n_shifts_pool,
        "p_value_resolution": (1.0 / (n_shifts_pool + 1)) if n_shifts_pool else None,
        "alpha": alpha,
        "n_net_sharpe_positive": int((table["sharpe_net"] > 0).sum()),
        "n_gross_sharpe_positive": int((table["sharpe_gross"] > 0).sum()),
        "best_signal": table.iloc[0]["name"],
        "best_sharpe_net": float(table.iloc[0]["sharpe_net"]),
        "best_sharpe_gross": float(table.iloc[0]["sharpe_gross"]),
        "best_turnover": float(table.iloc[0]["annual_turnover"]),
        "best_total_cost": float(table.iloc[0]["total_cost"]),
        "walkforward": wf_summary,
    }
    if "significant_bh" in table.columns:
        summary.update(
            {
                "n_significant_raw_p05": int((table["p_value"] <= alpha).sum()),
                "n_significant_bh": int(table["significant_bh"].sum()),
                "n_significant_bonferroni": int(table["significant_bonferroni"].sum()),
                "n_dsr_above_095": int((table["dsr"] > 0.95).sum()),
                "best_dsr": float(table.iloc[0]["dsr"]),
                "best_sr0_annual": float(table.iloc[0]["sr0_annual"]),
            }
        )

    (results_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------------------------------------------------- 打印
    print(f"\n{'=' * 96}")
    print("验证结果（按净夏普排序）")
    print(f"{'=' * 96}")
    cols = ["name", "sharpe_net", "sharpe_gross", "annual_turnover", "total_cost"]
    if "p_value" in table.columns:
        cols += ["p_value", "p_adjusted_bh", "dsr"]
    print(table[cols].to_string(index=False))

    print(f"\n{'=' * 96}")
    print("多重检验校正")
    print(f"{'=' * 96}")
    if "significant_bh" in table.columns:
        print(f"试验次数 N                        : {n_trials}")
        print(f"原始 p ≤ {alpha} 显著              : {summary['n_significant_raw_p05']} / {len(table)}")
        print(f"Benjamini-Hochberg FDR 校正后显著 : {summary['n_significant_bh']} / {len(table)}")
        print(f"Bonferroni 校正后显著             : {summary['n_significant_bonferroni']} / {len(table)}")
        print(f"Deflated Sharpe > 0.95            : {summary['n_dsr_above_095']} / {len(table)}")
        print(f"最优信号的 SR₀（N 次试验的噪声上限）: 年化 {summary['best_sr0_annual']:+.3f}")
        print(f"最优信号的实际净夏普               : {summary['best_sharpe_net']:+.3f}")
        verdict = (
            "通过 —— 超过噪声上限"
            if summary["best_sharpe_net"] > summary["best_sr0_annual"]
            else "未通过 —— 落在 N 次试验的噪声范围内"
        )
        print(f"结论                              : {verdict}")

    print(f"\n{'=' * 96}")
    print("Walk-forward 样本外检验")
    print(f"{'=' * 96}")
    if not folds.empty:
        print(
            folds[
                ["fold", "best_in_sample", "best_train_sharpe", "best_test_sharpe", "rank_ic"]
            ].to_string(index=False)
        )
        print(
            f"\n样本内最优夏普均值 {wf_summary['mean_best_train_sharpe']:+.3f}"
            f"  →  样本外 {wf_summary['mean_best_test_sharpe']:+.3f}"
            f"  （衰减 {wf_summary['sharpe_decay']:+.3f}）"
        )
        print(f"样本内最优 vs 样本内中位数策略的样本外夏普: "
              f"{wf_summary['mean_best_test_sharpe']:+.3f} vs "
              f"{wf_summary['mean_test_sharpe_of_median']:+.3f}")
        print(f"样本内排名 → 样本外排名的秩相关（均值）: {wf_summary['mean_rank_ic']:+.3f}"
              f"  （{wf_summary['n_folds_positive_rank_ic']}/{wf_summary['n_folds']} 折为正）")

    print(f"\n结果已写入 {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
