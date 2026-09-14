"""图表生成。

约定
----
* 图内文字用英文：不依赖中文字体，任何机器上都能正确渲染，
  也便于非中文读者阅读。中文说明放在 README。
* 每张图只回答一个问题。图题即结论。
* 全部函数返回保存路径，便于脚本汇总打印。

配色统一从 PALETTE 取，保证同一族在不同图里颜色一致。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境（CI / 服务器）也能出图

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRADING_DAYS = 252

PALETTE = {
    "pairs": "#1f77b4",
    "momentum": "#d62728",
    "reversal": "#2ca02c",
    "volatility": "#ff7f0e",
    "benchmark": "#7f7f7f",
    "gross": "#9ecae1",
    "net": "#08519c",
    "accent": "#d62728",
}

FAMILY_ORDER = ["pairs", "momentum", "volatility", "reversal"]
FAMILY_LABEL = {
    "pairs": "Pairs / stat-arb",
    "momentum": "Momentum",
    "reversal": "Reversal",
    "volatility": "Low volatility",
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
    }
)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _family_of(name: str) -> str:
    return name.split("[")[0]


def _annualized_sharpe(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def best_per_family(summary: pd.DataFrame) -> dict[str, str]:
    """每个信号族里净夏普最高的组合名。"""
    out = {}
    for fam in FAMILY_ORDER:
        sub = summary[summary["family"] == fam]
        if not sub.empty:
            out[fam] = sub.sort_values("sharpe", ascending=False).iloc[0]["name"]
    return out


# --------------------------------------------------------------------- 图 1


def equity_curves(
    net_returns: pd.DataFrame,
    summary: pd.DataFrame,
    benchmark: pd.Series,
    out_dir: Path,
) -> Path:
    """各信号族最优组合的累计净值，与等权基准对比。"""
    picks = best_per_family(summary)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for fam in FAMILY_ORDER:
        nm = picks.get(fam)
        if nm is None or nm not in net_returns.columns:
            continue
        eq = (1.0 + net_returns[nm].fillna(0.0)).cumprod()
        ax.plot(
            eq.index, eq.to_numpy(),
            color=PALETTE[fam], lw=1.5,
            label=f"{FAMILY_LABEL[fam]}  (Sharpe {summary.loc[summary['name'] == nm, 'sharpe'].iloc[0]:+.2f})",
        )

    if benchmark is not None and len(benchmark):
        bq = (1.0 + benchmark.fillna(0.0)).cumprod()
        ax.plot(bq.index, bq.to_numpy(), color=PALETTE["benchmark"],
                lw=1.3, ls="--", label="Equal-weight universe (buy & hold)")

    ax.axhline(1.0, color="black", lw=0.7, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative net value (log scale)")
    ax.set_title("Net equity: best parameter set per signal family, 2015-2024 (after all costs)")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out_dir, "fig1_equity_curves.png")


# --------------------------------------------------------------------- 图 2


def gross_vs_net_sharpe(summary: pd.DataFrame, out_dir: Path) -> Path:
    """毛夏普 vs 净夏普 —— 本项目的核心图：成本吃掉了多少。"""
    df = summary.sort_values("sharpe_gross", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.barh(y, df["sharpe_gross"], color=PALETTE["gross"], height=0.72,
            label="Gross Sharpe (before costs)")
    ax.barh(y, df["sharpe"], color=PALETTE["net"], height=0.42,
            label="Net Sharpe (after costs)")

    ax.set_yticks(y)
    labels = [
        f"{r['name']}   ({r['annual_turnover']:.0f}x/yr)"
        for _, r in df.iterrows()
    ]
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Annualised Sharpe ratio")
    ax.set_title("Costs erase the edge of every high-turnover signal")
    ax.legend(loc="lower right", fontsize=8)

    # 标注放在较长那根柱子远离 0 的一侧，避免压住柱体
    lo = min(0.0, float(df["sharpe"].min()), float(df["sharpe_gross"].min()))
    hi = max(0.0, float(df["sharpe"].max()), float(df["sharpe_gross"].max()))
    span = hi - lo
    for i, (_, r) in enumerate(df.iterrows()):
        end = max(r["sharpe_gross"], r["sharpe"])
        if end >= 0:
            ax.text(end + span * 0.015, i,
                    f"net {r['sharpe']:+.2f}  /  gross {r['sharpe_gross']:+.2f}",
                    va="center", ha="left", fontsize=6.5, color="#333333")
        else:
            ax.text(end - span * 0.015, i,
                    f"net {r['sharpe']:+.2f}  /  gross {r['sharpe_gross']:+.2f}",
                    va="center", ha="right", fontsize=6.5, color="#333333")
    ax.set_xlim(lo - span * 0.34, hi + span * 0.34)
    return _save(fig, out_dir, "fig2_gross_vs_net_sharpe.png")


# --------------------------------------------------------------------- 图 3


def turnover_vs_sharpe(summary: pd.DataFrame, out_dir: Path) -> Path:
    """换手率与夏普的关系：低换手是净收益的必要条件。"""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for fam in FAMILY_ORDER:
        sub = summary[summary["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(sub["annual_turnover"], sub["sharpe"],
                   s=58, color=PALETTE[fam], alpha=0.85,
                   edgecolor="white", linewidth=0.8, label=FAMILY_LABEL[fam])

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Annualised one-way turnover (x net asset value)")
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_title("Where the money goes: net Sharpe vs turnover")
    ax.legend(fontsize=8, loc="lower right")

    ax.axvspan(45, 56, color=PALETTE["accent"], alpha=0.07)
    ax.text(50, ax.get_ylim()[1] * 0.92,
            "~52x/yr turnover\n≈ 2.6%/yr cost",
            ha="center", fontsize=7.5, color=PALETTE["accent"])
    return _save(fig, out_dir, "fig3_turnover_vs_sharpe.png")


# --------------------------------------------------------------------- 图 4


def cost_sensitivity(curves: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """成本敏感性：策略在多少单边成本下失效。

    Parameters
    ----------
    curves
        信号名 -> DataFrame(index=单边成本 bp, columns=['sharpe'])
    """
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for nm, df in curves.items():
        fam = _family_of(nm)
        ax.plot(df.index, df["sharpe"], marker="o", ms=3.5, lw=1.6,
                color=PALETTE.get(fam, "black"), label=nm)

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Assumed one-way transaction cost (bp)")
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_title("Break-even cost: the cost level at which each strategy stops working")
    ax.legend(fontsize=7.5, loc="lower left")

    # 标出各策略的盈亏平衡成本
    for nm, df in curves.items():
        s = df["sharpe"].to_numpy(dtype=float)
        x = df.index.to_numpy(dtype=float)
        sign = np.sign(s)
        cross = np.nonzero(np.diff(sign) != 0)[0]
        if len(cross):
            i = cross[0]
            ax.plot(x[i], s[i], marker="x", ms=9, color=PALETTE["accent"], zorder=5)
    return _save(fig, out_dir, "fig4_cost_sensitivity.png")


# --------------------------------------------------------------------- 图 5


def null_distributions(
    nulls: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """置换检验的零分布：真实夏普落在零分布的什么位置。"""
    picks = best_per_family(summary)
    fams = [f for f in FAMILY_ORDER if picks.get(f) in nulls.columns]
    if not fams:
        return _save(plt.subplots(figsize=(6, 3))[0], out_dir, "fig5_null_distributions.png")

    fig, axes = plt.subplots(1, len(fams), figsize=(3.1 * len(fams), 3.4), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        nm = picks[fam]
        null = nulls[nm].dropna().to_numpy(dtype=float)
        obs = float(summary.loc[summary["name"] == nm, "sharpe_gross"].iloc[0])

        ax.hist(null, bins=32, color=PALETTE[fam], alpha=0.55,
                edgecolor="white", linewidth=0.5)
        ax.axvline(obs, color=PALETTE["accent"], lw=2.0,
                   label=f"observed {obs:+.2f}")
        ax.axvline(0, color="black", lw=0.7, alpha=0.6)
        ax.set_title(f"{FAMILY_LABEL[fam]}", fontsize=9)
        ax.set_xlabel("Sharpe under shifted prices", fontsize=8)
        ax.legend(fontsize=7)
        if ax is axes[0][0]:
            ax.set_ylabel("count", fontsize=8)

    fig.suptitle(
        "Permutation test: shifted price panels give the same Sharpe by chance",
        fontsize=10, fontweight="bold", y=1.03,
    )
    return _save(fig, out_dir, "fig5_null_distributions.png")


# --------------------------------------------------------------------- 图 6


def parameter_heatmap(summary: pd.DataFrame, family: str, out_dir: Path) -> Path:
    """某个族的参数热力图。行/列取该族网格里的两个参数。"""
    sub = summary[summary["family"] == family].copy()
    if sub.empty:
        raise ValueError(f"没有 {family} 族的结果")

    # 从 'a=1,b=2' 形式的 params 字符串里解析参数
    parsed = sub["params"].str.strip("{}").str.split(",").apply(
        lambda items: {
            k.strip().strip("'"): float(v) for k, v in (it.split(":") for it in items)
        }
    )
    keys = sorted({k for d in parsed for k in d})
    if len(keys) < 2:
        raise ValueError(f"{family} 族只有一个参数，无法画热力图")

    r_key, c_key = keys[0], keys[1]
    sub["_r"] = [d[r_key] for d in parsed]
    sub["_c"] = [d[c_key] for d in parsed]

    grid = sub.pivot_table(index="_r", columns="_c", values="sharpe")
    grid_gross = sub.pivot_table(index="_r", columns="_c", values="sharpe_gross")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, data, title in [
        (axes[0], grid_gross, "Gross Sharpe"),
        (axes[1], grid, "Net Sharpe"),
    ]:
        im = ax.imshow(data.to_numpy(), cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(data.columns)), [f"{c:g}" for c in data.columns])
        ax.set_yticks(range(len(data.index)), [f"{r:g}" for r in data.index])
        ax.set_xlabel(c_key)
        ax.set_ylabel(r_key)
        ax.set_title(title)
        ax.grid(False)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data.to_numpy()[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{FAMILY_LABEL.get(family, family)}: parameter sensitivity "
        f"(net column is what you actually get)",
        fontsize=10, fontweight="bold", y=1.04,
    )
    return _save(fig, out_dir, f"fig6_heatmap_{family}.png")


# --------------------------------------------------------------------- 图 7


def walkforward(folds: pd.DataFrame, out_dir: Path) -> Path:
    """样本内 vs 样本外：参数选择到底有没有样本外价值。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    ax = axes[0]
    x = folds["fold"].to_numpy()
    ax.bar(x - 0.19, folds["best_train_sharpe"], width=0.36,
           color=PALETTE["gross"], label="In-sample (best of 20)")
    ax.bar(x + 0.19, folds["best_test_sharpe"], width=0.36,
           color=PALETTE["net"], label="Out-of-sample (same parameter set)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("In-sample vs out-of-sample")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    colors = [PALETTE["accent"] if v < 0 else PALETTE["net"] for v in folds["rank_ic"]]
    ax.bar(x, folds["rank_ic"], color=colors, width=0.55)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("Spearman rank IC")
    ax.set_title("Does in-sample ranking predict out-of-sample ranking?")
    ax.text(0.02, 0.04, "bar > 0: parameter selection carries\ninformation forward",
            transform=ax.transAxes, fontsize=7.5, va="bottom")

    fig.suptitle("Walk-forward: is the parameter choice worth anything out of sample?",
                 fontsize=10, fontweight="bold", y=1.03)
    return _save(fig, out_dir, "fig7_walkforward.png")


# --------------------------------------------------------------------- 图 9


def cost_drag(
    net_returns: pd.DataFrame,
    gross_returns: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """成本拖累的机制：毛净值与净净值之间的缺口如何随时间累积。

    选两个对照：换手最低（配对）与毛夏普最高但换手很高的（动量 250 日）。
    两条曲线的缺口就是累计成本 —— 高换手那条的缺口几乎是一条直线，
    因为成本按成交额线性累积，而毛收益并不按同样速度复合。
    """
    picks = best_per_family(summary)
    candidates = [
        (picks.get("pairs"), "Pairs, ~10x/yr turnover"),
        ("momentum[lookback=250,skip=0]", "Momentum 250d, ~52x/yr turnover"),
    ]
    candidates = [(n, lab) for n, lab in candidates if n in net_returns.columns]
    if not candidates:
        return _save(plt.subplots(figsize=(6, 3))[0], out_dir, "fig9_cost_drag.png")

    fig, axes = plt.subplots(1, len(candidates), figsize=(4.6 * len(candidates), 4.0),
                             squeeze=False)
    for ax, (nm, lab) in zip(axes[0], candidates):
        g = (1.0 + gross_returns[nm].fillna(0.0)).cumprod()
        n = (1.0 + net_returns[nm].fillna(0.0)).cumprod()
        fam = _family_of(nm)

        ax.plot(g.index, g.to_numpy(), color=PALETTE[fam], lw=1.6, label="Gross (no costs)")
        ax.plot(n.index, n.to_numpy(), color=PALETTE[fam], lw=1.6, ls="--", label="Net (all costs)")
        ax.fill_between(g.index, n.to_numpy(), g.to_numpy(),
                        color=PALETTE["accent"], alpha=0.18)
        ax.axhline(1.0, color="black", lw=0.7, alpha=0.5)

        total = float(summary.loc[summary["name"] == nm, "total_cost"].iloc[0])
        ax.set_title(f"{lab}\ncost drag: {total * 100:.1f}% of initial capital over 10y",
                     fontsize=9)
        ax.set_ylabel("Cumulative value")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Why turnover is the first-class metric: the shaded area is money paid away",
                 fontsize=10, fontweight="bold", y=1.02)
    return _save(fig, out_dir, "fig9_cost_drag.png")


# --------------------------------------------------------------------- 图 8


def drawdown(net_returns: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> Path:
    """各信号族最优组合的回撤曲线。"""
    picks = best_per_family(summary)

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for fam in FAMILY_ORDER:
        nm = picks.get(fam)
        if nm is None or nm not in net_returns.columns:
            continue
        eq = (1.0 + net_returns[nm].fillna(0.0)).cumprod()
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.to_numpy() * 100.0, color=PALETTE[fam], lw=1.3,
                label=f"{FAMILY_LABEL[fam]} (max {dd.min() * 100:.1f}%)")

    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdowns are large relative to the returns — this is not a free lunch")
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, out_dir, "fig8_drawdown.png")
