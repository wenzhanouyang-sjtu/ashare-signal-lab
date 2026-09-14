"""Walk-forward 样本外检验。

要回答的问题
------------
"在历史上表现最好的那个参数组合，在未来是否仍然最好？"

这是量化研究里最容易被跳过、也最容易自欺的一步。如果样本内的排名
对样本外毫无预测力，那么"我挑了最优参数"这件事本身就是数据窥探，
无论那个夏普有多漂亮。

切分方式
--------
滚动窗口（rolling / anchored 可选）：

    锚定式 (anchored=True)          滚动式 (anchored=False)
    ├─train─┤test│                  ├─train─┤test│
    ├───train───┤test│                    ├─train─┤test│
    ├─────train─────┤test│                      ├─train─┤test│

锚定式训练集不断增长（模拟"数据越来越多"），滚动式训练集长度固定
（模拟"只用近期数据"，对市场结构变化更稳健）。默认滚动式。

关键输出
--------
**样本内排名与样本外表现的秩相关（rank IC）**。
接近 0 或为负 → 参数选择没有样本外价值，报告最优参数是无意义的。
本项目把这个数字直接写进 README，而不是只报告最优夏普。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


@dataclass(frozen=True)
class Split:
    """一个 walk-forward 折。索引均为整数位置。"""

    fold: int
    train_start: int
    train_end: int      # 不含
    test_start: int
    test_end: int       # 不含


def walkforward_splits(
    n_days: int,
    *,
    n_splits: int = 5,
    train_days: int = 750,
    test_days: int = 250,
    anchored: bool = False,
) -> list[Split]:
    """生成 walk-forward 折。

    训练集与测试集 **不重叠**，且测试集始终在训练集之后 —— 这是样本外的
    最低要求。折之间可以重叠（滚动式），这正是"每年重估一次参数"的语义。
    """
    if train_days + test_days > n_days:
        raise ValueError(
            f"数据长度 {n_days} 不足以切出 train={train_days} + test={test_days}"
        )

    # 让最后一折的测试集恰好落在数据末尾
    last_test_start = n_days - test_days
    if n_splits > 1:
        step = max((last_test_start - train_days) // (n_splits - 1), 1)
    else:
        step = 0

    splits: list[Split] = []
    for k in range(n_splits):
        test_start = last_test_start - step * (n_splits - 1 - k)
        test_end = test_start + test_days
        train_end = test_start
        train_start = 0 if anchored else train_end - train_days
        if train_start < 0 or test_end > n_days:
            continue
        splits.append(
            Split(
                fold=k + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return splits


def _annualized_sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def evaluate_selection(
    returns: pd.DataFrame,
    splits: list[Split],
    *,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """检验"样本内挑最优"是否有样本外价值。

    Parameters
    ----------
    returns
        各策略（信号参数组合）的日收益矩阵，列 = 策略名。
    splits
        walkforward_splits() 的输出。
    metric
        目前支持 "sharpe"。

    Returns
    -------
    DataFrame，每折一行：
        fold, n_train, n_test,
        best_in_sample        样本内夏普最高的策略名
        best_train_sharpe     它的样本内夏普
        best_test_sharpe      它在样本外的夏普
        test_sharpe_of_median 样本内中位数策略的样本外夏普（对照）
        rank_ic               样本内夏普排名与样本外夏普排名的 Spearman 秩相关
        rank_ic_p            该秩相关的 p 值
    """
    if metric != "sharpe":
        raise ValueError(f"暂不支持的 metric: {metric}")

    names = list(returns.columns)
    rows = []
    for sp in splits:
        tr = returns.iloc[sp.train_start : sp.train_end].to_numpy(dtype=float)
        te = returns.iloc[sp.test_start : sp.test_end].to_numpy(dtype=float)

        sr_tr = np.array([_annualized_sharpe(tr[:, j]) for j in range(len(names))])
        sr_te = np.array([_annualized_sharpe(te[:, j]) for j in range(len(names))])

        finite = np.isfinite(sr_tr) & np.isfinite(sr_te)
        if finite.sum() < 3:
            continue

        best = int(np.nanargmax(np.where(finite, sr_tr, -np.inf)))
        med = int(np.argsort(np.where(finite, sr_tr, -np.inf))[finite.sum() // 2])

        rho, pval = stats.spearmanr(sr_tr[finite], sr_te[finite])

        rows.append(
            {
                "fold": sp.fold,
                "n_train": sp.train_end - sp.train_start,
                "n_test": sp.test_end - sp.test_start,
                "best_in_sample": names[best],
                "best_train_sharpe": float(sr_tr[best]),
                "best_test_sharpe": float(sr_te[best]),
                "test_sharpe_of_median": float(sr_te[med]),
                "rank_ic": float(rho),
                "rank_ic_p": float(pval),
            }
        )
    return pd.DataFrame(rows)


def summarize_selection(fold_table: pd.DataFrame) -> dict:
    """把逐折结果压成一句话能说的结论。"""
    if fold_table.empty:
        return {"n_folds": 0}
    return {
        "n_folds": int(len(fold_table)),
        "mean_best_train_sharpe": float(fold_table["best_train_sharpe"].mean()),
        "mean_best_test_sharpe": float(fold_table["best_test_sharpe"].mean()),
        "mean_test_sharpe_of_median": float(fold_table["test_sharpe_of_median"].mean()),
        "mean_rank_ic": float(fold_table["rank_ic"].mean()),
        "n_folds_positive_rank_ic": int((fold_table["rank_ic"] > 0).sum()),
        # 样本内最优相对中位数的"优势"，在样本外还剩多少
        "sharpe_decay": float(
            fold_table["best_train_sharpe"].mean() - fold_table["best_test_sharpe"].mean()
        ),
    }
