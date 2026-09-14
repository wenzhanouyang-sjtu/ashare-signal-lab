#!/usr/bin/env python3
"""步骤 0（可选）：把行情面板与回测结果装载进 SQLite，供 SQL 查询。

为什么需要这一层
----------------
本项目的核心工作是 **点位（point-in-time）** 数据分析，而点位数据天然是关系型的：

    "在 2015-07-08 这天，沪深300 里有几只股票既没停牌、也没封涨停、且成交额足够大？"

这类问题用 pandas 写要绕好几行，用 SQL 是一句话。更重要的是，
**可交易性是一个随时间变化的属性**，把它物化成一张带索引的表，
后续任何信号研究都可以直接 JOIN 上去，而不必每次重算涨跌停。

用法::

    python scripts/00_build_db.py            # 构建
    python scripts/00_build_db.py --query    # 只跑预置查询

输出::

    data_cache/ashare.db           SQLite 数据库
    results/db_findings.md         预置查询的结果（供 README 引用）
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_lab.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# 建表语句。刻意把「行情」与「可交易性」分成两张表：
# 前者是市场事实，后者是 **由本项目从事实推导出的判定**（含规则假设）。
# 混在一起会让"这个 can_buy 是怎么来的"变得不可追溯。
# ---------------------------------------------------------------------------
SCHEMA = """
DROP TABLE IF EXISTS prices;
CREATE TABLE prices (
    date              TEXT    NOT NULL,
    code              TEXT    NOT NULL,
    open              REAL,
    high              REAL,
    low               REAL,
    close             REAL,
    volume            REAL,
    amount            REAL,     -- 成交额（元）
    turnover          REAL,     -- 换手率
    outstanding_share REAL,
    PRIMARY KEY (date, code)
);
CREATE INDEX idx_prices_code ON prices(code);
CREATE INDEX idx_prices_date ON prices(date);

DROP TABLE IF EXISTS tradability;
CREATE TABLE tradability (
    date       TEXT    NOT NULL,
    code       TEXT    NOT NULL,
    ret        REAL,             -- 当日收盘对前收的涨跌幅
    suspended  INTEGER NOT NULL, -- 停牌（当日无成交）
    limit_up   INTEGER NOT NULL, -- 收盘封涨停（不可买入）
    limit_down INTEGER NOT NULL, -- 收盘封跌停（不可卖出）
    can_buy    INTEGER NOT NULL,
    can_sell   INTEGER NOT NULL,
    PRIMARY KEY (date, code),
    FOREIGN KEY (date, code) REFERENCES prices(date, code)
);
CREATE INDEX idx_trad_date ON tradability(date);

DROP TABLE IF EXISTS universe_snapshot;
CREATE TABLE universe_snapshot (
    code          TEXT PRIMARY KEY,
    index_code    TEXT NOT NULL,
    snapshot_date TEXT NOT NULL   -- 成分股名单的获取日期（非历史成分）
);

DROP TABLE IF EXISTS backtest_summary;
CREATE TABLE backtest_summary (
    name             TEXT PRIMARY KEY,
    family           TEXT,
    sharpe           REAL,
    sharpe_gross     REAL,
    annual_return    REAL,
    annual_turnover  REAL,
    total_cost       REAL,
    max_drawdown     REAL
);
"""

# ---------------------------------------------------------------------------
# 预置查询。每条都对应研究里的一个真实问题，不是为演示 SQL 而写。
# ---------------------------------------------------------------------------
QUERIES: list[tuple[str, str]] = [
    (
        "每年不可交易的比例（涨跌停 / 停牌）",
        """
        SELECT substr(date, 1, 4)                       AS year,
               COUNT(*)                                 AS obs,
               ROUND(100.0 * SUM(limit_up)   / COUNT(*), 3) AS pct_limit_up,
               ROUND(100.0 * SUM(limit_down) / COUNT(*), 3) AS pct_limit_down,
               ROUND(100.0 * SUM(suspended)  / COUNT(*), 3) AS pct_suspended,
               ROUND(100.0 * SUM(1 - can_buy)  / COUNT(*), 3) AS pct_cannot_buy,
               ROUND(100.0 * SUM(1 - can_sell) / COUNT(*), 3) AS pct_cannot_sell
        FROM tradability
        GROUP BY year
        ORDER BY year
        """,
    ),
    (
        "极端行情日的可交易性（列涨幅/跌幅最极端的 8 天）",
        """
        SELECT date,
               COUNT(*)                                   AS names,
               ROUND(100.0 * AVG(limit_up),   1)          AS pct_limit_up,
               ROUND(100.0 * AVG(limit_down), 1)          AS pct_limit_down,
               ROUND(100.0 * AVG(suspended),  1)          AS pct_suspended,
               ROUND(100.0 * AVG(can_buy),    1)          AS pct_can_buy
        FROM tradability
        GROUP BY date
        HAVING COUNT(*) > 200
        ORDER BY pct_limit_down DESC
        LIMIT 8
        """,
    ),
    (
        "流动性：分年度成交额分位数（验证冲击成本假设是否合理）",
        """
        SELECT substr(p.date, 1, 4)                       AS year,
               COUNT(*)                                   AS obs,
               ROUND(AVG(p.amount)  / 1e8, 3)             AS mean_amount_yi,
               ROUND(MIN(p.amount)  / 1e8, 4)             AS min_amount_yi
        FROM prices p
        JOIN tradability t ON t.date = p.date AND t.code = p.code
        WHERE t.can_buy = 1
        GROUP BY year
        ORDER BY year
        """,
    ),
    (
        "各信号族的净夏普与换手率（毛/净对照）",
        """
        SELECT family,
               COUNT(*)                          AS n_configs,
               ROUND(AVG(sharpe_gross), 3)       AS avg_sharpe_gross,
               ROUND(AVG(sharpe), 3)             AS avg_sharpe_net,
               ROUND(AVG(annual_turnover), 1)    AS avg_turnover,
               ROUND(100 * AVG(total_cost), 2)   AS avg_cost_pct
        FROM backtest_summary
        GROUP BY family
        ORDER BY avg_sharpe_net DESC
        """,
    ),
    (
        "成本侵蚀：毛夏普为正但净夏普为负的组合",
        """
        SELECT name,
               ROUND(sharpe_gross, 3)         AS sharpe_gross,
               ROUND(sharpe, 3)               AS sharpe_net,
               ROUND(annual_turnover, 1)      AS turnover,
               ROUND(100 * total_cost, 2)     AS cost_pct
        FROM backtest_summary
        WHERE sharpe_gross > 0 AND sharpe < 0
        ORDER BY sharpe_gross DESC
        """,
    ),
]


def build(cfg) -> Path:
    """把清洗后的 parquet 装载进 SQLite。"""
    cache = cfg.path("data", "cache_dir")
    src = cache / "prices_clean.parquet"
    if not src.exists():
        raise SystemExit(f"找不到 {src}\n请先运行 scripts/01_fetch_data.py")

    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    db_path = cache / "ashare.db"
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    price_cols = [
        "date", "code", "open", "high", "low", "close",
        "volume", "amount", "turnover", "outstanding_share",
    ]
    have = [c for c in price_cols if c in df.columns]
    df[have].to_sql("prices", con, if_exists="append", index=False)

    trad_cols = ["date", "code", "ret", "suspended", "limit_up", "limit_down",
                 "can_buy", "can_sell"]
    trad = df[[c for c in trad_cols if c in df.columns]].copy()
    for c in ("suspended", "limit_up", "limit_down", "can_buy", "can_sell"):
        if c in trad.columns:
            trad[c] = trad[c].fillna(False).astype(int)
    trad.to_sql("tradability", con, if_exists="append", index=False)

    # 成分股名单快照：必须记录获取日期，因为它是"当前"名单而非历史名单
    #
    # ⚠ 文件名必须与 signal_lab/data/universe.py 的写入路径一致：
    #   那边写的是 f"universe_{index_code}.parquet"（如 universe_000300.parquet）。
    #   这里曾经写成 "universe.parquet"，于是 exists() 恒为 False，
    #   universe_snapshot 表**静默地永远是 0 行** —— 不报错、不警告，
    #   和本项目记录的其他几个坑同一个形状。
    #   守卫：tests/test_data.py::test_universe_snapshot_is_populated
    index_code = str(cfg.data.universe.index_code)
    uni_path = cache / f"universe_{index_code}.parquet"
    if not uni_path.exists():
        raise FileNotFoundError(
            f"找不到成分股缓存 {uni_path} —— 请先运行 scripts/01_fetch_data.py。"
            f"（这里刻意抛异常而不是静默跳过：跳过会让 universe_snapshot 变成空表，"
            f"而空表看起来和'没有数据'没有区别。）"
        )
    if uni_path.exists():
        uni = pd.read_parquet(uni_path)
        uni = uni.rename(columns={"index_code": "index_code"})
        if "index_code" not in uni.columns:
            uni["index_code"] = str(cfg.data.universe.index_code)
        uni["snapshot_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
        uni[[c for c in ("code", "index_code", "snapshot_date") if c in uni.columns]] \
            .drop_duplicates("code").to_sql(
                "universe_snapshot", con, if_exists="append", index=False
            )

    results_dir = cfg.path("report", "results_dir")
    summary_path = results_dir / "summary_all.csv"
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        keep = [c for c in ["name", "family", "sharpe", "sharpe_gross", "annual_return",
                            "annual_turnover", "total_cost", "max_drawdown"] if c in s.columns]
        s[keep].to_sql("backtest_summary", con, if_exists="append", index=False)

    con.execute("ANALYZE")
    con.commit()

    n_prices = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    n_trad = con.execute("SELECT COUNT(*) FROM tradability").fetchone()[0]
    con.close()

    print(f"数据库: {db_path}")
    print(f"  prices       {n_prices:>9,} 行")
    print(f"  tradability  {n_trad:>9,} 行")
    return db_path


def run_queries(cfg, db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    results_dir = cfg.path("report", "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# SQL 查询结果（自动生成）\n",
             f"数据库：`{db_path.name}`，由 `scripts/00_build_db.py` 生成。\n"]
    for title, sql in QUERIES:
        df = pd.read_sql_query(sql, con)
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print(df.to_string(index=False))
        lines.append(f"\n## {title}\n")
        lines.append(df.to_markdown(index=False))
    con.close()

    out = results_dir / "db_findings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n查询结果已写入 {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="装载行情面板进 SQLite 并跑预置查询")
    ap.add_argument("--query", action="store_true", help="跳过装载，只跑查询")
    args = ap.parse_args()

    cfg = load_config()
    db_path = cfg.path("data", "cache_dir") / "ashare.db"
    if not args.query:
        db_path = build(cfg)
    elif not db_path.exists():
        raise SystemExit(f"数据库不存在: {db_path}\n请先不带 --query 运行一次")

    run_queries(cfg, db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
