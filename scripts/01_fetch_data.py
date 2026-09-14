#!/usr/bin/env python3
"""步骤 1：抓取并缓存行情数据。

用法::

    python scripts/01_fetch_data.py                # 按 config.yaml 全量抓取
    python scripts/01_fetch_data.py --limit 10     # 只抓前 10 只（冒烟测试）
    python scripts/01_fetch_data.py --refresh      # 忽略缓存，强制重抓
    python scripts/01_fetch_data.py --verbose      # 打印每只股票的数据源

抓取 300 只约需 20 分钟。中途 Ctrl-C 后重跑会自动跳过已缓存的股票。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_lab.config import load_config  # noqa: E402
from signal_lab.data.clean import clean_prices, mark_tradeable  # noqa: E402
from signal_lab.data.fetch import fetch_many  # noqa: E402
from signal_lab.data.universe import get_universe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取 A 股行情数据")
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 只股票")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存强制重抓")
    ap.add_argument("--verbose", action="store_true", help="打印逐只明细")
    args = ap.parse_args()

    cfg = load_config()
    t0 = time.time()

    print(f"项目: {cfg.project.name}")
    print(f"区间: {cfg.data.start_date} ~ {cfg.data.end_date}")
    print(f"数据源优先级: {' > '.join(cfg.data.sources)}")

    universe = get_universe(cfg)
    codes = universe["code"].tolist()
    if args.limit:
        codes = codes[: args.limit]
    print(f"股票池: {cfg.data.universe.index_name} — {len(codes)} 只\n")

    raw, failures = fetch_many(
        codes, cfg, use_cache=not args.refresh, verbose=args.verbose
    )
    print(f"\n抓取完成: {raw['code'].nunique()} 只成功, {len(failures)} 只失败 "
          f"({time.time() - t0:.0f}s)")

    if failures:
        print("\n失败的股票（不影响整体，已跳过）:")
        for code, err in failures[:10]:
            print(f"  {code}: {err[:90]}")
        if len(failures) > 10:
            print(f"  ... 另有 {len(failures) - 10} 只")

    cleaned = clean_prices(raw)
    cleaned = mark_tradeable(cleaned, cfg)

    cache_dir = cfg.path("data", "cache_dir")
    out_path = cache_dir / "prices_clean.parquet"
    cleaned.to_parquet(out_path, index=False)

    print(f"\n清洗后数据: {len(cleaned):,} 行 × {cleaned['code'].nunique()} 只")
    print(f"日期范围: {cleaned['date'].min().date()} ~ {cleaned['date'].max().date()}")
    print(f"停牌占比: {cleaned['suspended'].mean():.3%}")
    print(f"涨停日占比: {cleaned['limit_up'].mean():.3%}")
    print(f"跌停日占比: {cleaned['limit_down'].mean():.3%}")
    print(f"\n已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
