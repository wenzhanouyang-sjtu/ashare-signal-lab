"""本地 parquet 缓存。

为什么必须有缓存：
  1. 抓取 300 只股票的日线约需 20 分钟，研究迭代不可能每次重抓
  2. 新浪接口会限流，重复请求既是浪费也容易触发封禁
  3. 断点续传 —— 中途中断后重跑只补缺失的部分

缓存文件命名：{cache_dir}/prices/{code}.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from signal_lab.data.symbols import normalize


def prices_dir(cache_dir: str | Path) -> Path:
    d = Path(cache_dir) / "prices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(cache_dir: str | Path, code: str) -> Path:
    return prices_dir(cache_dir) / f"{normalize(code)}.parquet"


def load(cache_dir: str | Path, code: str) -> pd.DataFrame | None:
    """读取缓存；不存在则返回 None。"""
    path = cache_path(cache_dir, code)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        # 缓存损坏（例如上次写入时被中断）时视为未命中，触发重新抓取
        return None


def save(cache_dir: str | Path, code: str, df: pd.DataFrame) -> Path:
    """先写临时文件再原子替换，避免中断留下半截文件。"""
    path = cache_path(cache_dir, code)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def cached_codes(cache_dir: str | Path) -> set[str]:
    """已缓存的股票代码集合，用于断点续传时跳过。"""
    d = Path(cache_dir) / "prices"
    if not d.exists():
        return set()
    return {p.stem for p in d.glob("*.parquet")}


def covers(df: pd.DataFrame, start: str, end: str) -> bool:
    """判断缓存是否覆盖了请求区间（容许首尾非交易日造成的偏差）。"""
    if df is None or df.empty or "date" not in df.columns:
        return False
    lo, hi = pd.to_datetime(df["date"]).min(), pd.to_datetime(df["date"]).max()
    want_lo, want_hi = pd.to_datetime(start), pd.to_datetime(end)
    # 数据源的实际首日在请求起始日之后是正常的（例如股票当时尚未上市），
    # 因此只在"末尾明显不足"时判定缓存失效。
    return hi >= want_hi - pd.Timedelta(days=10) and lo <= want_lo + pd.Timedelta(days=20)
