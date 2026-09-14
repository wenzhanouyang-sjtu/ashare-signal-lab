"""行情数据抓取：多源降级 + 指数退避 + 本地缓存。

设计依据（规划阶段实测，非假设）：
  * 东方财富接口 (ak.stock_zh_a_hist) 在本机网络下持续拒绝连接
    (RemoteDisconnected)，因此不出现在数据源列表中。
  * 新浪接口可用且返回流通股本 —— 这是计算真实换手率与冲击成本的基础，
    因此作为主源。
  * 腾讯接口可用但不含流通股本，作为备源。
  * 抓取 300 只股票约需 20 分钟，且接口会限流，故必须缓存 + 限速。

统一 schema（下游代码只需面对这一种）::

    date  open  high  low  close  volume  amount  turnover  outstanding_share  code
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Sequence

import pandas as pd

from signal_lab.data import cache
from signal_lab.data.symbols import normalize, to_prefixed

# 统一列集合。数据源缺失的列补 NaN，避免下游分支判断。
STANDARD_COLUMNS = [
    "date", "open", "high", "low", "close",
    "volume", "amount", "turnover", "outstanding_share",
]


# ---------------------------------------------------------------- 数据源适配器
def _from_sina(symbol: str, start: str, end: str) -> pd.DataFrame:
    """新浪财经。主源 —— 含 outstanding_share，可算换手率与冲击成本。"""
    import akshare as ak

    return ak.stock_zh_a_daily(
        symbol=symbol, start_date=start, end_date=end, adjust="qfq"
    )


def _from_tencent(symbol: str, start: str, end: str) -> pd.DataFrame:
    """腾讯财经。备源 —— 缺流通股本，但日线完整。"""
    import akshare as ak

    return ak.stock_zh_a_hist_tx(
        symbol=symbol, start_date=start, end_date=end, adjust="qfq"
    )


SOURCES: dict[str, Callable[[str, str, str], pd.DataFrame]] = {
    "sina": _from_sina,
    "tencent": _from_tencent,
}


# ---------------------------------------------------------------- 内部工具
def _normalize(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    """把任意数据源的返回整理成统一 schema。"""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[STANDARD_COLUMNS].copy()

    df["date"] = pd.to_datetime(df["date"])
    numeric = [c for c in STANDARD_COLUMNS if c != "date"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["code"] = normalize(code)
    return (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    return df.loc[mask].reset_index(drop=True)


def _with_retry(
    fn: Callable[[], pd.DataFrame],
    *,
    retries: int,
    backoff_base: float,
    sleep_between: float,
    label: str,
    verbose: bool,
) -> pd.DataFrame:
    """指数退避重试。限流是常态而非异常，因此重试是必选项。"""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 —— 数据源会抛各种网络异常
            last_exc = exc
            if attempt < retries - 1:
                wait = sleep_between * (backoff_base ** (attempt + 1))
                if verbose:
                    print(
                        f"      重试 {attempt + 1}/{retries - 1} "
                        f"({type(exc).__name__}) — {wait:.1f}s 后重试"
                    )
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------- 对外接口
def fetch_one(
    code: str,
    cfg,
    *,
    start: str | None = None,
    end: str | None = None,
    sources: Sequence[str] | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """抓取单只股票的日线。命中缓存则直接返回，否则按数据源优先级逐个尝试。"""
    code = normalize(code)
    start = start or cfg.data.start_date
    end = end or cfg.data.end_date
    cache_dir = cfg.path("data", "cache_dir")
    fetch_cfg = cfg.data.fetch

    if use_cache:
        cached = cache.load(cache_dir, code)
        if cached is not None and cache.covers(cached, start, end):
            if verbose:
                print(f"      [缓存] {code}: {len(cached)} 行")
            return _slice(cached, start, end)

    sources = list(sources or cfg.data.sources)
    symbol = to_prefixed(code)
    failures: list[str] = []

    for src_name in sources:
        adapter = SOURCES.get(src_name)
        if adapter is None:
            failures.append(f"{src_name}(未知数据源)")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = _with_retry(
                    lambda a=adapter: a(symbol, start, end),
                    retries=fetch_cfg.max_retries,
                    backoff_base=fetch_cfg.backoff_base,
                    sleep_between=fetch_cfg.sleep_between,
                    label=f"{code}@{src_name}",
                    verbose=verbose,
                )
            df = _normalize(raw, code)
            if df.empty:
                raise ValueError("数据源返回空表")

            if use_cache:
                cache.save(cache_dir, code, df)
            if verbose:
                print(f"      [{src_name}] {code}: {len(df)} 行")
            return _slice(df, start, end)

        except Exception as exc:  # noqa: BLE001
            failures.append(f"{src_name}({type(exc).__name__})")
            if verbose:
                print(f"      [{src_name}] {code} 失败: {type(exc).__name__}")
        finally:
            time.sleep(fetch_cfg.sleep_between)

    raise RuntimeError(f"{code} 所有数据源均失败: {', '.join(failures)}")


def fetch_many(
    codes: Sequence[str],
    cfg,
    *,
    use_cache: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """批量抓取。

    返回 (长表 DataFrame, 失败列表)。单只失败不会中断整批 ——
    300 只股票里有个别停牌/退市的属正常，不应让整轮研究白跑。
    """
    try:
        from tqdm import tqdm

        iterator = tqdm(codes, desc="抓取行情", disable=not show_progress)
    except ImportError:
        iterator = codes  # type: ignore[assignment]

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    for code in iterator:
        try:
            frames.append(fetch_one(code, cfg, use_cache=use_cache, verbose=verbose))
        except Exception as exc:  # noqa: BLE001
            failures.append((str(code), str(exc)))
            if show_progress:
                print(f"\n  ! {code} 抓取失败: {exc}")

    if not frames:
        raise RuntimeError("全部股票抓取失败，请检查网络与数据源可用性")

    combined = pd.concat(frames, ignore_index=True)
    return combined, failures
