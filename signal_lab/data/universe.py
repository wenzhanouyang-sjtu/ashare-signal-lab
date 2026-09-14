"""股票池构建。

数据来源：中证指数官网（ak.index_stock_cons_csindex）—— 实测可用。
东方财富与 baostock 的对应接口在本机网络下不可达。

已知局限 —— 幸存者偏差：
    该接口只返回 **当前** 成分股，拿不到历史成分名单。因此用当前成分股
    回溯历史，等于事先剔除了后来被调出/退市的股票。
    对多头策略这会系统性高估收益；对配对/中性策略影响小得多，因为
    多空两条腿同时存在，个股层面的存续偏差大部分被对冲。
    该局限在 README 与结果中明确披露，并附稳健性检验。
"""

from __future__ import annotations

import pandas as pd

from signal_lab.data.symbols import normalize


def get_universe(cfg, *, use_cache: bool = True) -> pd.DataFrame:
    """返回指数成分股列表。

    Returns
    -------
    DataFrame with columns: code, name
    """
    index_code = cfg.data.universe.index_code
    cache_dir = cfg.path("data", "cache_dir")
    cached_path = cache_dir / f"universe_{index_code}.parquet"

    if use_cache and cached_path.exists():
        return pd.read_parquet(cached_path)

    df = _fetch_constituents(index_code)
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached_path, index=False)

    max_stocks = cfg.data.universe.get("max_stocks")
    if max_stocks:
        df = df.head(int(max_stocks)).reset_index(drop=True)
    return df


def _fetch_constituents(index_code: str) -> pd.DataFrame:
    """按优先级尝试多个成分股接口。"""
    import akshare as ak

    attempts = [
        ("csindex", lambda: ak.index_stock_cons_csindex(symbol=index_code)),
        ("sina", lambda: ak.index_stock_cons(symbol=index_code)),
    ]
    errors: list[str] = []

    for name, fn in attempts:
        try:
            raw = fn()
            df = _standardize(raw)
            if not df.empty:
                return df
            errors.append(f"{name}: 空表")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}")

    raise RuntimeError(f"无法获取指数 {index_code} 的成分股 -> {'; '.join(errors)}")


def _standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """不同接口的列名不一致，统一成 code / name。"""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    code_col = next(
        (c for c in ("成分券代码", "品种代码", "代码", "symbol") if c in df.columns),
        None,
    )
    name_col = next(
        (c for c in ("成分券名称", "品种名称", "名称", "name") if c in df.columns),
        None,
    )
    if code_col is None:
        raise ValueError(f"无法识别代码列，实际列名: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "code": df[code_col].astype(str).map(_safe_normalize),
            "name": df[name_col].astype(str) if name_col else "",
        }
    )
    return (
        out.dropna(subset=["code"])
        .drop_duplicates("code")
        .reset_index(drop=True)
    )


def _safe_normalize(code: str) -> str | None:
    try:
        return normalize(code)
    except ValueError:
        return None
