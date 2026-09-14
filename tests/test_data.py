"""数据清洗与可交易性标记的测试。

这里锁住的是一个**曾经真实存在**的问题：数据源在停牌日不返回任何行，
而不是返回一行成交量为 0 的记录。于是"停牌"会和"尚未上市"混为一谈，
`volume <= 0` 这样的判据一次都不会触发 —— 标记列存在、代码能跑、
结果是错的。这类错误没有异常，只能靠测试发现。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_lab.config import load_config
from signal_lab.data.clean import build_panels, clean_prices, mark_tradeable


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_raw(gaps: dict[str, list[str]] | None = None,
             start: str = "2020-01-01", periods: int = 10) -> pd.DataFrame:
    """构造原始行情：所有股票每天都有行，再按 `gaps` 删掉指定的行。

    gaps={"A": ["2020-01-06", ...]} 表示 A 在这几天"没有数据"。
    """
    dates = pd.bdate_range(start, periods=periods)
    codes = ["000001", "600000"]
    rows = [
        {"date": d, "code": c, "open": 10.0, "high": 10.0, "low": 10.0,
         "close": 10.0, "volume": 1e6, "amount": 1e7}
        for d in dates for c in codes
    ]
    df = pd.DataFrame(rows)
    for code, days in (gaps or {}).items():
        drop = df["code"].eq(code) & df["date"].isin(pd.to_datetime(days))
        df = df[~drop]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 停牌的表示方式
# ---------------------------------------------------------------------------

def test_suspended_flag_actually_fires(cfg):
    """核心回归：整行缺失的日子必须被识别为停牌。

    旧实现用 `volume.isna() | (volume <= 0)`，在真实数据上命中 0 次。
    """
    raw = make_raw(gaps={"000001": ["2020-01-07", "2020-01-08"]})
    out = mark_tradeable(clean_prices(raw), cfg)

    a = out[(out["code"] == "000001")
            & out["date"].isin(pd.to_datetime(["2020-01-07", "2020-01-08"]))]
    assert len(a) == 2, "停牌行应当被补出来，而不是继续缺失"
    assert a["suspended"].all()
    assert a["close"].isna().all()
    assert not a["can_buy"].any()
    assert not a["can_sell"].any()

    b = out[(out["code"] == "600000")]
    assert not b["suspended"].any(), "另一只股票不受影响"


def test_pre_listing_is_not_marked_suspended(cfg):
    """尚未上市 ≠ 停牌。首次出现之前的日期根本不该有行。

    两者对回测的含义完全不同：前者不该有仓位，后者有仓位但卖不掉。
    """
    raw = make_raw()
    # 600000 从第三个交易日才开始有数据（模拟新上市）
    cutoff = sorted(raw["date"].unique())[2]
    raw = raw[~((raw["code"] == "600000") & (raw["date"] < cutoff))]

    out = mark_tradeable(clean_prices(raw), cfg)
    b = out[out["code"] == "600000"]

    assert b["date"].min() == cutoff, "上市前的日期不应被补成停牌行"
    assert not b["suspended"].any()


def test_suspension_panels_match_absent_row_encoding(cfg):
    """补全前后的**下游面板必须逐位一致**。

    这是允许把补全逻辑加进来而不重跑历史结果的前提：引擎读的是面板，
    只要面板不变，引擎行为就不变。停牌行价格是 NaN，经 ffill 后与
    "整行缺失 → pivot 出 NaN → ffill" 完全等价。
    """
    raw = make_raw(gaps={"000001": ["2020-01-07", "2020-01-08"]})
    out = mark_tradeable(clean_prices(raw), cfg)
    panels = build_panels(out)

    # 手工构造"停牌日无行"的等价面板：非停牌行的值不变，停牌处为 NaN
    ref_close = out.pivot_table(index="date", columns="code", values="close",
                                aggfunc="last").sort_index()
    obs = panels["close"]
    assert np.array_equal(obs.to_numpy(), ref_close.to_numpy(), equal_nan=True)
    assert obs.loc[pd.Timestamp("2020-01-07"), "000001"] != obs.loc[
        pd.Timestamp("2020-01-07"), "000001"]  # NaN

    # 可交易性面板：停牌 → False，与"缺失 fillna(False)"一致
    assert not panels["can_buy"].loc[pd.Timestamp("2020-01-07"), "000001"]
    assert panels["can_buy"].loc[pd.Timestamp("2020-01-07"), "600000"]
    assert panels["can_buy"].dtypes.iloc[0] == bool


def test_resumption_return_includes_the_gap(cfg):
    """复牌首日的收益相对**最后成交价**计算，因此包含停牌期间的跳空。

    若把停牌简单地按"收益记 0"处理而丢掉跳空，长停牌复牌后的风险会被
    系统性低估 —— 真实数据里最长的一段停牌有 311 个交易日。
    """
    raw = make_raw(gaps={"000001": ["2020-01-07", "2020-01-08"]})
    raw.loc[(raw["code"] == "000001") & (raw["date"] == "2020-01-09"),
            ["open", "high", "low", "close"]] = 8.0   # 复牌跌 20%
    out = mark_tradeable(clean_prices(raw), cfg)

    a = out[out["code"] == "000001"].sort_values("date")
    halted = a[a["suspended"]]
    assert (halted["ret"] == 0.0).all(), "停牌期间收益为 0（持仓冻结）"

    resume = a[a["date"] == pd.Timestamp("2020-01-09")].iloc[0]
    assert resume["ret"] == pytest.approx(np.log(0.8)), "复牌日必须计入跳空"
    assert resume["prev_close"] == 10.0, "前收应为最后成交价，而非停牌期间的填充价"


def test_suspension_can_be_disabled(cfg):
    """关掉停牌约束后，补出来的行仍应存在，但不再阻断交易。

    保留行是刻意的：价格缺失是数据事实，是否因此禁止成交是建模选择。
    """
    raw = make_raw(gaps={"000001": ["2020-01-07"]})
    cfg2 = load_config()
    # Config.__getattr__ 每次都返回包着同一底层 dict 的新 wrapper，
    # 所以这样赋值会改到共享的嵌套 dict 上；用 object.__setattr__ 则会丢失。
    cfg2.backtest.execution.suspension = False

    out = mark_tradeable(clean_prices(raw), cfg2)
    a = out[(out["code"] == "000001") & (out["date"] == "2020-01-07")]
    assert len(a) == 1
    assert a["suspended"].all(), "数据事实不受建模开关影响"
    assert a["can_buy"].all(), "但建模上允许交易"


def test_limit_up_price_uses_rounded_tick(cfg):
    """涨跌停按四舍五入后的价格判定，而不是比较百分比。

    3.33 元的股票涨停价是 3.66 元，涨幅 9.91% ——
    用 `pct_change >= 0.10` 会漏判，用价格的 tick 才准。
    """
    dates = pd.bdate_range("2020-01-01", periods=3)
    rows = [
        {"date": d, "code": "600000", "open": px, "high": px, "low": px,
         "close": px, "volume": 1e6, "amount": 1e7}
        for d, px in zip(dates, [3.33, 3.66, 3.66])
    ]
    out = mark_tradeable(clean_prices(pd.DataFrame(rows)), cfg)
    # 3.33 * 1.1 = 3.663 → round(2) = 3.66，收盘 3.66 即封板
    day2 = out[out["date"] == dates[1]].iloc[0]
    assert day2["limit_up"], "按 tick 判定应当识别为涨停"
    assert not day2["can_buy"], "封涨停不可买入"
    assert day2["can_sell"], "涨停仍可卖出"


def test_panels_keep_missing_distinct_from_zero(cfg):
    """面板不做填充：缺失保持 NaN，让下游能区分"无数据"与"价格不变"。"""
    raw = make_raw()
    cutoff = sorted(raw["date"].unique())[3]
    raw = raw[~((raw["code"] == "600000") & (raw["date"] < cutoff))]
    out = mark_tradeable(clean_prices(raw), cfg)
    close = build_panels(out)["close"]

    assert close.loc[close.index < cutoff, "600000"].isna().all(), \
        "上市前必须是 NaN，不能填 0"
    assert close.loc[close.index >= cutoff, "600000"].notna().all()


def test_universe_cache_filename_matches_build_db_contract(cfg):
    """成分股缓存文件名必须与 00_build_db.py 寻找的一致。

    这里曾经不一致：universe.py 写 `f"universe_{index_code}.parquet"`，
    而 00_build_db.py 找 `"universe.parquet"` —— 于是 exists() 恒为 False，
    `universe_snapshot` 表**静默地永远是 0 行**。不报错、不警告，
    和本项目记录的其他几个坑同一个形状：安静地什么都没做。

    两层守卫：
      1) 源码契约 —— 脚本里必须用带 index_code 的 f-string；
      2) 行为 —— 若数据库已建好，universe_snapshot 不得为空表。
    """
    index_code = str(cfg.data.universe.index_code)
    root = Path(__file__).resolve().parent.parent
    src = (root / "scripts" / "00_build_db.py").read_text(encoding="utf-8")

    # 只看真正的赋值语句，不看注释 —— 否则脚本里任何一句解释性注释
    # 都能让这条断言通过（本测试第一版就是这么被骗过去的：注释里
    # 写了正确的 f-string，而代码那行是错的，断言照样通过）。
    assigns = [
        ln.strip() for ln in src.splitlines()
        if ln.strip().startswith("uni_path") and "=" in ln
    ]
    assert assigns, "00_build_db.py 里找不到 uni_path 赋值语句"
    bad = [ln for ln in assigns if f'f"universe_{{index_code}}.parquet"' not in ln]
    assert not bad, (
        f"00_build_db.py 必须按 universe_<index_code>.parquet 查找成分股缓存，"
        f"否则 universe_snapshot 会静默装载 0 行。实际写法: {bad}"
    )

    db = cfg.path("data", "cache_dir") / "ashare.db"
    if not db.exists():
        pytest.skip("尚未建库，跳过行为检查（先跑 scripts/00_build_db.py）")

    import sqlite3
    con = sqlite3.connect(db)
    n = con.execute("select count(*) from universe_snapshot").fetchone()[0]
    con.close()
    assert n > 0, (
        f"universe_snapshot 有 {n} 行 —— 空表说明成分股快照从未装载，"
        f"而它存在的意义正是记录'这份名单是什么时候取的'"
    )
