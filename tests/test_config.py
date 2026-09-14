"""配置对象的行为测试。

盯住一个具体的、曾经真实存在的静默失效：对嵌套配置赋值看似生效、
实际改的是一个临时副本，赋值不报错、读回来还是旧值。

为什么这条测试重要：本项目的成本敏感性分析（`fig4_cost_sensitivity`）
靠的就是"改配置 → 重跑引擎"来产出曲线。若赋值静默失效，
整条曲线会退化成一条水平线，而**不会有任何报错**。
"""

from __future__ import annotations

import copy

import pytest

from signal_lab.config import load_config


@pytest.fixture()
def cfg():
    return load_config()


def test_attribute_assignment_persists(cfg):
    """点号赋值必须真的改到底层字典，并且读回来是新值。"""
    original = cfg.backtest.costs.commission
    cfg.backtest.costs.commission = 0.0

    assert cfg.backtest.costs.commission == 0.0
    assert cfg["backtest"]["costs"]["commission"] == 0.0, "两种访问方式必须看到同一个值"

    cfg.backtest.costs.commission = original
    assert cfg.backtest.costs.commission == original


def test_nested_objects_are_not_copies(cfg):
    """同一路径取两次必须拿到同一个对象，否则赋值会落在副本上。"""
    assert cfg.backtest.costs is cfg.backtest.costs
    assert cfg["backtest"]["costs"] is cfg.backtest.costs


def test_assignment_survives_round_trip_through_parent(cfg):
    """从父节点取到子节点再赋值，改的必须是同一份数据。"""
    node = cfg.backtest.execution
    node.suspension = False
    assert cfg.backtest.execution.suspension is False


def test_indexing_and_attribute_are_equivalent(cfg):
    cfg["backtest"]["costs"]["stamp_tax"] = 0.001
    assert cfg.backtest.costs.stamp_tax == pytest.approx(0.001)


def test_setitem_wraps_nested_dicts(cfg):
    """通过 __setitem__ 塞进去的嵌套字典也应当支持点号访问。"""
    cfg["_tmp_test"] = {"a": {"b": 1}}
    assert cfg["_tmp_test"].a.b == 1
    assert cfg._tmp_test.a.b == 1


def test_missing_key_raises_attribute_error(cfg):
    with pytest.raises(AttributeError, match="配置项不存在"):
        _ = cfg.definitely_not_a_real_key


def test_deepcopy_keeps_config_semantics(cfg):
    """成本敏感性分析用 deepcopy 造变体，副本必须同样可点号读写。"""
    cfg.backtest.costs.commission = 0.05

    variant = copy.deepcopy(cfg)
    variant.backtest.costs.commission = 0.0

    assert variant.backtest.costs.commission == 0.0
    assert cfg.backtest.costs.commission == 0.05, "副本不得污染原配置"
    assert isinstance(variant.backtest, type(cfg.backtest))


def test_path_resolves_against_project_root(cfg):
    p = cfg.path("data", "cache_dir")
    assert p.is_absolute()
    assert p.name == "data_cache"


def test_no_dead_price_limit_keys(cfg):
    """price_limits 里不得存在 board_of() 永远返回不了的键。

    为什么要有这条测试：`ST: 0.05` 在配置里躺了很久，README 也照着它宣称
    "ST ±5% 已实现"。但 board_of() 只按代码前缀返回 STAR / ChiNext / BSE /
    default，**永远返回不了 "ST"** —— 于是那行配置从未被任何代码读到，
    而 ST 股票的封板被按 ±10% 少判，结果偏乐观。

    死配置比缺配置更危险：缺配置你知道要补，死配置让你以为已经补了。
    这条测试把 price_limits 的键域与 board_of() 的值域锁在一起，
    以后再加板块（或有人手滑写个新键）会立刻失败。
    """
    from signal_lab.data.symbols import board_of

    # board_of 的值域：用各板块的代表性代码探出来
    samples = ["600000", "000001", "300750", "301236", "688981", "830799", "430047"]
    reachable = {board_of(c) for c in samples}
    assert reachable == {"default", "ChiNext", "STAR", "BSE"}, (
        f"board_of 值域变了: {reachable} —— 请同步更新本条测试与 config.yaml"
    )

    keys = set(cfg.backtest.price_limits.keys())
    dead = keys - reachable - {"default"}
    assert not dead, (
        f"price_limits 里有 board_of() 永远返回不了的键: {sorted(dead)}。"
        f"它们是从未被读到的死配置，会让人误以为该板块已建模。"
    )
