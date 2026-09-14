"""配置加载。

所有影响结果的参数都必须来自 config.yaml —— 代码中不得出现硬编码的数值。
这既是可复现性的要求，也让"我们试了多少组参数"这件事可被核查，
而后者正是多重检验校正 (FDR / Deflated Sharpe) 的输入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


class Config(dict):
    """嵌套字典，支持点号访问：cfg.backtest.costs.commission

    注意 `__getattr__` **必须返回内层字典本身，而不是它的副本**。
    早期实现写的是 `return Config(value)` —— 而 `dict.__init__` 会复制一层，
    于是 `cfg.backtest.costs` 返回的是一个用完即弃的壳：

        cfg.backtest.costs.commission = 0.0     # 看起来改了，实际什么都没发生

    这类"静默失效"最难发现：赋值不报错、读回来还是旧值、下游照常跑完。
    本项目的成本敏感性分析正是靠改配置重跑来产生曲线的，
    一旦踩中这个坑，整条曲线会是一条水平线而没有任何提示。
    现在嵌套字典在构造时就递归包装成 Config，取值不再复制。

    要改配置，两种写法都安全：
        cfg["backtest"]["costs"]["commission"] = 0.0
        cfg.backtest.costs.commission = 0.0
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, Config):
                super().__setitem__(key, Config(value))

    def __getattr__(self, name: str) -> Any:
        # 只在常规属性查找失败后调用，因此不会与 dict 自身的方法冲突
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"配置项不存在: {name}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def path(self, *keys: str) -> Path:
        """把配置里的相对路径解析成基于项目根目录的绝对路径。"""
        node: Any = self
        for k in keys:
            node = node[k]
        p = Path(node)
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
