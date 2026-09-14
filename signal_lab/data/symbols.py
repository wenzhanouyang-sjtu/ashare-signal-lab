"""A股代码格式转换。

数据源对代码格式的要求不一致：
    新浪 / 腾讯  ->  "sh600519"
    中证指数官网 ->  "600519"
本项目内部统一使用 6 位纯数字代码，仅在调用数据源时转换。
"""

from __future__ import annotations

# 板块前缀规则，按首位数字判断
_SH_PREFIXES = ("6", "9")          # 沪市主板、科创板(688)、沪市B股
_SZ_PREFIXES = ("0", "2", "3")     # 深市主板、创业板(300)、深市B股
_BJ_PREFIXES = ("4", "8")          # 北交所


def to_prefixed(code: str) -> str:
    """'600519' -> 'sh600519'（新浪/腾讯接口要求的格式）。"""
    code = normalize(code)
    head = code[0]
    if head in _SH_PREFIXES:
        return f"sh{code}"
    if head in _SZ_PREFIXES:
        return f"sz{code}"
    if head in _BJ_PREFIXES:
        return f"bj{code}"
    raise ValueError(f"无法识别的股票代码: {code}")


def normalize(code: str) -> str:
    """去除交易所前缀与空白，返回 6 位纯数字代码。"""
    code = str(code).strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if code.startswith(prefix):
            code = code[len(prefix):]
    code = code.strip()
    if not (len(code) == 6 and code.isdigit()):
        raise ValueError(f"非法的股票代码: {code!r}")
    return code


def board_of(code: str) -> str:
    """判断所属板块 —— 涨跌停幅度按板块不同（主板 10%，双创 20%）。"""
    code = normalize(code)
    if code.startswith("688"):
        return "STAR"        # 科创板 ±20%
    if code.startswith("300") or code.startswith("301"):
        return "ChiNext"     # 创业板 ±20%
    if code.startswith(("4", "8")):
        return "BSE"         # 北交所 ±30%
    return "default"         # 主板 ±10%
