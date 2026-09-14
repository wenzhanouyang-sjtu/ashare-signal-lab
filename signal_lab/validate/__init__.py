"""验证层：判断一个漂亮的回测结果究竟是信号还是运气。

四道关卡，各针对一类具体的自欺方式：

    walkforward        样本外：在历史上挑最好的，在未来是否仍然最好？
    permutation        显著性：把持仓与收益的时间对齐关系打乱，还能有这个夏普吗？
    deflated_sharpe    多重检验：试了 N 次之后，"最好那次的夏普"本身就偏高
    multiple_testing   FDR：在 N 次检验中，控制假阳性的比例
"""

from signal_lab.validate.deflated_sharpe import deflated_sharpe_ratio
from signal_lab.validate.multiple_testing import benjamini_hochberg, bonferroni
from signal_lab.validate.permutation import permutation_test
from signal_lab.validate.walkforward import (
    evaluate_selection,
    walkforward_splits,
)

__all__ = [
    "deflated_sharpe_ratio",
    "benjamini_hochberg",
    "bonferroni",
    "permutation_test",
    "walkforward_splits",
    "evaluate_selection",
]
