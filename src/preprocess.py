"""规则化清洗步骤（供 rule-based 基线调用）。feature/baseline 负责实现。"""

from __future__ import annotations

import pandas as pd

from src.contracts import CleaningResult


def rule_based_clean(dirty_train: pd.DataFrame) -> CleaningResult:
    """对脏训练集执行固定规则清洗，并报告检测到的缺陷。

    规则参考开题报告：均值/众数填充缺失、删除重复行、按与目标的高相关
    剔除疑似泄漏列、统一格式等。返回 CleaningResult（含 reported_defects）。
    """
    raise NotImplementedError
