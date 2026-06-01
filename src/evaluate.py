"""评测指标：Detection(P/R) 与 Recovery Rate。

本模块提供两类评测能力：

1. ``train_and_auc``：在（可能很脏的）训练集上训练 XGBoost，并在固定的干净测试集上
   返回 AUC-ROC。它只做让模型「能跑」所需的最小数值化，不做任何清洗/插补——
   清洗是 baseline / agent 的职责，这里只负责量化「训练集脏到什么程度会拖垮下游」。
2. ``detection_scores``：把「被报告的缺陷」与 ground truth 按缺陷类型逐一比对，
   计算 per-type 与整体（micro 平均）的 precision/recall/f1。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.contracts import DefectReport, DetectionScore


def train_and_auc(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "target",
) -> float:
    """在 train 上训练 XGBoost，在固定的 test 上返回 AUC-ROC。

    设计要点：
    - 特征列取自 train（去掉 target 列）；test 用 ``reindex(columns=特征列)`` 对齐，
      test 缺的列填 NaN、多余列丢弃。这模拟「训练期出现的泄漏特征在测试期不可得」。
    - 逐列 ``pd.to_numeric(errors="coerce")`` 做最小数值化（非数值→NaN），不做插补，
      XGBoost 原生支持 NaN。这只是让模型能跑，不等于清洗。
    - 二分类用正类概率、多分类用 ``multi_class="ovr"`` 计算 AUC。
    - 边界情况（test 中标签只有单一类别、无法计算 AUC 等）返回 0.5（随机水平）。

    返回：float 形式的 AUC-ROC。
    """
    feature_cols = [c for c in train.columns if c != target_col]

    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_test = test.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")

    y_train = train[target_col]
    y_test = test[target_col]

    # 训练集本身只有单一类别时无法训练出有意义的分类器。
    if y_train.nunique(dropna=True) < 2:
        return 0.5
    # 测试集只有单一类别时 ROC-AUC 无定义。
    if y_test.nunique(dropna=True) < 2:
        return 0.5

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    # XGBClassifier 要求标签为从 0 开始的连续整数；用 category 编码兜底任意标签类型。
    y_train_enc = y_train.astype("category")
    classes = list(y_train_enc.cat.categories)
    y_train_codes = y_train_enc.cat.codes

    model.fit(x_train, y_train_codes)

    proba = model.predict_proba(x_test)

    try:
        if len(classes) == 2:
            # 正类 = 训练集 category 中的第二个类别。
            pos_label = classes[1]
            y_true_bin = (y_test == pos_label).astype(int)
            if y_true_bin.nunique() < 2:
                return 0.5
            return float(roc_auc_score(y_true_bin, proba[:, 1]))
        else:
            # 多分类：把 test 标签映射到训练编码空间，未见过的类别会导致无法计算。
            cat_map = {c: i for i, c in enumerate(classes)}
            y_test_codes = y_test.map(cat_map)
            if y_test_codes.isna().any() or y_test_codes.nunique() < 2:
                return 0.5
            return float(
                roc_auc_score(
                    y_test_codes,
                    proba,
                    multi_class="ovr",
                    labels=list(range(len(classes))),
                )
            )
    except ValueError:
        return 0.5


def detection_scores(reported: DefectReport, ground_truth: DefectReport) -> DetectionScore:
    """按 contracts 的 items 集合，分缺陷类型计算 P/R/F1 并 micro 平均。"""
    raise NotImplementedError


def load_ground_truth(path: str | Path) -> DefectReport:
    raise NotImplementedError
