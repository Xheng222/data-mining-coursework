"""评测指标：Detection(P/R) 与 Recovery Rate。

本模块提供两类评测能力：

1. ``train_and_auc``：在（可能很脏的）训练集上训练 XGBoost，并在固定的干净测试集上
   返回 AUC-ROC。它只做让模型「能跑」所需的最小数值化，不做任何清洗/插补——
   清洗是 baseline / agent 的职责，这里只负责量化「训练集脏到什么程度会拖垮下游」。
2. ``detection_scores``：把「被报告的缺陷」与 ground truth 按缺陷类型逐一比对，
   计算 per-type 与整体（micro 平均）的 precision/recall/f1。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.contracts import DEFECT_TYPES, DefectReport, DetectionScore


def _label_encode_object_columns(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对 object/string 列做标签编码，使 tree-based 模型能利用类别信息。

    缺失值或训练集未出现的类别统一编码为 -1（XGBoost 视为缺失）。
    """
    x_train = x_train.copy()
    x_test = x_test.copy()
    for col in x_train.columns:
        if x_train[col].dtype == object or x_train[col].dtype.name == "category":
            # 以训练集类别构建编码映射
            cats = x_train[col].astype("category").cat.categories
            cat_to_code = {c: i for i, c in enumerate(cats)}
            x_train[col] = x_train[col].map(cat_to_code).fillna(-1).astype(int)
            x_test[col] = x_test[col].map(cat_to_code).fillna(-1).astype(int)
    return x_train, x_test


def train_and_auc(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "target",
) -> float:
    """在 train 上训练 XGBoost，在固定的 test 上返回 AUC-ROC。

    设计要点：
    - 特征列取自 train（去掉 target 列）；test 用 ``reindex(columns=特征列)`` 对齐，
      test 缺的列填 NaN、多余列丢弃。这模拟「训练期出现的泄漏特征在测试期不可得」。
    - 先对字符串列做标签编码，再对剩余列做 ``pd.to_numeric(errors="coerce")``，
      XGBoost 原生支持 NaN。缺失值编码为 -1。
    - 二分类用正类概率、多分类用 ``multi_class="ovr"`` 计算 AUC。
    - 边界情况（test 中标签只有单一类别、无法计算 AUC 等）返回 0.5（随机水平）。

    返回：float 形式的 AUC-ROC。
    """
    feature_cols = [c for c in train.columns if c != target_col]

    # 先对 object/string 列做标签编码（在 to_numeric 之前，否则字符串会被转为 NaN 丢失信息）
    x_train = train[feature_cols].copy()
    x_test = test.reindex(columns=feature_cols).copy()
    x_train, x_test = _label_encode_object_columns(x_train, x_test)

    # 再对剩余列做最小数值化（非数值→NaN），XGBoost 原生支持 NaN
    x_train = x_train.apply(pd.to_numeric, errors="coerce")
    x_test = x_test.apply(pd.to_numeric, errors="coerce")

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


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """由 tp/fp/fn 计算 precision/recall/f1。

    空集合约定：当 tp+fp == 0（什么都没报告）时 precision = 1.0；
    当 tp+fn == 0（该类型本无缺陷）时 recall = 1.0。
    因此「truth 空且 reported 空」会得到 P=R=F1=1.0（正确地什么都没报）。
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def detection_scores(reported: DefectReport, ground_truth: DefectReport) -> DetectionScore:
    """按 contracts 的 items 集合，分缺陷类型计算 P/R/F1 并 micro 平均。

    对每个 ``DEFECT_TYPES`` 中的类型 t，取 ``reported[t]["items"]`` 与
    ``ground_truth[t]["items"]`` 两个集合，计算：
      - tp = 交集大小、fp = reported 独有、fn = truth 独有。
    per-type 用上述集合各自算 P/R/F1（空集合约定见 ``_prf``）。
    整体用 **micro 平均**：累加所有类型的 tp/fp/fn 后统一算 P/R/F1。

    ``reported`` / ``ground_truth`` 可能缺某些 type 键，用
    ``.get(t, {}).get("items", [])`` 兜底为空集合。
    """
    per_type: dict[str, dict[str, float]] = {}
    total_tp = total_fp = total_fn = 0

    for t in DEFECT_TYPES:
        rep_items = set(reported.get(t, {}).get("items", []) or [])
        gt_items = set(ground_truth.get(t, {}).get("items", []) or [])

        tp = len(rep_items & gt_items)
        fp = len(rep_items - gt_items)
        fn = len(gt_items - rep_items)

        precision, recall, f1 = _prf(tp, fp, fn)
        per_type[t] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision, recall, f1 = _prf(total_tp, total_fp, total_fn)
    return DetectionScore(precision=precision, recall=recall, f1=f1, per_type=per_type)


def load_ground_truth(path: str | Path) -> DefectReport:
    """从 JSON 文件读取 ground-truth 缺陷报告，返回 DefectReport（dict）。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
