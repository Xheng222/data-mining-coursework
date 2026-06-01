"""规则化清洗步骤（供 rule-based 基线调用）。

本模块只实现「固定规则」的清洗逻辑，不依赖任何 LLM。规则覆盖六类缺陷中
规则法可处理或可检测的部分：

    - missing              : 检测含 NaN 的列，数值列均值填充、非数值列众数填充
    - format_inconsistency : object 列但可被 to_numeric 解析的，强制转数值后填充
    - near_duplicate       : 完全重复（含四舍五入后重复）的行，删除并保留首次出现
    - leakage              : 与 target 的 |Pearson 相关| > 0.95 的特征列，视为泄漏并删除
    - class_imbalance      : 少数类占比 < 0.2 仅检测记录（不强制重采样）
    - label_noise          : 规则法无法可靠检测，不报告

落盘约定（重要）
----------------
``rule_based_clean`` **不负责落盘**，只返回内存中的清洗后 DataFrame：

    - ``CleaningResult.extra["cleaned_df"]`` 存放清洗后的 ``pd.DataFrame``；
    - ``CleaningResult.cleaned_train_path`` 字段类型为 ``Path``，此处只填一个
      占位路径（``Path("<in-memory>")``），真正落盘由调用方（``baseline.py``）
      在拿到 ``extra["cleaned_df"]`` 后完成，并自行写回真实路径。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

from src.contracts import (
    TARGET_COL,
    CleaningResult,
    empty_report,
    normalize_pair,
)

# 占位路径：rule_based_clean 不落盘，由调用方写回真实路径。
_PLACEHOLDER_PATH = Path("<in-memory>")

# 泄漏判定阈值：与 target 的 |Pearson 相关| 超过此值的特征视为泄漏。
_LEAKAGE_CORR_THRESHOLD = 0.95

# 类别不平衡判定阈值：少数类占比低于此值则记为存在不平衡。
_IMBALANCE_MINORITY_RATIO = 0.2


def rule_based_clean(dirty_train: pd.DataFrame) -> CleaningResult:
    """对脏训练集执行固定规则清洗，并报告检测到的缺陷。

    处理顺序：格式统一 -> 缺失填充 -> 近似重复删除 -> 泄漏列删除 ->
    类别不平衡检测。返回的 ``CleaningResult`` 中：

        - ``reported_defects`` 按 ``contracts`` 的 schema 填充各缺陷类型的 items；
        - ``extra["cleaned_df"]`` 为清洗后的 DataFrame（**不落盘**，详见模块 docstring）；
        - ``cleaned_train_path`` 为占位路径，落盘交由调用方处理；
        - ``log`` 记录每一步实际做了什么。

    参数
    ----
    dirty_train : pd.DataFrame
        含特征列与 ``TARGET_COL`` 目标列的脏训练集。

    返回
    ----
    CleaningResult
    """
    log: list[str] = []
    report = empty_report()

    # 在副本上操作，避免修改调用方传入的对象。
    df = dirty_train.copy()

    feature_cols = [c for c in df.columns if c != TARGET_COL]

    # --- 1. 格式不一致：object 列但可解析为数值 ----------------------------- #
    format_cols: list[str] = []
    for col in feature_cols:
        if not ptypes.is_object_dtype(df[col]) and not ptypes.is_string_dtype(df[col]):
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_null = int(df[col].notna().sum())
        # 若原本非空的值在强制转换后大多数能成为数值，则认为是「数值列被存成了文本」。
        if non_null > 0 and int(coerced.notna().sum()) >= 0.5 * non_null:
            df[col] = coerced
            format_cols.append(col)
    if format_cols:
        report["format_inconsistency"]["items"] = list(format_cols)
        log.append(
            f"格式不一致：将 {format_cols} 转为数值（pd.to_numeric, errors='coerce'）"
        )
    else:
        log.append("格式不一致：未发现可转为数值的 object 列")

    # --- 2. 缺失值填充 ------------------------------------------------------ #
    # 经过上一步 coerce 后，无法解析的文本会变成 NaN，这里一并填充。
    missing_cols = [c for c in df.columns if c != TARGET_COL and df[c].isna().any()]
    for col in missing_cols:
        if ptypes.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else ""
            df[col] = df[col].fillna(fill_value)
    if missing_cols:
        report["missing"]["items"] = list(missing_cols)
        log.append(f"缺失值：对 {missing_cols} 做均值/众数填充")
    else:
        log.append("缺失值：未发现含 NaN 的列")

    # --- 3. 近似重复：完全重复（含四舍五入后重复）的行 ---------------------- #
    # 规则法只能抓精确重复；对数值列做轻度四舍五入以捕捉「近似」重复。
    dup_basis = df.copy()
    num_cols = [c for c in dup_basis.columns if ptypes.is_numeric_dtype(dup_basis[c])]
    if num_cols:
        dup_basis[num_cols] = dup_basis[num_cols].round(6)
    dup_mask = dup_basis.duplicated(keep="first")
    if dup_mask.any():
        # 为每个重复行找到其首次出现的行下标（按位置），组成规范化样本对。
        seen: dict[tuple, int] = {}
        dup_pairs: list[str] = []
        rows_as_tuples = [
            tuple(r) for r in dup_basis.itertuples(index=False, name=None)
        ]
        for pos, key in enumerate(rows_as_tuples):
            if key in seen:
                dup_pairs.append(normalize_pair(seen[key], pos))
            else:
                seen[key] = pos
        df = df.loc[~dup_mask.values].reset_index(drop=True)
        report["near_duplicate"]["items"] = list(dup_pairs)
        log.append(
            f"近似重复：检测到 {int(dup_mask.sum())} 个重复行，删除并保留首次出现"
        )
    else:
        log.append("近似重复：未发现完全/四舍五入后重复的行")

    # --- 4. 泄漏：与 target 高相关的特征列 ---------------------------------- #
    leakage_cols: list[str] = []
    if TARGET_COL in df.columns:
        target = pd.to_numeric(df[TARGET_COL], errors="coerce")
        for col in [c for c in df.columns if c != TARGET_COL]:
            if not ptypes.is_numeric_dtype(df[col]):
                continue
            corr = df[col].corr(target)
            if pd.notna(corr) and abs(corr) > _LEAKAGE_CORR_THRESHOLD:
                leakage_cols.append(col)
    if leakage_cols:
        df = df.drop(columns=leakage_cols)
        report["leakage"]["items"] = list(leakage_cols)
        log.append(
            f"泄漏：删除与 target 相关性 >|{_LEAKAGE_CORR_THRESHOLD}| 的列 {leakage_cols}"
        )
    else:
        log.append("泄漏：未发现与 target 高相关的特征列")

    # --- 5. 类别不平衡：仅检测 --------------------------------------------- #
    if TARGET_COL in df.columns and len(df) > 0:
        counts = df[TARGET_COL].value_counts(normalize=True)
        minority_ratio = float(counts.min()) if not counts.empty else 1.0
        if minority_ratio < _IMBALANCE_MINORITY_RATIO:
            report["class_imbalance"]["items"] = ["present"]
            report["class_imbalance"]["meta"] = {"minority_ratio": minority_ratio}
            log.append(
                f"类别不平衡：少数类占比 {minority_ratio:.3f} < "
                f"{_IMBALANCE_MINORITY_RATIO}，记为存在（规则法仅检测不重采样）"
            )
        else:
            log.append(
                f"类别不平衡：少数类占比 {minority_ratio:.3f}，未达不平衡阈值"
            )

    # --- 6. 标签噪声：规则法不报告 ----------------------------------------- #
    log.append("标签噪声：规则法无法可靠检测，不报告")

    return CleaningResult(
        cleaned_train_path=_PLACEHOLDER_PATH,
        reported_defects=report,
        log=log,
        extra={"cleaned_df": df},
    )
