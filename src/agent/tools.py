"""Executor 可调用的数据操作工具。

全部为确定性的纯 pandas / numpy 操作，不含任何 LLM 调用。Executor 节点先用
LLM 产出结构化判断，再调用这里的函数真正执行修复，从而让「决策」与「执行」
分离：决策可消融、可解释，执行可复现、可测试。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.contracts import TARGET_COL, normalize_pair


# --------------------------------------------------------------------------- #
# 画像（profile）
# --------------------------------------------------------------------------- #

def profile_data(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict[str, Any]:
    """对脏训练集做确定性统计画像，供 LLM 决策与启发式兜底使用。

    返回 dict 包含：
      - n_rows / n_cols
      - columns：列名列表
      - dtypes：列 -> dtype 字符串
      - missing：列 -> 缺失值个数（仅含缺失>0 的列）
      - missing_frac：列 -> 缺失比例
      - n_duplicate_rows：完全重复行数
      - duplicate_pairs：近似/完全重复样本对（规范化 "i-j"，最多若干对）
      - target_corr：数值列与 target 的 |相关系数|（降序），用于发现疑似泄漏
      - high_corr_cols：|相关系数| 极高（>=0.98）的列，强烈疑似泄漏
      - class_counts：target 各类别计数
      - imbalance_ratio：多数类 / 少数类
      - object_cols：object/类别型列（可能存在格式不一致）
      - format_suspect_cols：值里混入空白/大小写/别名等格式不一致迹象的列
      - label_noise_signal：基于 CV 残差的标签噪声疑似信号
      - near_duplicate_pairs：基于数值特征距离的近似重复对（超越精确匹配）
    """
    prof: dict[str, Any] = {}
    prof["n_rows"] = int(len(df))
    prof["n_cols"] = int(df.shape[1])
    prof["columns"] = list(map(str, df.columns))
    prof["dtypes"] = {str(c): str(df[c].dtype) for c in df.columns}

    # 缺失
    miss = df.isna().sum()
    miss = miss[miss > 0]
    prof["missing"] = {str(c): int(v) for c, v in miss.items()}
    prof["missing_frac"] = {
        str(c): round(float(v) / max(len(df), 1), 4) for c, v in miss.items()
    }

    # 重复
    dup_mask = df.duplicated(keep="first")
    prof["n_duplicate_rows"] = int(dup_mask.sum())
    prof["duplicate_pairs"] = _duplicate_pairs(df, max_pairs=50)

    # 与 target 的相关性（仅数值列）
    target_corr: dict[str, float] = {}
    high_corr: list[str] = []
    if target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce")
        for c in df.columns:
            if c == target_col:
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() < 2 or x.nunique(dropna=True) < 2:
                continue
            try:
                r = x.corr(y)
            except Exception:
                r = np.nan
            if pd.notna(r):
                target_corr[str(c)] = round(abs(float(r)), 4)
                if abs(float(r)) >= 0.98:
                    high_corr.append(str(c))
    prof["target_corr"] = dict(
        sorted(target_corr.items(), key=lambda kv: kv[1], reverse=True)
    )
    prof["high_corr_cols"] = high_corr

    # 类别分布 / 不平衡
    if target_col in df.columns:
        vc = df[target_col].value_counts(dropna=True)
        prof["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            prof["imbalance_ratio"] = round(float(vc.max()) / max(float(vc.min()), 1.0), 3)
        else:
            prof["imbalance_ratio"] = float("inf")
    else:
        prof["class_counts"] = {}
        prof["imbalance_ratio"] = 1.0

    # object 列与格式不一致迹象
    object_cols = [str(c) for c in df.columns if df[c].dtype == object and c != target_col]
    prof["object_cols"] = object_cols
    prof["format_suspect_cols"] = [c for c in object_cols if _looks_format_inconsistent(df[c])]

    # 标签噪声信号：CV 残差
    prof["label_noise_signal"] = _cv_label_noise_signal(df, target_col)

    # 近似重复：基于距离的检测（超越精确匹配）
    prof["near_duplicate_pairs"] = _near_duplicate_by_distance(df, target_col, max_pairs=50)

    return prof


def _duplicate_pairs(df: pd.DataFrame, max_pairs: int = 50) -> list[str]:
    """找出重复（含近似：在非 target 特征上完全相等）的样本对，返回规范化 "i-j"。

    以「除 target 外全部列相等」为重复判据，与 datagen 的近似重复注入语义对齐
    （注入时通常复制特征行）。用首次出现行作为代表，与其它同组行配对。
    """
    pairs: list[str] = []
    feat_cols = [c for c in df.columns if c != TARGET_COL]
    if not feat_cols:
        return pairs
    # 用可哈希的元组分组，避免 O(n^2)
    key = df[feat_cols].astype(str).apply(lambda row: "".join(row.values), axis=1)
    groups = key.groupby(key).groups
    positions = {idx: pos for pos, idx in enumerate(df.index)}
    for _, idxs in groups.items():
        if len(idxs) < 2:
            continue
        pos_sorted = sorted(positions[i] for i in idxs)
        rep = pos_sorted[0]
        for other in pos_sorted[1:]:
            pairs.append(normalize_pair(rep, other))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _looks_format_inconsistent(s: pd.Series) -> bool:
    """启发式：一个 object 列是否存在格式不一致（首尾空白、大小写混用、等价别名）。"""
    vals = s.dropna().astype(str)
    if vals.empty:
        return False
    # 首尾空白
    if (vals != vals.str.strip()).any():
        return True
    stripped = vals.str.strip()
    # 同一字符串去大小写后类别更少 => 大小写不一致
    if stripped.str.lower().nunique() < stripped.nunique():
        return True
    return False


# --------------------------------------------------------------------------- #
# 标签噪声信号：基于交叉验证残差
# --------------------------------------------------------------------------- #

def _cv_label_noise_signal(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict[str, Any]:
    """用 LogisticRegression + 5-fold CV 检测疑似标签噪声。

    对每折计算预测概率与真实标签的绝对残差，取全样本残差平均值。
    若 top-5% 残差均值远超整体均值（>3 倍），则认为存在标签噪声。

    返回
    ----
    dict:
        - suspected : bool, 是否疑似存在标签噪声
        - mean_residual : float, 全体样本平均绝对残差
        - top5_residual : float, 残差最大的 5% 样本的平均残差
        - ratio : float, top5_residual / mean_residual
    """
    result: dict[str, Any] = {"suspected": False, "mean_residual": 0.0, "top5_residual": 0.0, "ratio": 1.0}

    if target_col not in df.columns:
        return result

    # 只取数值特征
    feat_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(feat_cols) < 2:
        return result

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)

    # 去掉含 NaN 的行
    valid = ~(np.isnan(x).any(axis=1) | np.isnan(y))
    if valid.sum() < 10:
        return result
    x, y = x[valid], y[valid]

    if len(np.unique(y)) < 2:
        return result

    # 5-fold CV 残差
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    residuals = np.full(len(y), np.nan)

    for train_idx, val_idx in skf.split(x, y):
        try:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_val = scaler.transform(x[val_idx])

            model = LogisticRegression(max_iter=500, random_state=42)
            model.fit(x_train, y[train_idx])
            prob = model.predict_proba(x_val)[:, 1]
            residuals[val_idx] = np.abs(y[val_idx] - prob)
        except Exception:
            continue

    valid_residuals = residuals[~np.isnan(residuals)]
    if len(valid_residuals) < 10:
        return result

    mean_res = float(np.mean(valid_residuals))
    top5_threshold = float(np.percentile(valid_residuals, 95))
    top5_res = float(valid_residuals[valid_residuals >= top5_threshold].mean()) if (valid_residuals >= top5_threshold).any() else mean_res
    ratio = top5_res / mean_res if mean_res > 1e-9 else 1.0

    result["mean_residual"] = round(mean_res, 4)
    result["top5_residual"] = round(top5_res, 4)
    result["ratio"] = round(ratio, 3)
    # 若 top-5% 残差均值远超整体均值，且平均残差非极小，认为存在疑似标签噪声
    result["suspected"] = (ratio > 2.0 and mean_res > 0.08) or ratio > 3.0

    return result


# --------------------------------------------------------------------------- #
# 近似重复：基于数值特征距离的检测
# --------------------------------------------------------------------------- #

def _near_duplicate_by_distance(
    df: pd.DataFrame, target_col: str = TARGET_COL, max_pairs: int = 50
) -> list[str]:
    """用最近邻距离检测数值特征上的近似重复行。

    对标准化后的数值特征做 NearestNeighbors，找出距离 < 自适应阈值的行对，
    排除已由 ``_duplicate_pairs`` 覆盖的精确重复。

    返回
    ----
    list[str]
        规范化样本对 "i-j" (i<j)，最多 max_pairs 对。
    """
    feat_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(feat_cols) < 2:
        return []

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(x).any(axis=1)
    if valid.sum() < 5:
        return []

    x_clean = x[valid]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_clean)

    nn = NearestNeighbors(n_neighbors=min(3, len(x_scaled)), metric="euclidean")
    nn.fit(x_scaled)
    distances, indices = nn.kneighbors(x_scaled)

    # 自适应阈值：第二近邻距离（排除自身第一近邻）的 10 分位
    second_dist = distances[:, 1]
    threshold = float(np.percentile(second_dist, 10)) * 1.5

    # 同时设置绝对阈值上限（避免大距离被误判）
    threshold = min(threshold, 0.5)

    pairs: list[str] = []
    seen: set[tuple[int, int]] = set()

    # 已有精确重复，跳过它们
    exact_dup_set: set[tuple[int, int]] = set()
    for p in _duplicate_pairs(df, max_pairs=1000):
        try:
            a, b = p.split("-", 1)
            exact_dup_set.add((int(a), int(b)))
        except Exception:
            pass

    valid_indices = np.where(valid)[0]

    n_neighbors = distances.shape[1]
    for i in range(len(x_scaled)):
        for k in range(1, n_neighbors):
            if distances[i, k] >= threshold:
                continue
            j = int(indices[i, k])
            if j <= i:
                continue
            a, b = int(valid_indices[i]), int(valid_indices[j])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen or key in exact_dup_set:
                continue
            seen.add(key)
            pairs.append(normalize_pair(a, b))
            if len(pairs) >= max_pairs:
                return pairs

    return pairs


# --------------------------------------------------------------------------- #
# 修复操作（确定性）
# --------------------------------------------------------------------------- #

def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """删除指定列（忽略不存在的列、绝不删除 target）。返回新 DataFrame。"""
    to_drop = [c for c in columns if c in df.columns and c != TARGET_COL]
    return df.drop(columns=to_drop)


def impute_missing(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """填补缺失值：数值列用中位数，非数值列用众数（无众数则用占位串）。

    columns 为 None 时处理所有含缺失的列；target 列若缺失则删除对应行（标签不可臆造）。
    返回新 DataFrame。
    """
    out = df.copy()
    if TARGET_COL in out.columns and out[TARGET_COL].isna().any():
        out = out[out[TARGET_COL].notna()].reset_index(drop=True)

    target_cols = columns if columns is not None else list(out.columns)
    for c in target_cols:
        if c not in out.columns or c == TARGET_COL:
            continue
        if not out[c].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            fill = out[c].median()
            if pd.isna(fill):
                fill = 0.0
        else:
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "missing"
        out[c] = out[c].fillna(fill)
    return out


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """删除在非 target 特征上完全相同的重复行，保留首次出现。返回新 DataFrame。"""
    feat_cols = [c for c in df.columns if c != TARGET_COL]
    if not feat_cols:
        return df.copy()
    return df.drop_duplicates(subset=feat_cols, keep="first").reset_index(drop=True)


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """把指定列规整为数值：先做常见格式归一（去空白、去千分位逗号、百分号），
    再 to_numeric。无法转换的值变 NaN（交由后续 impute 处理）。返回新 DataFrame。
    """
    out = df.copy()
    for c in columns:
        if c not in out.columns or c == TARGET_COL:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        cleaned = (
            out[c]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        out[c] = pd.to_numeric(cleaned, errors="coerce")
    return out


def normalize_text(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """统一文本格式：去首尾空白 + 统一小写，用于修复格式不一致的类别列。

    target 列不处理；仅作用于实际为 object 类型的列。返回新 DataFrame。
    """
    out = df.copy()
    for c in columns:
        if c not in out.columns or c == TARGET_COL:
            continue
        if out[c].dtype != object:
            continue
        out[c] = out[c].astype(str).str.strip().str.lower()
    return out
