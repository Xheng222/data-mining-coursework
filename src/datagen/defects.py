"""六类缺陷注入函数。

每个 ``inject_*`` 接收一个 :class:`pandas.DataFrame`，返回一个二元组
``(注入后的 DataFrame, ground_truth 片段)``。ground_truth 片段的格式遵循
:mod:`src.contracts` 中描述的 ``DefectReport`` schema，即::

    {"items": [<可识别单元>, ...], "meta": {...}}

各缺陷类型 ``items`` 的粒度见 ``src/contracts.py`` 顶部 docstring：

    - leakage              : 泄漏列的列名
    - missing              : 含缺失值的列名
    - label_noise          : ["present"]
    - near_duplicate       : 规范化样本对 "i-j"
    - class_imbalance      : ["present"]
    - format_inconsistency : 受影响列名

注意约定
--------
- 函数不修改入参，统一在副本上操作后返回。
- 涉及行索引的缺陷（label_noise / near_duplicate / format_inconsistency）
  记录的是**当前 DataFrame 位置序号**（0..len-1）。调用方应在所有注入完成、
  最终 reset_index 之后保证这些位置序号与落盘行号一致；为此本模块约定：
  注入顺序中先做不改变行集合的缺陷，最后做 near_duplicate 追加行。
  详见 ``generate.py`` 的编排。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import TARGET_COL, normalize_pair


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """返回除标签列外的所有列名。"""
    return [c for c in df.columns if c != TARGET_COL]


def inject_leakage(
    df: pd.DataFrame,
    n_cols: int = 2,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """注入数据泄漏：新增 ``n_cols`` 个与 target 强相关的列。

    每个泄漏列形如 ``leak_k = target * scale + 高斯噪声``，列名 ``leak_0`` 起。

    items
        泄漏列名列表。
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    target = out[TARGET_COL].to_numpy(dtype=float)

    leak_cols: list[str] = []
    for k in range(n_cols):
        scale = float(rng.uniform(3.0, 6.0))
        noise = rng.normal(0.0, 0.3, size=len(out))
        col = f"leak_{k}"
        out[col] = target * scale + noise
        leak_cols.append(col)

    # 把泄漏列移到标签列之前，保持 target 在最后一列。
    ordered = [c for c in out.columns if c != TARGET_COL] + [TARGET_COL]
    out = out[ordered]

    meta = {"n_cols": n_cols, "columns": leak_cols}
    return out, {"items": leak_cols, "meta": meta}


def inject_missing(
    df: pd.DataFrame,
    frac: float = 0.1,
    mechanism: str = "MCAR",
    n_cols: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """对若干特征列按指定机制置 NaN。

    mechanism
        - ``"MCAR"``：完全随机缺失，每个目标列独立按 ``frac`` 抽样置空。
        - ``"MAR"``：缺失依赖另一可观测列（用一个驱动列的高值行更易缺失）。
        - ``"MNAR"``：缺失依赖列自身取值（高值行更易缺失）。

    items
        含缺失值的列名列表。
    meta
        记录 ``mechanism``、置空 cell 总数及每列计数。
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    feats = _feature_columns(out)
    n_cols = min(n_cols, len(feats))
    # 选择前 n_cols 个数值特征作为缺失目标（稳定可复现）。
    target_cols = list(rng.permutation(feats))[:n_cols]

    n = len(out)
    cells_per_col: dict[str, int] = {}
    total_cells = 0

    # MAR 需要一个驱动列：用一个不在 target_cols 中的特征。
    driver_col = None
    if mechanism == "MAR":
        candidates = [c for c in feats if c not in target_cols]
        driver_col = candidates[0] if candidates else feats[0]
        driver_vals = out[driver_col].to_numpy(dtype=float)
        driver_rank = pd.Series(driver_vals).rank(method="first").to_numpy()

    for col in target_cols:
        if mechanism == "MCAR":
            mask = rng.random(n) < frac
        elif mechanism == "MAR":
            # 驱动列排名越高，缺失概率越大；整体期望约为 frac。
            prob = (driver_rank / n) * (2.0 * frac)
            mask = rng.random(n) < prob
        elif mechanism == "MNAR":
            # 列自身取值越大越易缺失。
            vals = out[col].to_numpy(dtype=float)
            rank = pd.Series(vals).rank(method="first").to_numpy()
            prob = (rank / n) * (2.0 * frac)
            mask = rng.random(n) < prob
        else:
            raise ValueError(f"未知缺失机制：{mechanism!r}（应为 MCAR/MAR/MNAR）")

        count = int(mask.sum())
        if count:
            out.loc[mask, col] = np.nan
        cells_per_col[col] = count
        total_cells += count

    meta = {
        "mechanism": mechanism,
        "frac": frac,
        "n_cols": n_cols,
        "driver_col": driver_col,
        "cells_per_col": cells_per_col,
        "total_missing_cells": total_cells,
    }
    return out, {"items": target_cols, "meta": meta}


def inject_label_noise(
    df: pd.DataFrame,
    rate: float = 0.05,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """随机翻转 ``rate`` 比例的 target 标签（0<->1）。

    items
        ``["present"]``。
    meta
        记录被翻转行的位置序号 ``flipped_indices`` 与 ``rate``。
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    n_flip = int(round(n * rate))

    positions = rng.choice(n, size=n_flip, replace=False) if n_flip > 0 else np.array([], dtype=int)
    positions = np.sort(positions)

    if n_flip > 0:
        col_idx = out.columns.get_loc(TARGET_COL)
        cur = out.iloc[positions, col_idx].to_numpy(dtype=int)
        out.iloc[positions, col_idx] = 1 - cur

    meta = {"rate": rate, "n_flipped": int(n_flip), "flipped_indices": [int(p) for p in positions]}
    return out, {"items": ["present"], "meta": meta}


def inject_near_duplicate(
    df: pd.DataFrame,
    n_pairs: int = 50,
    noise_std: float = 0.01,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """复制若干样本并加微小高斯扰动后追加到表尾。

    每复制一行，新行位置序号为原表长度递增。标签保持不变，仅数值特征加噪。

    items
        ``contracts.normalize_pair(新行位置, 源行位置)`` 得到的 "i-j" 列表。
    meta
        记录对数与 ``noise_std``。

    Notes
    -----
    本函数会**改变行数**。为保证 ground_truth 的位置序号与最终落盘行号一致，
    调用方应在所有不改变行集合的缺陷之后再调用本函数，且之后不再增删行、
    并在落盘前做一次 ``reset_index(drop=True)``（本函数返回值已是连续索引）。
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)
    n_pairs = min(n_pairs, n)

    feats = _feature_columns(out)
    src_positions = rng.choice(n, size=n_pairs, replace=False) if n_pairs > 0 else np.array([], dtype=int)

    new_rows = []
    for src in src_positions:
        row = out.iloc[int(src)].copy()
        noise = rng.normal(0.0, noise_std, size=len(feats))
        row[feats] = row[feats].to_numpy(dtype=float) + noise
        new_rows.append(row)

    pairs: list[str] = []
    if new_rows:
        dup_df = pd.DataFrame(new_rows).reset_index(drop=True)
        out = pd.concat([out, dup_df], ignore_index=True)
        for offset, src in enumerate(src_positions):
            new_pos = n + offset
            pairs.append(normalize_pair(int(new_pos), int(src)))

    meta = {"n_pairs": int(n_pairs), "noise_std": noise_std}
    return out, {"items": pairs, "meta": meta}


def inject_class_imbalance(
    df: pd.DataFrame,
    minority_frac: float = 0.1,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """对少数类欠采样，使其占比降到目标比例附近，制造类别不平衡。

    以当前样本量较小的类别为少数类，对其欠采样，令
    ``minority / total ≈ minority_frac``。

    items
        ``["present"]``。
    meta
        记录原始与当前的类别分布。
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)

    counts = out[TARGET_COL].value_counts()
    orig_dist = {int(k): int(v) for k, v in counts.items()}

    minority_cls = int(counts.idxmin())
    majority_cls = int(counts.idxmax())
    n_majority = int(counts[majority_cls])

    # 由 minority/(minority+majority)=minority_frac 解出目标少数类样本数。
    target_minority = int(round(minority_frac / (1.0 - minority_frac) * n_majority))
    cur_minority = int(counts[minority_cls])
    target_minority = max(1, min(cur_minority, target_minority))

    minority_pos = np.where(out[TARGET_COL].to_numpy() == minority_cls)[0]
    keep = rng.choice(minority_pos, size=target_minority, replace=False)
    keep_set = set(int(p) for p in keep)

    drop_positions = [p for p in minority_pos if int(p) not in keep_set]
    out = out.drop(index=drop_positions).reset_index(drop=True)

    new_counts = out[TARGET_COL].value_counts()
    cur_dist = {int(k): int(v) for k, v in new_counts.items()}

    meta = {
        "minority_frac": minority_frac,
        "minority_class": minority_cls,
        "majority_class": majority_cls,
        "original_distribution": orig_dist,
        "current_distribution": cur_dist,
        "n_dropped": len(drop_positions),
    }
    return out, {"items": ["present"], "meta": meta}


# 用于注入格式不一致时的若干变换器：把一个数值改写成不一致格式的字符串。
# 关键约束：所有变体的输出都必须是**无法直接解析回数字**的字符串，否则
# 落盘并重新读取 CSV 时 pandas 会把整列解析回数值 dtype，缺陷就消失了。
# 基座特征数值通常较小（约 [-5, 5]），因此仅靠千分位分组并不可靠，
# 必须显式引入逗号小数点 / 单位后缀 / 货币前缀 / 多余空白等非数字标记。
def _to_thousands(v: float) -> str:
    """千分位 + 两位小数，并以逗号作小数点，如 1234.5 -> "1,234,50"。

    始终包含逗号小数点，保证小数值（如 0.5 -> "0,50"）也不会被解析回数字。
    """
    s = f"{v:,.2f}"          # 例如 "1,234.50" 或 "0.50"
    return s.replace(".", ",")  # -> "1,234,50" / "0,50"，含逗号，非数字


def _to_unit(v: float) -> str:
    """带单位后缀写法，如 12.3 -> "12.30kg"。"""
    return f"{v:.2f}kg"


def _to_comma_decimal(v: float) -> str:
    """欧式小数写法（逗号作小数点），如 12.34 -> "12,34"。"""
    return f"{v:.2f}".replace(".", ",")


def _to_currency(v: float) -> str:
    """货币前缀写法，如 12.3 -> "$12.30"。"""
    return f"${v:.2f}"


def _to_padded(v: float) -> str:
    """带多余空白与单位的写法，如 12.3 -> "  12.30 units "。"""
    return f"  {v:.2f} units "


_FORMAT_VARIANTS = (_to_thousands, _to_unit, _to_comma_decimal, _to_currency, _to_padded)


def inject_format_inconsistency(
    df: pd.DataFrame,
    n_cols: int = 3,
    frac: float = 0.1,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """把若干数值列的部分行改写为不一致格式的字符串，使列变为 object 混合类型。

    items
        受影响的列名列表。
    meta
        记录每列被改写的行位置序号与所用格式变体。
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    feats = _feature_columns(out)
    n_cols = min(n_cols, len(feats))
    target_cols = list(rng.permutation(feats))[:n_cols]

    n = len(out)
    affected_rows: dict[str, list[int]] = {}
    variants_used: dict[str, str] = {}

    for col in target_cols:
        # 整列先转成 object，避免逐格赋值触发 dtype 警告/截断。
        out[col] = out[col].astype(object)
        mask = rng.random(n) < frac
        positions = np.where(mask)[0]
        variant = _FORMAT_VARIANTS[int(rng.integers(0, len(_FORMAT_VARIANTS)))]
        for p in positions:
            try:
                out.iat[int(p), out.columns.get_loc(col)] = variant(float(df.iloc[int(p)][col]))
            except (ValueError, TypeError):
                # 原值已非数（理论上不会发生），跳过该格。
                continue
        affected_rows[col] = [int(p) for p in positions]
        variants_used[col] = variant.__name__

    meta = {
        "n_cols": n_cols,
        "frac": frac,
        "affected_rows": affected_rows,
        "variants": variants_used,
    }
    return out, {"items": target_cols, "meta": meta}
