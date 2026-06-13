"""合成数据集生成入口。

``generate_dataset`` 负责把基座生成、缺陷注入、数据划分与落盘串成完整流程，
最终返回一个 :class:`src.contracts.DatasetBundle`。落盘产物包括：

    - ``dirty_train.csv``   注入六类缺陷后的脏训练集
    - ``clean_train.csv``   同一划分但未注入缺陷的训练集（性能上界用）
    - ``clean_test.csv``    全程冻结的干净测试集
    - ``ground_truth.json`` 完整的 DefectReport（六类型聚合）
    - ``meta.json``         数据集元信息与各缺陷参数
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.contracts import (
    F_CLEAN_TEST,
    F_CLEAN_TRAIN,
    F_DIRTY_TRAIN,
    F_GROUND_TRUTH,
    F_META,
    TARGET_COL,
    DatasetBundle,
    DefectReport,
    empty_report,
)
from src.datagen.base import make_base
from src.datagen.defects import (
    inject_class_imbalance,
    inject_format_inconsistency,
    inject_label_noise,
    inject_leakage,
    inject_missing,
    inject_near_duplicate,
)

# --------------------------------------------------------------------------- #
# 难度预设：控制各缺陷的注入强度，easy < medium < hard 单调递增。
# --------------------------------------------------------------------------- #
DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy": {
        "leakage": {"n_cols": 1},
        "missing": {"frac": 0.05, "mechanism": "MCAR", "n_cols": 3},
        "label_noise": {"rate": 0.02},
        "near_duplicate": {"n_pairs": 20, "noise_std": 0.01},
        "class_imbalance": {"minority_frac": 0.3},
        "format_inconsistency": {"n_cols": 1, "frac": 0.05},
    },
    "medium": {
        "leakage": {"n_cols": 2},
        "missing": {"frac": 0.10, "mechanism": "MAR", "n_cols": 5},
        "label_noise": {"rate": 0.05},
        "near_duplicate": {"n_pairs": 50, "noise_std": 0.01},
        "class_imbalance": {"minority_frac": 0.15},
        "format_inconsistency": {"n_cols": 3, "frac": 0.10},
    },
    "hard": {
        "leakage": {"n_cols": 4},
        "missing": {"frac": 0.20, "mechanism": "MNAR", "n_cols": 8},
        "label_noise": {"rate": 0.10},
        "near_duplicate": {"n_pairs": 120, "noise_std": 0.02},
        "class_imbalance": {"minority_frac": 0.05},
        "format_inconsistency": {"n_cols": 5, "frac": 0.20},
    },
}


def generate_dataset(
    dataset_id: str,
    difficulty: str = "medium",
    seed: int = 42,
    out_dir: str | Path = "data/synthetic",
    n_rows: int = 10000,
    n_features: int = 30,
    base_df: pd.DataFrame | None = None,
) -> DatasetBundle:
    """生成一个带已知缺陷的合成分类数据集，落盘并返回 :class:`DatasetBundle`。

    流程
    ----
    1. 若提供 ``base_df`` 则直接使用（真实数据基座），否则 ``make_base`` 造干净基座。
    2. 分层 ``train_test_split``（test_size=0.2）切出冻结干净测试集与干净训练集。
    3. 干净训练集复制两份：一份原样存为 ``clean_train.csv``；另一份依次注入六类缺陷。
    4. 注入强度由 ``difficulty`` 预设决定。
    5. 落盘 4 个 CSV + ``ground_truth.json`` + ``meta.json``。

    Parameters
    ----------
    base_df : pd.DataFrame, optional
        外部提供的干净基座 DataFrame（必须含 ``target`` 列），
        用于真实数据基座场景；不传则用 ``make_base`` 生成合成基座。

    Returns
    -------
    DatasetBundle
        ``n_rows`` / ``n_features`` 取脏训练集落盘后的实际行列数（特征列，不含 target）。
    """
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(
            f"未知难度：{difficulty!r}（应为 {tuple(DIFFICULTY_PRESETS)} 之一）"
        )
    preset = DIFFICULTY_PRESETS[difficulty]

    # 1. 干净基座：真实数据 or 合成
    if base_df is not None:
        base = base_df.copy()
    else:
        base = make_base(n_rows=n_rows, n_features=n_features, seed=seed)

    # 2. 分层切出冻结的干净测试集与干净训练集
    train_clean, test_clean = train_test_split(
        base,
        test_size=0.2,
        stratify=base[TARGET_COL],
        random_state=seed,
    )
    train_clean = train_clean.reset_index(drop=True)
    test_clean = test_clean.reset_index(drop=True)

    # 3. clean_train 原样保留；dirty 在副本上注入
    dirty = train_clean.copy()
    report: DefectReport = empty_report()

    # 4. 依次注入。顺序经过设计以保证两点：
    #    (a) ground_truth 的行位置序号与最终落盘行号一致；
    #    (b) near_duplicate 复制行时特征列仍是纯数值（可加噪），故必须在
    #        format_inconsistency 把列改成混合字符串之前完成。
    #    具体顺序：
    #    - leakage：只增列。
    #    - missing：只改值。
    #    - class_imbalance：删行，之后 reset，行集合规模随后只增不减。
    #    - label_noise：在当前行集合上记录位置序号（near_duplicate 只追加行，
    #      不影响已有行的位置，故这些序号在最终表中保持有效）。
    #    - near_duplicate：追加行，新行位置序号 = 追加前长度起递增。
    #    - format_inconsistency：最后在完整（含重复行）的行集合上改写并记录位置。
    #      此后不再增删行，落盘行号与所有记录的位置序号一致。

    # leakage
    dirty, frag = inject_leakage(dirty, seed=seed + 1, **preset["leakage"])
    report["leakage"] = frag

    # missing
    dirty, frag = inject_missing(dirty, seed=seed + 2, **preset["missing"])
    report["missing"] = frag

    # class_imbalance（会删行）
    dirty, frag = inject_class_imbalance(
        dirty, seed=seed + 3, **preset["class_imbalance"]
    )
    report["class_imbalance"] = frag

    # 删行后统一 reset 一次，使后续记录的位置序号有连续的起点。
    dirty = dirty.reset_index(drop=True)

    # label_noise（记录行位置）
    dirty, frag = inject_label_noise(dirty, seed=seed + 4, **preset["label_noise"])
    report["label_noise"] = frag

    # near_duplicate（追加行；此时特征列仍为纯数值）
    dirty, frag = inject_near_duplicate(
        dirty, seed=seed + 5, **preset["near_duplicate"]
    )
    report["near_duplicate"] = frag

    # format_inconsistency（最后改写列类型并记录行位置，此后行集合固定）
    dirty, frag = inject_format_inconsistency(
        dirty, seed=seed + 6, **preset["format_inconsistency"]
    )
    report["format_inconsistency"] = frag

    # 落盘前最终 reset，确保连续行号（幂等保险）。
    dirty = dirty.reset_index(drop=True)

    # 5. 落盘
    root = Path(out_dir) / dataset_id
    root.mkdir(parents=True, exist_ok=True)

    dirty.to_csv(root / F_DIRTY_TRAIN, index=False)
    train_clean.to_csv(root / F_CLEAN_TRAIN, index=False)
    test_clean.to_csv(root / F_CLEAN_TEST, index=False)

    with (root / F_GROUND_TRUTH).open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    dirty_feature_cols = [c for c in dirty.columns if c != TARGET_COL]
    meta = {
        "dataset_id": dataset_id,
        "difficulty": difficulty,
        "seed": seed,
        "n_rows": int(len(dirty)),
        "n_features": int(len(dirty_feature_cols)),
        "target_col": TARGET_COL,
        "base_n_rows": n_rows,
        "base_n_features": n_features,
        "n_clean_train": int(len(train_clean)),
        "n_clean_test": int(len(test_clean)),
        "defect_params": preset,
    }
    with (root / F_META).open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # 6. 返回 bundle，行列数取脏训练集实际值。
    return DatasetBundle(
        dataset_id=dataset_id,
        difficulty=difficulty,
        root=root,
        n_rows=int(len(dirty)),
        n_features=int(len(dirty_feature_cols)),
    )
