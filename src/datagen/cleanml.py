"""CleanML 真实数据集加载器：将脏/净配对的表格分类数据转为统一的 DatasetBundle 结构。

每个 CleanML 数据集都包含一份干净的 ground-truth 数据和一份注入实际
脏数据类型的训练集。本模块负责读取原始 CSV、按需重命名目标列，并落盘为
与 :mod:`src.datagen.generate` 兼容的文件布局（dirty_train.csv /
clean_train.csv / clean_test.csv / ground_truth.json）。

用法::

    bundle = load_cleanml("Credit", out_dir="data/synthetic")
    print(bundle.dirty_train)   # -> data/synthetic/Credit/dirty_train.csv
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
    TARGET_COL,
    DatasetBundle,
    empty_report,
)

# CleanML 数据集的根目录（相对于项目根）
_CLEANML_ROOT = Path("data/CleanML/data")

# 各数据集的默认目标列名（加载时会统一重命名为 TARGET_COL）
CLEANML_TARGET_COLS: dict[str, str] = {
    "Credit": "SeriousDlqin2yrs",
    "EEG": "Eye",
    "Marketing": "Income",
}


def load_cleanml(name: str, out_dir: str = "data/synthetic", seed: int = 42) -> DatasetBundle:
    """加载一个 CleanML 数据集并写入统一的目录结构。

    参数
    ----------
    name : str
        数据集名称，须在 CLEANML_TARGET_COLS 中定义。
    out_dir : str
        输出根目录（默认为 data/synthetic），其下会创建 ``{name}/``
        子目录存放 dirty_train / clean_train / clean_test / ground_truth。
    seed : int
        用于 Marketing 数据集 train/test 切分的随机种子。

    返回
    -------
    DatasetBundle
        指向 ``out_dir/{name}/`` 的 bundle。
    """
    if name not in CLEANML_TARGET_COLS:
        msg = f"未知 CleanML 数据集：{name}，可用：{list(CLEANML_TARGET_COLS)}"
        raise ValueError(msg)

    target_col = CLEANML_TARGET_COLS[name]
    src_dir = _CLEANML_ROOT / name / "raw"

    # ---- 读取干净 ground truth ----
    clean_full: pd.DataFrame = pd.read_csv(src_dir / "raw.csv")

    # ---- 根据数据集类型选择 dirty / split 策略 ----
    if name in ("Credit", "EEG"):
        # 有预定义的 train/test 切分
        dirty_train = pd.read_csv(src_dir / "dirty_train.csv")
        idx_train = pd.read_csv(src_dir / "idx_train.csv")
        idx_test = pd.read_csv(src_dir / "idx_test.csv")

        # 用 idx 文件从 clean_full 中取出对应的子集
        clean_train = clean_full.iloc[idx_train.iloc[:, 0]].reset_index(drop=True)
        clean_test = clean_full.iloc[idx_test.iloc[:, 0]].reset_index(drop=True)

    elif name == "Marketing":
        # 无预定义切分；orgin.csv 即为脏版本
        dirty_train = pd.read_csv(src_dir / "orgin.csv")
        clean_train, clean_test = train_test_split(
            clean_full, test_size=0.2, random_state=seed
        )

    # ---- 统一目标列名为 TARGET_COL ----
    rename_map = {target_col: TARGET_COL}
    dirty_train = dirty_train.rename(columns=rename_map)
    clean_train = clean_train.rename(columns=rename_map)
    clean_test = clean_test.rename(columns=rename_map)

    # ---- 落盘 ----
    dataset_dir = Path(out_dir) / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dirty_train.to_csv(dataset_dir / F_DIRTY_TRAIN, index=False)
    clean_train.to_csv(dataset_dir / F_CLEAN_TRAIN, index=False)
    clean_test.to_csv(dataset_dir / F_CLEAN_TEST, index=False)

    with open(dataset_dir / F_GROUND_TRUTH, "w", encoding="utf-8") as f:
        json.dump(empty_report(), f, ensure_ascii=False, indent=2)

    return DatasetBundle(
        dataset_id=name,
        difficulty="real",
        root=dataset_dir,
        n_rows=len(clean_full),
        n_features=len(clean_full.columns) - 1,
    )
