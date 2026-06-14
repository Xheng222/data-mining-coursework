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

# 内部子目录映射：对外名称 → 实际数据子目录（用于使用 _major 标签噪声变体）
_CLEANML_SRC_MAP: dict[str, str] = {
    "Credit": "Credit_major",
    "EEG": "EEG_major",
    "Marketing": "Marketing",
}


def _load_major_variant(name: str, src_dir: Path, target_col: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载 _major 标签噪声变体（dirty=raw.csv, clean=mislabel_clean_raw.csv）。

    _major 变体不提供预定义的 train/test 切分，需要在这里创建。

    返回 (dirty_train, clean_train, clean_test)。
    """
    dirty_full = pd.read_csv(src_dir / "raw.csv")
    clean_full = pd.read_csv(src_dir / "mislabel_clean_raw.csv")

    # 先切分 clean data
    clean_train, clean_test = train_test_split(
        clean_full, test_size=0.2, random_state=seed
    )

    # dirty 用同样的行索引（dirty 与 clean 行数一致且按行对齐）
    dirty_train = dirty_full.iloc[clean_train.index].reset_index(drop=True)
    clean_train = clean_train.reset_index(drop=True)
    clean_test = clean_test.reset_index(drop=True)

    return dirty_train, clean_train, clean_test


def _load_marketing(src_dir: Path, target_col: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载 Marketing（orgin.csv = 脏, raw.csv = 净）。

    train/test 按 80/20 随机切分。
    """
    dirty_full = pd.read_csv(src_dir / "orgin.csv")
    clean_full = pd.read_csv(src_dir / "raw.csv")

    # dirty 与 clean 行数一致且按行对齐
    clean_train, clean_test = train_test_split(
        clean_full, test_size=0.2, random_state=seed
    )
    dirty_train = dirty_full.iloc[clean_train.index].reset_index(drop=True)
    clean_train = clean_train.reset_index(drop=True)
    clean_test = clean_test.reset_index(drop=True)

    return dirty_train, clean_train, clean_test


def load_cleanml_clean(name: str) -> pd.DataFrame:
    """仅加载 CleanML 数据集的**干净版本**，返回完整 DataFrame。

    与 :func:`load_cleanml` 不同，本函数不切分 train/test、不落盘、
    不返回 bundle；只将干净 CSV 读入内存，以便后续注入受控缺陷
    （作为 ``generate_dataset(…, base_df=…)`` 的基座输入）。

    参数
    ----------
    name : str
        数据集名称，须在 CLEANML_TARGET_COLS 中定义。

    返回
    -------
    pd.DataFrame
        未经切分的干净 DataFrame（仍保留原始列名，未重命名为 TARGET_COL）。
    """
    if name not in CLEANML_TARGET_COLS:
        msg = f"未知 CleanML 数据集：{name}，可用：{list(CLEANML_TARGET_COLS)}"
        raise ValueError(msg)

    target_col = CLEANML_TARGET_COLS[name]
    src_subdir = _CLEANML_SRC_MAP[name]
    src_dir = _CLEANML_ROOT / src_subdir / "raw"

    if name in ("Credit", "EEG"):
        # _major 变体：mislabel_clean_raw.csv 为干净版本（原始文件名中的 clean）
        clean_full = pd.read_csv(src_dir / "mislabel_clean_raw.csv")
    elif name == "Marketing":
        clean_full = pd.read_csv(src_dir / "raw.csv")

    rename_map = {target_col: TARGET_COL}
    return clean_full.rename(columns=rename_map)


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
        用于 train/test 切分的随机种子。

    返回
    -------
    DatasetBundle
        指向 ``out_dir/{name}/`` 的 bundle。
    """
    if name not in CLEANML_TARGET_COLS:
        msg = f"未知 CleanML 数据集：{name}，可用：{list(CLEANML_TARGET_COLS)}"
        raise ValueError(msg)

    target_col = CLEANML_TARGET_COLS[name]
    src_subdir = _CLEANML_SRC_MAP[name]
    src_dir = _CLEANML_ROOT / src_subdir / "raw"

    # ---- 根据数据集选择加载策略 ----
    if name in ("Credit", "EEG"):
        dirty_train, clean_train, clean_test = _load_major_variant(name, src_dir, target_col, seed)
    elif name == "Marketing":
        dirty_train, clean_train, clean_test = _load_marketing(src_dir, target_col, seed)

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
        n_rows=len(clean_train) + len(clean_test),
        n_features=len(clean_train.columns) - 1,
    )
