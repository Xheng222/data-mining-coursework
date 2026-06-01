"""datagen.generate_dataset 的行为测试。

注意：合并前 generate_dataset 在本分支仍是桩（raise NotImplementedError），
本文件保证按约定接口书写、语法/import 正确，由编排者合并后统一运行。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.contracts import (
    DEFECT_TYPES,
    F_CLEAN_TEST,
    F_CLEAN_TRAIN,
    F_DIRTY_TRAIN,
    F_GROUND_TRUTH,
    F_META,
    TARGET_COL,
)
from src.datagen import generate_dataset


N_ROWS = 300
N_FEATURES = 10
TEST_FRAC = 0.2


@pytest.fixture
def bundle(tmp_path):
    """生成一个小型 easy 数据集，供本模块多个用例复用。"""
    return generate_dataset(
        "t",
        difficulty="easy",
        seed=0,
        out_dir=tmp_path,
        n_rows=N_ROWS,
        n_features=N_FEATURES,
    )


def test_all_files_exist(bundle):
    """四个 CSV + ground_truth.json + meta.json 都应落盘。"""
    root = bundle.root
    assert bundle.dirty_train.exists()
    assert bundle.clean_train.exists()
    assert bundle.clean_test.exists()
    assert bundle.ground_truth_path.exists()
    assert (root / F_META).exists()

    # 文件名常量应与约定一致
    assert bundle.dirty_train.name == F_DIRTY_TRAIN
    assert bundle.clean_train.name == F_CLEAN_TRAIN
    assert bundle.clean_test.name == F_CLEAN_TEST
    assert bundle.ground_truth_path.name == F_GROUND_TRUTH


def test_bundle_metadata(bundle):
    """DatasetBundle 标量字段：id/difficulty 回传参数；n_rows/n_features 反映脏训练集实际规模。

    注：注入近似重复会增行、类别不平衡欠采样会减行，因此 n_rows 不等于入参 N_ROWS，
    而应与落盘的 dirty_train 行数一致；n_features 同理对应脏训练集的特征列数。
    """
    import pandas as pd

    assert bundle.dataset_id == "t"
    assert bundle.difficulty == "easy"

    dirty = pd.read_csv(bundle.dirty_train)
    assert bundle.n_rows == len(dirty)
    assert bundle.n_features == dirty.shape[1] - 1  # 去掉 target 列
    assert bundle.n_rows > 0 and bundle.n_features > 0


def test_ground_truth_schema(bundle):
    """ground_truth 应包含全部 DEFECT_TYPES，且每项是 {items, meta}。"""
    gt = json.loads(bundle.ground_truth_path.read_text(encoding="utf-8"))
    assert set(gt.keys()) == set(DEFECT_TYPES)
    for t in DEFECT_TYPES:
        entry = gt[t]
        assert "items" in entry
        assert "meta" in entry
        assert isinstance(entry["items"], list)
        assert isinstance(entry["meta"], dict)


def test_dirty_train_has_missing(bundle):
    """缺失注入应生效：dirty_train 至少含一个 NaN。"""
    dirty = pd.read_csv(bundle.dirty_train)
    assert dirty.isna().to_numpy().any(), "dirty_train 应包含至少一个缺失值"


def test_leak_column_present(bundle):
    """泄漏注入应生效：dirty_train 至少有一个以 'leak' 开头的列。"""
    dirty = pd.read_csv(bundle.dirty_train)
    leak_cols = [c for c in dirty.columns if str(c).startswith("leak")]
    assert leak_cols, f"期望存在以 'leak' 开头的列，实际列：{list(dirty.columns)}"

    # 泄漏列也应被记录在 ground truth 的 leakage.items 中
    gt = json.loads(bundle.ground_truth_path.read_text(encoding="utf-8"))
    assert set(leak_cols) & set(gt["leakage"]["items"])


def test_clean_test_row_count(bundle):
    """clean_test 行数 ≈ n_rows * 0.2（允许 ±5% 的取整/划分误差）。"""
    test_df = pd.read_csv(bundle.clean_test)
    expected = N_ROWS * TEST_FRAC
    assert test_df.shape[0] == pytest.approx(expected, abs=max(2, 0.05 * N_ROWS))


def test_target_column_present(bundle):
    """三个数据划分都应含 target 列。"""
    for path in (bundle.dirty_train, bundle.clean_train, bundle.clean_test):
        df = pd.read_csv(path)
        assert TARGET_COL in df.columns, f"{path.name} 缺少 {TARGET_COL} 列"


def test_clean_test_is_clean(bundle):
    """冻结测试集应当无缺失值（全程干净）。"""
    test_df = pd.read_csv(bundle.clean_test)
    assert not test_df.isna().to_numpy().any(), "clean_test 不应含缺失值"
