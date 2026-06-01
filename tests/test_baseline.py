"""baseline 模块测试：run_no_clean / run_clean_upper / run_rule_based。

先用 datagen 造一个小数据集，再跑三种基线。
注意：合并前相关模块仍是桩，本文件保证按约定接口书写、断言合理。
"""

from __future__ import annotations

import pytest

from src.baseline import run_clean_upper, run_no_clean, run_rule_based
from src.contracts import MethodOutcome
from src.datagen import generate_dataset


@pytest.fixture
def dataset_root(tmp_path):
    """造一个小型脏数据集，返回其落盘根目录。"""
    bundle = generate_dataset(
        "b",
        difficulty="easy",
        seed=0,
        out_dir=tmp_path,
        n_rows=400,
        n_features=10,
    )
    return bundle.root


def _check_outcome(outcome, expected_method):
    assert isinstance(outcome, MethodOutcome)
    assert outcome.method == expected_method or expected_method in outcome.method
    assert 0.0 <= outcome.auc <= 1.0


def test_run_no_clean(dataset_root):
    outcome = run_no_clean(dataset_root)
    _check_outcome(outcome, "no_clean")


def test_run_clean_upper(dataset_root):
    outcome = run_clean_upper(dataset_root)
    _check_outcome(outcome, "clean_upper")


def test_run_rule_based(dataset_root):
    outcome = run_rule_based(dataset_root)
    _check_outcome(outcome, "rule_based")

    # rule_based 应产出 detection 结果
    assert outcome.detection is not None
    assert 0.0 <= outcome.detection.precision <= 1.0
    assert 0.0 <= outcome.detection.recall <= 1.0


def test_rule_based_reports_missing_or_leakage(dataset_root):
    """规则基线至少应报告缺失或泄漏（这是规则清洗最基本的能力）。"""
    outcome = run_rule_based(dataset_root)
    det = outcome.detection
    assert det is not None
    missing_tp = det.per_type["missing"]["tp"]
    leakage_tp = det.per_type["leakage"]["tp"]
    assert (missing_tp > 0) or (leakage_tp > 0), (
        "rule_based 至少应命中缺失或泄漏中的一类"
    )


def test_clean_upper_not_worse_than_no_clean(dataset_root):
    """一般而言干净上界 AUC 应不低于脏数据下界。

    用 >= 加一个宽松容差，避免小样本下偶发翻转导致 flaky。
    """
    upper = run_clean_upper(dataset_root)
    lower = run_no_clean(dataset_root)
    tol = 0.05
    assert upper.auc >= lower.auc - tol, (
        f"clean_upper.auc={upper.auc} 不应明显低于 no_clean.auc={lower.auc}"
    )
