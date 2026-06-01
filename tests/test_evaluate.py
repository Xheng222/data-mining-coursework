"""evaluate 模块测试：train_and_auc / detection_scores，并顺带 recovery_rate。

注意：合并前 evaluate 在本分支仍是桩，本文件保证按约定接口书写、断言合理。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts import (
    DetectionScore,
    TARGET_COL,
    normalize_pair,
    recovery_rate,
)
from src.evaluate import detection_scores, train_and_auc


# --------------------------------------------------------------------------- #
# train_and_auc
# --------------------------------------------------------------------------- #

def _make_separable(n: int, seed: int) -> pd.DataFrame:
    """造一个线性可分的小数据集：两个特征随类别整体平移。"""
    rng = np.random.default_rng(seed)
    half = n // 2
    neg = rng.normal(loc=-3.0, scale=0.5, size=(half, 2))
    pos = rng.normal(loc=3.0, scale=0.5, size=(n - half, 2))
    X = np.vstack([neg, pos])
    y = np.array([0] * half + [1] * (n - half))
    df = pd.DataFrame(X, columns=["f0", "f1"])
    df[TARGET_COL] = y
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_train_and_auc_range_and_separable():
    train = _make_separable(200, seed=1)
    test = _make_separable(80, seed=2)
    auc = train_and_auc(train, test, target_col=TARGET_COL)
    assert 0.0 <= auc <= 1.0
    # 线性可分数据上，AUC 应明显优于随机
    assert auc > 0.5


# --------------------------------------------------------------------------- #
# detection_scores
# --------------------------------------------------------------------------- #

def _report(**types) -> dict:
    """便捷构造 DefectReport：仅填入给定缺陷类型的 items。"""
    return {t: {"items": list(items), "meta": {}} for t, items in types.items()}


def test_detection_scores_partial_hit():
    """构造部分命中 + 部分漏报 + 一个误报，核对手算 P/R。

    ground truth：
        missing  = {f1, f2}
        leakage  = {leak0}
    reported：
        missing  = {f1, f3}   -> tp=1 (f1), fp=1 (f3), fn=1 (f2)
        leakage  = {}         -> tp=0,      fp=0,      fn=1 (leak0)

    micro 汇总：tp=1, fp=1, fn=2
        precision = 1/(1+1) = 0.5
        recall    = 1/(1+2) = 1/3
    """
    gt = _report(missing=["f1", "f2"], leakage=["leak0"])
    reported = _report(missing=["f1", "f3"], leakage=[])

    score = detection_scores(reported, gt)
    assert isinstance(score, DetectionScore)

    # per_type: missing
    m = score.per_type["missing"]
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)

    # per_type: leakage（全漏报）
    lk = score.per_type["leakage"]
    assert lk["tp"] == 0
    assert lk["fn"] == 1
    assert lk["recall"] == pytest.approx(0.0)

    # micro 平均
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(1.0 / 3.0)
    expected_f1 = 2 * 0.5 * (1.0 / 3.0) / (0.5 + 1.0 / 3.0)
    assert score.f1 == pytest.approx(expected_f1)


def test_detection_scores_perfect_match():
    """完全命中时 P/R/F1 均为 1。"""
    gt = _report(
        missing=["f1"],
        near_duplicate=[normalize_pair(0, 5), normalize_pair(3, 9)],
        label_noise=["present"],
    )
    reported = _report(
        missing=["f1"],
        near_duplicate=[normalize_pair(5, 0), normalize_pair(9, 3)],
        label_noise=["present"],
    )
    score = detection_scores(reported, gt)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.f1 == pytest.approx(1.0)


def test_detection_scores_empty_truth_and_report():
    """ground truth 与 report 都为空：约定 P/R 取 1.0（无可判定为完美）或 0.0。

    这里只断言取值在 [0,1] 且不抛异常，避免对边界约定过度耦合。
    """
    gt = _report()
    reported = _report()
    score = detection_scores(reported, gt)
    assert 0.0 <= score.precision <= 1.0
    assert 0.0 <= score.recall <= 1.0
    assert 0.0 <= score.f1 <= 1.0


# --------------------------------------------------------------------------- #
# recovery_rate（公式核对，独立于桩实现）
# --------------------------------------------------------------------------- #

def test_recovery_rate_values():
    # method 恰好恢复一半差距
    assert recovery_rate(0.6, 0.8, 1.0) == pytest.approx(0.5)
    # method 完全恢复到 clean 水平
    assert recovery_rate(0.6, 1.0, 1.0) == pytest.approx(1.0)
    # method 等于 dirty，未恢复
    assert recovery_rate(0.6, 0.6, 1.0) == pytest.approx(0.0)
    # 分母过小时安全返回 0.0
    assert recovery_rate(0.7, 0.9, 0.7) == 0.0
    # method 反而更差，可为负
    assert recovery_rate(0.6, 0.5, 1.0) == pytest.approx(-0.25)
