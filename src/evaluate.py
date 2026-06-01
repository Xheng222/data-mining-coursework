"""评测指标：Detection(P/R) 与 Recovery Rate。feature/evaluate 负责实现。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.contracts import DefectReport, DetectionScore


def train_and_auc(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "target",
) -> float:
    """在 train 上训练 XGBoost，在固定的 test 上返回 AUC-ROC。"""
    raise NotImplementedError


def detection_scores(reported: DefectReport, ground_truth: DefectReport) -> DetectionScore:
    """按 contracts 的 items 集合，分缺陷类型计算 P/R/F1 并 micro 平均。"""
    raise NotImplementedError


def load_ground_truth(path: str | Path) -> DefectReport:
    raise NotImplementedError
