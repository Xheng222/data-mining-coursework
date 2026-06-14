"""基线方法与命令行入口。

基线：
  - no_clean    ：脏数据直接训练（性能下界）
  - rule_based  ：调用 preprocess.rule_based_clean 清洗后训练
  - clean_upper ：干净训练集直接训练（性能上界）

CLI（中期一键复现）：
  uv run python -m src.baseline --dataset data/synthetic/<id> --out results/baseline.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from src.contracts import (
    F_CLEAN_TEST,
    F_CLEAN_TRAIN,
    F_DIRTY_TRAIN,
    F_GROUND_TRUTH,
    MethodOutcome,
    recovery_rate,
)
from src.evaluate import detection_scores, load_ground_truth, train_and_auc
from src.preprocess import rule_based_clean

# 规则清洗后训练集的落盘目录。
_PROCESSED_DIR = Path("data/processed")


def _read_csv(path: Path) -> pd.DataFrame:
    """读取一个数据集 CSV 文件。"""
    return pd.read_csv(path, encoding="utf-8")


def run_no_clean(dataset_root: str | Path, model_name: str = "xgb") -> MethodOutcome:
    """脏训练集直接训练，得到性能下界。"""
    root = Path(dataset_root)
    dirty_train = _read_csv(root / F_DIRTY_TRAIN)
    clean_test = _read_csv(root / F_CLEAN_TEST)
    auc = train_and_auc(dirty_train, clean_test, model_name=model_name)
    return MethodOutcome(dataset_id=root.name, method="no_clean", model=model_name, auc=auc)


def run_clean_upper(dataset_root: str | Path, model_name: str = "xgb") -> MethodOutcome:
    """干净训练集直接训练，得到性能上界。"""
    root = Path(dataset_root)
    clean_train = _read_csv(root / F_CLEAN_TRAIN)
    clean_test = _read_csv(root / F_CLEAN_TEST)
    auc = train_and_auc(clean_train, clean_test, model_name=model_name)
    return MethodOutcome(dataset_id=root.name, method="clean_upper", model=model_name, auc=auc)


def run_rule_based(dataset_root: str | Path, model_name: str = "xgb") -> MethodOutcome:
    """规则清洗后训练，并计算缺陷检测指标。

    清洗后的 DataFrame 由本函数负责落盘到
    ``data/processed/<dataset_id>_rule.csv``（index=False），再用于训练。
    """
    root = Path(dataset_root)
    dataset_id = root.name
    dirty_train = _read_csv(root / F_DIRTY_TRAIN)
    clean_test = _read_csv(root / F_CLEAN_TEST)

    result = rule_based_clean(dirty_train)
    cleaned_df: pd.DataFrame = result.extra["cleaned_df"]

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PROCESSED_DIR / f"{dataset_id}_rule.csv"
    cleaned_df.to_csv(out_path, index=False)

    auc = train_and_auc(cleaned_df, clean_test, model_name=model_name)
    detection = detection_scores(
        result.reported_defects,
        load_ground_truth(root / F_GROUND_TRUTH),
    )
    return MethodOutcome(
        dataset_id=dataset_id,
        method="rule_based",
        model=model_name,
        auc=auc,
        detection=detection,
    )


def _outcome_to_dict(outcome: MethodOutcome) -> dict:
    """把 MethodOutcome 序列化为可 JSON 化的 dict（含嵌套 DetectionScore）。"""
    data = dataclasses.asdict(outcome)
    return data


def main() -> None:
    """CLI 入口：在一个数据集上跑三种基线并把结果写入 JSON。"""
    parser = argparse.ArgumentParser(
        description="脏数据基准：no_clean / rule_based / clean_upper 基线评测",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="数据集根目录，如 data/synthetic/<id>",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="结果 JSON 输出路径，如 results/baseline.json",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)

    no_clean = run_no_clean(dataset_root)
    clean_upper = run_clean_upper(dataset_root)
    rule = run_rule_based(dataset_root)

    # rule_based 的 Recovery Rate：相对 no_clean 下界与 clean_upper 上界的归一化。
    rule.recovery_rate = recovery_rate(no_clean.auc, rule.auc, clean_upper.auc)

    outcomes = [no_clean, rule, clean_upper]
    payload = {
        "dataset_id": dataset_root.name,
        "outcomes": [_outcome_to_dict(o) for o in outcomes],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已写入结果：{out_path}")


if __name__ == "__main__":
    main()
