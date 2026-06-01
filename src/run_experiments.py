"""端到端实验编排：生成数据 -> 跑各方法 -> 算指标 -> 落盘结果与图表。

CLI::

    uv run python -m src.run_experiments --config configs/midterm.yaml
    uv run python -m src.run_experiments --config configs/midterm.yaml --quick
    uv run python -m src.run_experiments --config configs/midterm.yaml --skip-agent --max-datasets 1

流程概述：

1. 读 YAML 配置，逐个数据集调 :func:`datagen.generate_dataset`。
2. 对每个数据集算 clean_upper（上界）、no_clean（下界）、rule_based（含检测）。
3. 若未 ``--skip-agent``，对每个 ablation 构造 :class:`AgentConfig` 跑 agent，
   再用清洗后训练集算 AUC、用报告的缺陷算 detection。agent 调用全程
   try/except，单个变体失败不影响其它结果。
4. 对 rule_based 与各 agent 变体算 Recovery Rate（相对 no_clean/clean_upper）。
5. 汇总落盘 ``results.csv`` / ``detection_by_type.csv`` / ``summary.json``。
6. 调 :mod:`src.viz` 生成图表到 ``results/figures/``。

设计要点：baseline 与 agent 解耦，即使 LLM 全部失败，基线结果照样产出。
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from src import viz
from src.agent import AgentConfig, run_agent
from src.baseline import run_clean_upper, run_no_clean, run_rule_based
from src.contracts import (
    TARGET_COL,
    DatasetBundle,
    DetectionScore,
    MethodOutcome,
    recovery_rate,
)
from src.datagen import generate_dataset
from src.evaluate import detection_scores, load_ground_truth, train_and_auc

# quick 冒烟模式下的小规模覆盖参数。
_QUICK_N_ROWS = 800
_QUICK_N_FEATURES = 12

# 默认清洗任务描述，传给 agent。
_AGENT_TASK = "请清洗该数据集并训练一个二分类模型"


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 实验配置为 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# 结果行 -> 扁平 dict（写 CSV / JSON 用）
# --------------------------------------------------------------------------- #

def _detection_to_dict(det: Optional[DetectionScore]) -> Optional[dict[str, Any]]:
    """把 DetectionScore 转成可 JSON 序列化的 dict（None 透传）。"""
    if det is None:
        return None
    return {
        "precision": det.precision,
        "recall": det.recall,
        "f1": det.f1,
        "per_type": det.per_type,
    }


def _outcome_summary_row(outcome: MethodOutcome) -> dict[str, Any]:
    """把单个 MethodOutcome 转成结构化 dict（写入 summary.json）。"""
    return {
        "dataset_id": outcome.dataset_id,
        "method": outcome.method,
        "auc": outcome.auc,
        "recovery_rate": outcome.recovery_rate,
        "detection": _detection_to_dict(outcome.detection),
        "extra": outcome.extra,
    }


def _outcome_results_row(outcome: MethodOutcome) -> dict[str, Any]:
    """把单个 MethodOutcome 转成 results.csv 的一行。"""
    det = outcome.detection
    return {
        "dataset_id": outcome.dataset_id,
        "method": outcome.method,
        "auc": outcome.auc,
        "recovery_rate": outcome.recovery_rate,
        "detection_precision": det.precision if det else None,
        "detection_recall": det.recall if det else None,
        "detection_f1": det.f1 if det else None,
    }


def _detection_by_type_rows(outcome: MethodOutcome) -> list[dict[str, Any]]:
    """把单个 MethodOutcome 的分缺陷类型检测分展开成多行。"""
    det = outcome.detection
    if det is None:
        return []
    rows: list[dict[str, Any]] = []
    for defect_type, scores in det.per_type.items():
        rows.append(
            {
                "dataset_id": outcome.dataset_id,
                "method": outcome.method,
                "defect_type": defect_type,
                "precision": scores.get("precision"),
                "recall": scores.get("recall"),
                "f1": scores.get("f1"),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 单个数据集的实验
# --------------------------------------------------------------------------- #

def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def run_dataset(
    spec: dict[str, Any],
    base: dict[str, Any],
    ablations: list[dict[str, Any]],
    model: str,
    skip_agent: bool,
    quick: bool,
    out_dir: Path,
) -> list[MethodOutcome]:
    """跑通单个数据集上的全部方法，返回该数据集的所有 MethodOutcome。"""
    dataset_id = spec["id"]
    difficulty = spec.get("difficulty", "medium")
    seed = int(spec.get("seed", 42))

    n_rows = _QUICK_N_ROWS if quick else int(base.get("n_rows", 10000))
    n_features = _QUICK_N_FEATURES if quick else int(base.get("n_features", 30))

    data_root = out_dir.parent / "data" / "synthetic"

    print(f"\n=== 数据集 {dataset_id} (difficulty={difficulty}, seed={seed}, "
          f"n_rows={n_rows}, n_features={n_features}) ===")
    print(f"[gen] 生成合成数据到 {data_root / dataset_id} ...")
    bundle: DatasetBundle = generate_dataset(
        dataset_id,
        difficulty=difficulty,
        seed=seed,
        out_dir=str(data_root),
        n_rows=n_rows,
        n_features=n_features,
    )
    dataset_root = bundle.root

    outcomes: list[MethodOutcome] = []

    # --- 上界 / 下界 / 规则基线 ----------------------------------------- #
    print("[clean_upper] 干净数据训练（AUC 上界）...")
    clean_upper = run_clean_upper(dataset_root)
    clean_upper.dataset_id = clean_upper.dataset_id or dataset_id
    outcomes.append(clean_upper)
    print(f"    AUC_clean = {clean_upper.auc:.4f}")

    print("[no_clean] 脏数据直接训练（AUC 下界）...")
    no_clean = run_no_clean(dataset_root)
    no_clean.dataset_id = no_clean.dataset_id or dataset_id
    outcomes.append(no_clean)
    print(f"    AUC_dirty = {no_clean.auc:.4f}")

    print("[rule_based] 规则清洗 + 训练 + 检测...")
    rule_based = run_rule_based(dataset_root)
    rule_based.dataset_id = rule_based.dataset_id or dataset_id
    outcomes.append(rule_based)
    print(f"    AUC_rule  = {rule_based.auc:.4f}")

    auc_dirty = no_clean.auc
    auc_clean = clean_upper.auc

    # rule_based 的 Recovery Rate。
    rule_based.recovery_rate = recovery_rate(auc_dirty, rule_based.auc, auc_clean)
    print(f"    recovery_rate(rule_based) = {rule_based.recovery_rate:.4f}")

    # --- Agent 消融 ----------------------------------------------------- #
    if skip_agent:
        print("[agent] 已跳过（--skip-agent）。")
        return outcomes

    ground_truth = load_ground_truth(bundle.ground_truth_path)
    clean_test = _read_csv(bundle.clean_test)

    for ab in ablations:
        name = ab.get("name", "agent")
        method_name = name if name.startswith("agent") else f"agent_{name}"
        config = AgentConfig(
            use_planner=bool(ab.get("use_planner", True)),
            use_reviewer=bool(ab.get("use_reviewer", True)),
            prompt_strategy=str(ab.get("prompt_strategy", "cot")),
            model=model,
        )
        print(f"[agent:{method_name}] use_planner={config.use_planner} "
              f"use_reviewer={config.use_reviewer} prompt={config.prompt_strategy} ...")

        outcome = MethodOutcome(dataset_id=dataset_id, method=method_name, auc=math.nan)
        try:
            result = run_agent(dataset_root, task=_AGENT_TASK, config=config)

            cleaned_train = _read_csv(result.cleaned_train_path)
            outcome.auc = train_and_auc(cleaned_train, clean_test, target_col=TARGET_COL)
            outcome.detection = detection_scores(result.reported_defects, ground_truth)
            outcome.recovery_rate = recovery_rate(auc_dirty, outcome.auc, auc_clean)
            outcome.extra = {"log": result.log}
            print(f"    AUC={outcome.auc:.4f} "
                  f"recovery_rate={outcome.recovery_rate:.4f} "
                  f"detection_f1={outcome.detection.f1:.4f}")
        except Exception as exc:  # noqa: BLE001 - 容错：单个 agent 变体失败不影响整体
            print(f"    [WARN] agent 变体 {method_name} 在 {dataset_id} 上失败：{exc}")
            traceback.print_exc()
            # auc 置 NaN、detection/recovery 置 None，其余结果照常产出。
            outcome.auc = math.nan
            outcome.detection = None
            outcome.recovery_rate = None
            outcome.extra = {"error": str(exc)}

        outcomes.append(outcome)

    return outcomes


# --------------------------------------------------------------------------- #
# 落盘与可视化
# --------------------------------------------------------------------------- #

def write_results(all_outcomes: list[MethodOutcome], out_dir: Path) -> dict[str, Path]:
    """把全部 MethodOutcome 写成 CSV / JSON，返回写出的文件路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    results_rows = [_outcome_results_row(o) for o in all_outcomes]
    results_df = pd.DataFrame(results_rows)
    results_path = out_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[write] results -> {results_path} ({len(results_df)} 行)")

    detection_rows: list[dict[str, Any]] = []
    for o in all_outcomes:
        detection_rows.extend(_detection_by_type_rows(o))
    detection_df = pd.DataFrame(
        detection_rows,
        columns=["dataset_id", "method", "defect_type", "precision", "recall", "f1"],
    )
    detection_path = out_dir / "detection_by_type.csv"
    detection_df.to_csv(detection_path, index=False)
    print(f"[write] detection_by_type -> {detection_path} ({len(detection_df)} 行)")

    summary = {
        "n_outcomes": len(all_outcomes),
        "results": [_outcome_summary_row(o) for o in all_outcomes],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[write] summary -> {summary_path}")

    return {
        "results": results_path,
        "detection": detection_path,
        "summary": summary_path,
    }


def make_figures(out_dir: Path) -> list[Path]:
    """从落盘的 CSV 读回数据并生成全部图表，返回图片路径列表。"""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(out_dir / "results.csv")
    detection_df = pd.read_csv(out_dir / "detection_by_type.csv")

    paths = [
        viz.plot_auc_comparison(results_df, fig_dir / "auc_comparison.png"),
        viz.plot_recovery_rate(results_df, fig_dir / "recovery_rate.png"),
        viz.plot_detection_by_type(detection_df, fig_dir / "detection_by_type.png"),
    ]
    for p in paths:
        print(f"[figure] {p}")
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="脏数据基准：端到端实验编排（生成数据 -> 跑方法 -> 指标 -> 图表）。",
    )
    parser.add_argument(
        "--config",
        default="configs/midterm.yaml",
        help="实验配置 YAML 路径（默认 configs/midterm.yaml）。",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="跳过所有 agent 变体，只跑 baseline。",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="最多跑前 N 个数据集（调试用）。",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="冒烟模式：用很小的 n_rows/n_features 快速跑通。",
    )
    return parser


def main() -> None:
    """实验编排主入口，解析 CLI 后逐数据集跑通并落盘结果与图表。"""
    args = build_parser().parse_args()

    print(f"[config] 读取 {args.config}")
    cfg = load_config(args.config)

    datasets = cfg.get("datasets", [])
    base = cfg.get("base", {})
    ablations = cfg.get("ablations", [])
    model = cfg.get("model", "deepseek-chat")
    out_dir = Path(cfg.get("out_dir", "results"))

    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]

    print(f"[config] 数据集={len(datasets)} ablations={len(ablations)} "
          f"model={model} out_dir={out_dir} "
          f"quick={args.quick} skip_agent={args.skip_agent}")

    all_outcomes: list[MethodOutcome] = []
    for spec in datasets:
        outcomes = run_dataset(
            spec=spec,
            base=base,
            ablations=ablations,
            model=model,
            skip_agent=args.skip_agent,
            quick=args.quick,
            out_dir=out_dir,
        )
        all_outcomes.extend(outcomes)

    print(f"\n[summary] 共 {len(all_outcomes)} 条 MethodOutcome，开始落盘。")
    write_results(all_outcomes, out_dir)
    make_figures(out_dir)
    print("[done] 实验编排完成。")


if __name__ == "__main__":
    main()
