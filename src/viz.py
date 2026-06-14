"""结果可视化。

把 ``run_experiments`` 汇总出的 DataFrame 渲染成 PNG 图表：

- :func:`plot_auc_comparison`     —— 各方法在各数据集上的 AUC 分组柱状图。
- :func:`plot_recovery_rate`      —— rule_based 与各 agent 变体的 Recovery Rate。
- :func:`plot_detection_by_type`  —— 分缺陷类型的检测召回率对比。

为避免中文字体缺失导致乱码，所有图内文本统一使用英文标签。使用 Agg 后端，
不弹窗、只落盘。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 参考线方法（上界 / 下界），在 AUC 图里单独标注。
_REFERENCE_METHODS = ("clean_upper", "no_clean")


def _save(fig: "plt.Figure", out_path: str | Path) -> Path:
    """统一落盘逻辑：建目录、写 PNG、关闭 figure，返回路径。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _filter_model(df: pd.DataFrame, model: str | None = "xgb") -> pd.DataFrame:
    """若 DataFrame 含 ``model`` 列，则按指定模型过滤；否则原样返回。"""
    if model is not None and "model" in df.columns:
        return df[df["model"] == model].copy()
    return df.copy()


def plot_auc_comparison(results_df: pd.DataFrame, out_path: str | Path,
                        model: str | None = "xgb") -> Path:
    """各方法在各数据集上的 AUC 分组柱状图。

    x 轴为数据集，每个数据集内并排画出全部方法的 AUC；clean_upper / no_clean
    作为上下界一并展示。NaN（如 agent 调用失败）的条目自动跳过。

    Parameters
    ----------
    results_df:
        含列 ``dataset_id``、``method``、``auc`` 的汇总表。若有 ``model`` 列则自动过滤。
    out_path:
        输出 PNG 路径。
    model:
        若 DataFrame 含 ``model`` 列，只绘制该模型的结果。设为 ``None`` 显示全部模型。
    Returns
    -------
    pathlib.Path
    """
    df = _filter_model(results_df, model).dropna(subset=["auc"]).copy()
    datasets = sorted(df["dataset_id"].unique())

    # 方法排序：上下界放两端，其余按字母序，便于视觉对比。
    methods = list(df["method"].unique())
    middle = sorted(m for m in methods if m not in _REFERENCE_METHODS)
    ordered: list[str] = []
    if "no_clean" in methods:
        ordered.append("no_clean")
    ordered.extend(middle)
    if "clean_upper" in methods:
        ordered.append("clean_upper")

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(datasets)), 5))
    n_methods = max(len(ordered), 1)
    total_width = 0.8
    bar_width = total_width / n_methods
    x = np.arange(len(datasets))

    for i, method in enumerate(ordered):
        sub = df[df["method"] == method].set_index("dataset_id")["auc"]
        heights = [sub.get(d, np.nan) for d in datasets]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(x + offset, heights, bar_width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("AUC-ROC (frozen clean test set)")
    ax.set_title("AUC Comparison Across Methods")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Method", loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    return _save(fig, out_path)


def plot_recovery_rate(results_df: pd.DataFrame, out_path: str | Path,
                       model: str | None = "xgb") -> Path:
    """rule_based 与各 agent 变体的 Recovery Rate 分组柱状图。

    Recovery Rate 把方法 AUC 在 [no_clean, clean_upper] 区间内归一化，1.0 表示
    完全恢复到干净数据水平。上下界方法本身（recovery 恒为 0/1）不参与绘制。
    """
    df = _filter_model(results_df, model).dropna(subset=["recovery_rate"]).copy()
    df = df[~df["method"].isin(_REFERENCE_METHODS)]

    datasets = sorted(df["dataset_id"].unique())
    methods = sorted(df["method"].unique())

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(datasets)), 5))
    n_methods = max(len(methods), 1)
    total_width = 0.8
    bar_width = total_width / n_methods
    x = np.arange(len(datasets))

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("dataset_id")["recovery_rate"]
        heights = [sub.get(d, np.nan) for d in datasets]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(x + offset, heights, bar_width, label=method)

    ax.axhline(1.0, color="green", linestyle="--", alpha=0.6, label="full recovery")
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.6, label="no recovery")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Recovery Rate by Cleaning Method")
    ax.legend(title="Method", loc="best", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    return _save(fig, out_path)


def plot_model_comparison(results_df: pd.DataFrame, out_path: str | Path) -> Path:
    """各模型在各（数据集, 方法）上的 AUC 散点对比图。

    若数据不包含多模型，直接返回占位图。
    """
    df = results_df.dropna(subset=["auc"]).copy()
    if "model" not in df.columns or df["model"].nunique() < 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Single model only – no comparison needed", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_path)

    models = sorted(df["model"].unique())
    # 对每个 (dataset, method) 按模型展开 AUC
    pivot = df.pivot_table(index=["dataset_id", "method"], columns="model", values="auc")
    pivot = pivot.dropna()

    fig, axes = plt.subplots(1, len(models) - 1, figsize=(6 * (len(models) - 1), 5),
                             sharex=False, sharey=True)
    if len(models) - 1 == 1:
        axes = [axes]

    ref_model = models[0]
    for ax, cmp_model in zip(axes, models[1:]):
        ax.scatter(pivot[ref_model], pivot[cmp_model], alpha=0.6)
        lims = [min(pivot[[ref_model, cmp_model]].min().min(), 0.4),
                max(pivot[[ref_model, cmp_model]].max().max(), 1.0)]
        ax.plot(lims, lims, "r--", alpha=0.4, label="y=x")
        ax.set_xlabel(f"AUC ({ref_model})")
        ax.set_ylabel(f"AUC ({cmp_model})")
        ax.set_title(f"{ref_model} vs {cmp_model}")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Model Robustness: AUC Comparison Across Classifiers")
    return _save(fig, out_path)


def plot_detection_by_type(detection_df: pd.DataFrame, out_path: str | Path) -> Path:
    """分缺陷类型的检测召回率对比（rule_based vs agent 变体）。

    x 轴为缺陷类型，每个类型内并排画出各检测方法在所有数据集上的平均召回率。

    Parameters
    ----------
    detection_df:
        含列 ``dataset_id``、``method``、``defect_type``、``recall`` 的明细表。
    out_path:
        输出 PNG 路径。
    """
    df = detection_df.dropna(subset=["recall"]).copy()

    if df.empty:
        # 没有任何检测结果（例如全程 --skip-agent 且基线未报告）时给一张占位图，
        # 避免编排流程因空数据崩溃。
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No detection results available", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title("Detection Recall by Defect Type")
        return _save(fig, out_path)

    # 跨数据集对同一 (method, defect_type) 取平均召回率。
    agg = (
        df.groupby(["defect_type", "method"])["recall"].mean().reset_index()
    )
    defect_types = sorted(agg["defect_type"].unique())
    methods = sorted(agg["method"].unique())

    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(defect_types)), 5))
    n_methods = max(len(methods), 1)
    total_width = 0.8
    bar_width = total_width / n_methods
    x = np.arange(len(defect_types))

    pivot = agg.pivot(index="defect_type", columns="method", values="recall")
    for i, method in enumerate(methods):
        heights = [pivot.loc[t, method] if t in pivot.index else np.nan for t in defect_types]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(x + offset, heights, bar_width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(defect_types, rotation=30, ha="right")
    ax.set_xlabel("Defect Type")
    ax.set_ylabel("Detection Recall (avg over datasets)")
    ax.set_title("Detection Recall by Defect Type")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Method", loc="best", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    return _save(fig, out_path)
