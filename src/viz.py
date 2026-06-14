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

# ── 专业配色（Tailwind-inspired palette） ────────────────
# Agent 变体系列：蓝→青渐变，表示同源方法
# rule_based：琥珀色，与 Agent 区分
# no_clean / clean_upper：红/绿，直观表示下界/上界
_COLORS = {
    "agent_full":         "#2563EB",   # blue-600
    "agent_no_reviewer":  "#3B82F6",   # blue-500
    "agent_no_planner":   "#0EA5E9",   # sky-500
    "agent_zero_shot":    "#06B6D4",   # cyan-500
    "agent_few_shot":     "#14B8A6",   # teal-500
    "rule_based":         "#F59E0B",   # amber-500
    "no_clean":           "#EF4444",   # red-500
    "clean_upper":        "#10B981",   # emerald-500
}
_DEFAULT_COLOR = "#6B7280"


def _method_color(method: str) -> str:
    return _COLORS.get(method, _DEFAULT_COLOR)


def _save(fig: "plt.Figure", out_path: str | Path) -> Path:
    """统一落盘逻辑：建目录、写 PNG、关闭 figure，返回路径。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _filter_model(df: pd.DataFrame, model: str | None = "xgb") -> pd.DataFrame:
    """若 DataFrame 含 ``model`` 列，则按指定模型过滤；否则原样返回。"""
    if model is not None and "model" in df.columns:
        return df[df["model"] == model].copy()
    return df.copy()


def plot_auc_comparison(results_df: pd.DataFrame, out_path: str | Path,
                        model: str | None = "xgb") -> Path:
    """各方法在各数据集上的 AUC 分组柱状图。"""
    df = _filter_model(results_df, model).dropna(subset=["auc"]).copy()
    datasets = sorted(df["dataset_id"].unique())

    methods = list(df["method"].unique())
    middle = sorted(m for m in methods if m not in _REFERENCE_METHODS)
    ordered: list[str] = []
    if "no_clean" in methods:
        ordered.append("no_clean")
    ordered.extend(middle)
    if "clean_upper" in methods:
        ordered.append("clean_upper")

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(datasets)), 5.5))
    n_methods = max(len(ordered), 1)
    total_width = 0.8
    bar_width = total_width / n_methods
    x = np.arange(len(datasets))

    for i, method in enumerate(ordered):
        sub = df[df["method"] == method].set_index("dataset_id")["auc"]
        heights = [sub.get(d, np.nan) for d in datasets]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(x + offset, heights, bar_width, label=method, color=_method_color(method))

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right", fontsize=9)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("AUC-ROC (frozen clean test set)", fontsize=10)
    ax.set_title("AUC Comparison Across Methods", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Method", loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, out_path)


def plot_recovery_rate(results_df: pd.DataFrame, out_path: str | Path,
                       model: str | None = "xgb") -> Path:
    """rule_based 与各 agent 变体的 Recovery Rate 分组柱状图。"""
    df = _filter_model(results_df, model).dropna(subset=["recovery_rate"]).copy()
    df = df[~df["method"].isin(_REFERENCE_METHODS)]

    datasets = sorted(df["dataset_id"].unique())
    methods = sorted(df["method"].unique())

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(datasets)), 5.5))
    n_methods = max(len(methods), 1)
    total_width = 0.8
    bar_width = total_width / n_methods
    x = np.arange(len(datasets))

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("dataset_id")["recovery_rate"]
        heights = [sub.get(d, np.nan) for d in datasets]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(x + offset, heights, bar_width, label=method, color=_method_color(method))

    ax.axhline(1.0, color="#10B981", linestyle="--", alpha=0.7, linewidth=1.5, label="full recovery (RR=1)")
    ax.axhline(0.0, color="#EF4444", linestyle="--", alpha=0.7, linewidth=1.5, label="no recovery (RR=0)")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right", fontsize=9)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("Recovery Rate", fontsize=10)
    ax.set_title("Recovery Rate by Cleaning Method", fontsize=12, fontweight="bold")
    ax.legend(title="Method", loc="best", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, out_path)


def plot_model_comparison(results_df: pd.DataFrame, out_path: str | Path) -> Path:
    """各模型在各（数据集, 方法）上的 AUC 散点对比图。"""
    df = results_df.dropna(subset=["auc"]).copy()
    if "model" not in df.columns or df["model"].nunique() < 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Single model only – no comparison needed",
                ha="center", va="center", fontsize=12, color=_DEFAULT_COLOR)
        ax.set_axis_off()
        return _save(fig, out_path)

    models = sorted(df["model"].unique())
    pivot = df.pivot_table(index=["dataset_id", "method"], columns="model", values="auc")
    pivot = pivot.dropna()

    fig, axes = plt.subplots(1, len(models) - 1, figsize=(6 * (len(models) - 1), 5),
                             sharex=False, sharey=True)
    if len(models) - 1 == 1:
        axes = [axes]

    ref_model = models[0]
    model_cmap = {"xgb": "#2563EB", "rf": "#F59E0B", "lr": "#10B981"}
    for ax, cmp_model in zip(axes, models[1:]):
        colors = [model_cmap.get(m, _DEFAULT_COLOR) for m in pivot.index.get_level_values("method")]
        ax.scatter(pivot[ref_model], pivot[cmp_model], c=colors, alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
        lims = [min(pivot[[ref_model, cmp_model]].min().min(), 0.4),
                max(pivot[[ref_model, cmp_model]].max().max(), 1.0)]
        ax.plot(lims, lims, "--", color="#EF4444", alpha=0.4, label="y=x")
        ax.set_xlabel(f"AUC ({ref_model})", fontsize=10)
        ax.set_ylabel(f"AUC ({cmp_model})", fontsize=10)
        ax.set_title(f"{ref_model} vs {cmp_model}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Model Robustness: AUC Comparison Across Classifiers",
                 fontsize=12, fontweight="bold")
    return _save(fig, out_path)


def plot_detection_by_type(detection_df: pd.DataFrame, out_path: str | Path) -> Path:
    """分缺陷类型的检测召回率对比（rule_based vs agent 变体）。"""
    df = detection_df.dropna(subset=["recall"]).copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No detection results available",
                ha="center", va="center", fontsize=12, color=_DEFAULT_COLOR)
        ax.set_axis_off()
        ax.set_title("Detection Recall by Defect Type")
        return _save(fig, out_path)

    agg = df.groupby(["defect_type", "method"])["recall"].mean().reset_index()
    defect_types = sorted(agg["defect_type"].unique())
    methods = sorted(agg["method"].unique())

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(defect_types)), 5.5))
    n_methods = max(len(methods), 1)
    total_width = 0.8
    bar_width = total_width / n_methods
    x = np.arange(len(defect_types))

    pivot = agg.pivot(index="defect_type", columns="method", values="recall")
    for i, method in enumerate(methods):
        heights = [pivot.loc[t, method] if t in pivot.index else np.nan for t in defect_types]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(x + offset, heights, bar_width, label=method, color=_method_color(method))

    ax.set_xticks(x)
    ax.set_xticklabels(defect_types, rotation=25, ha="right", fontsize=9)
    ax.set_xlabel("Defect Type", fontsize=10)
    ax.set_ylabel("Detection Recall (avg over datasets)", fontsize=10)
    ax.set_title("Detection Recall by Defect Type", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Method", loc="best", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    return _save(fig, out_path)
