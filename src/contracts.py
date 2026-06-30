"""共享契约：所有模块都依赖本文件中的类型、常量与 schema。

设计目的：让 datagen / preprocess / baseline / evaluate / agent 可以在各自的
feature 分支上并行开发而不互相阻塞。任何跨模块的数据格式都在此处定义，
开发期间本文件视为只读（不要在功能分支里修改它）。

核心数据流
----------
1. datagen 生成一个 ``DatasetBundle``，落盘 4 个文件 + 1 个 ground-truth JSON。
2. baseline / agent 读取脏训练集，产出 (清洗后训练集, ReportedDefects)。
3. evaluate 用固定的干净测试集计算 Detection(P/R) 与 Recovery Rate。

缺陷与检测的统一表示
--------------------
ground truth 和「被报告的缺陷」都用同一个 schema，便于统一打分：

    {
        "<defect_type>": {
            "items": [<hashable>, ...],   # 该缺陷的可识别单元（见下）
            "meta":  {...}                # 附加信息，不参与打分
        },
        ...
    }

各缺陷类型的 ``items`` 粒度（datagen / baseline / agent 必须一致地产出）：

    - leakage              : 泄漏列的列名 (str)
    - missing              : 含缺失值的列名 (str)
    - label_noise          : ["present"]（是否检测到标签噪声；如能给出样本下标则用 "row:{i}"）
    - near_duplicate       : 重复样本对，规范化为 "i-j"（i<j）
    - class_imbalance      : ["present"]
    - format_inconsistency : 格式不一致的列名 (str)

Detection 打分按缺陷类型分别计算集合交并，再做 micro 平均。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

TARGET_COL = "target"

DEFECT_TYPES = (
    "leakage",
    "missing",
    "label_noise",
    "near_duplicate",
    "class_imbalance",
    "format_inconsistency",
)

DIFFICULTIES = ("easy", "medium", "hard", "real")

# CleanML 真实数据集（脏/净配对表格分类任务）
CLEANML_DATASETS = ("Credit", "EEG", "Marketing")

# DatasetBundle 落盘文件名（固定，跨模块约定）
F_DIRTY_TRAIN = "dirty_train.csv"   # 注入缺陷后的训练集（X + target）
F_CLEAN_TRAIN = "clean_train.csv"   # 同样划分但未注入缺陷的训练集（性能上界用）
F_CLEAN_TEST = "clean_test.csv"     # 全程冻结的干净测试集（X + target）
F_GROUND_TRUTH = "ground_truth.json"
F_META = "meta.json"


# --------------------------------------------------------------------------- #
# 数据类
# --------------------------------------------------------------------------- #

@dataclass
class DatasetBundle:
    """一个合成数据集的全部产物（路径形式）。"""

    dataset_id: str
    difficulty: str
    root: Path
    n_rows: int
    n_features: int

    @property
    def dirty_train(self) -> Path:
        return self.root / F_DIRTY_TRAIN

    @property
    def clean_train(self) -> Path:
        return self.root / F_CLEAN_TRAIN

    @property
    def clean_test(self) -> Path:
        return self.root / F_CLEAN_TEST

    @property
    def ground_truth_path(self) -> Path:
        return self.root / F_GROUND_TRUTH


# 「缺陷报告」与「ground truth」共用的类型别名
DefectReport = dict[str, dict[str, Any]]


@dataclass
class CleaningResult:
    """一次清洗（规则基线或 Agent）的产物。"""

    cleaned_train_path: Path           # 清洗后训练集落盘路径（X + target）
    reported_defects: DefectReport     # 该方法报告/修复的缺陷，schema 见模块 docstring
    log: list[str] = field(default_factory=list)   # 处理过程的人类可读步骤
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionScore:
    """Detection 指标：整体 + 分缺陷类型。"""

    precision: float
    recall: float
    f1: float
    per_type: dict[str, dict[str, float]]   # type -> {"precision","recall","f1","tp","fp","fn"}


@dataclass
class MethodOutcome:
    """单个方法在单个数据集上的完整评测结果（写入 results）。"""

    dataset_id: str
    method: str                         # "no_clean" | "rule_based" | "agent[...]" | "clean_upper"
    auc: float                          # 在固定干净测试集上的 AUC-ROC
    model: str = "xgb"                  # 下游分类模型名称：xgb / rf / lr
    detection: Optional[DetectionScore] = None
    recovery_rate: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 公式：Recovery Rate
# --------------------------------------------------------------------------- #

def recovery_rate(auc_dirty: float, auc_method: float, auc_clean: float) -> float:
    """Recovery Rate = (AUC_method - AUC_dirty) / (AUC_clean - AUC_dirty)。

    三个 AUC 均来自同一个固定干净测试集。

    返回 0.0 表示该数据集不构成有效评估区间：
      - 分母过小（``|AUC_clean - AUC_dirty| < 1e-9``）：脏/净几乎无差异，无可恢复空间；
      - 分母为负（``AUC_clean <= AUC_dirty``）：清洗上界并不优于脏数据，
        此时 RR 的符号会被反转而失去意义，应视为无效而非给出误导性数值。
    """
    denom = auc_clean - auc_dirty
    if denom < 1e-9:
        return 0.0
    return (auc_method - auc_dirty) / denom


def normalize_pair(i: int, j: int) -> str:
    """near_duplicate 样本对的规范化字符串表示，保证 i<j。"""
    a, b = (i, j) if i < j else (j, i)
    return f"{a}-{b}"


def empty_report() -> DefectReport:
    """返回一个空的、结构完整的缺陷报告，便于增量填充。"""
    return {t: {"items": [], "meta": {}} for t in DEFECT_TYPES}
