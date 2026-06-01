"""Agent 状态机装配与运行入口。feature/agent 负责实现。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.contracts import CleaningResult


@dataclass
class AgentConfig:
    """消融开关。"""

    use_planner: bool = True       # 有/无 Planner 节点
    use_reviewer: bool = True      # 有/无 Reviewer 节点（reflection）
    prompt_strategy: str = "cot"   # "zero_shot" | "few_shot" | "cot"
    model: str = "deepseek-chat"
    max_review_rounds: int = 2


def run_agent(
    dataset_root: str | Path,
    task: str = "请清洗该数据集并训练一个二分类模型",
    config: AgentConfig | None = None,
) -> CleaningResult:
    """对脏训练集运行 Data Agent，返回清洗结果与报告的缺陷。"""
    raise NotImplementedError
