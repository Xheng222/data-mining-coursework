"""Agent 图状态定义（TypedDict）。

LangGraph 在节点间传递的共享状态。每个节点读取部分字段、写回部分字段，
LangGraph 用浅合并（节点返回的 dict 覆盖同名键）来更新状态。
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from src.contracts import DefectReport


class AgentState(TypedDict, total=False):
    """状态机在节点间流转的全部字段（total=False：允许逐步填充）。"""

    # ---- 输入（由 run_agent 初始化） ----
    task: str                       # 自然语言任务描述
    dataset_root: str               # 数据集根目录
    dataset_id: str                 # 数据集标识（dataset_root 的目录名）

    # ---- 数据 ----
    dirty_df: pd.DataFrame          # 读入的脏训练集（只读，不被原地修改）
    profile: dict[str, Any]         # tools.profile_data 产出的统计画像
    cleaned_df: pd.DataFrame        # 当前清洗后的训练集

    # ---- Planner ----
    plan: str                       # LLM 产出的清洗计划（自由文本）

    # ---- Executor ----
    reported_defects: DefectReport  # 报告/修复的缺陷（schema 同 contracts）

    # ---- Reviewer ----
    review_feedback: str            # Reviewer 给出的修订建议
    needs_revision: bool            # 是否需要回到 Executor 再修一轮
    review_round: int               # 已进行的审视轮数

    # ---- 配置镜像（节点无法直接拿到 config，放进状态） ----
    use_planner: bool
    use_reviewer: bool
    prompt_strategy: str
    model: str
    max_review_rounds: int

    # ---- Reporter 产出 ----
    cleaned_train_path: str         # 清洗后训练集落盘路径

    # ---- 过程日志 ----
    log: list[str]
