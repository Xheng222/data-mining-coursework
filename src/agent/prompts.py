"""各节点的提示词模板，支持 zero_shot / few_shot / cot 三种策略。

设计要点
--------
- 所有「向 LLM 要结论」的提示词都要求返回**严格 JSON**，便于稳健解析；解析失败时
  上层节点会退化为基于 profile 的启发式判断，不依赖 LLM 也能跑通。
- ``prompt_strategy`` 通过在系统提示尾部追加一段「策略指令」实现切换：
    * zero_shot：直接给结论，不示例、不解释。
    * few_shot ：先给 1-2 个输入->JSON 的示例，再让模型照做。
    * cot      ：要求模型在 "reasoning" 字段里逐步推理，再在结构化字段给结论。
- 提示词只描述「依据 profile 判断」，**绝不引导模型去读 ground_truth / clean_test**。
"""

from __future__ import annotations

import json
from typing import Any

from src.contracts import DEFECT_TYPES

STRATEGIES = ("zero_shot", "few_shot", "cot")


def _strategy_suffix(prompt_strategy: str, few_shot_example: str | None = None) -> str:
    """根据策略生成追加到提示尾部的指令片段。"""
    if prompt_strategy == "zero_shot":
        return "\n\n【输出要求】直接输出最终 JSON，不要任何解释或额外文字。"
    if prompt_strategy == "few_shot":
        ex = few_shot_example or ""
        return (
            "\n\n【示例】请参照以下输入->输出的对应关系作答：\n"
            f"{ex}\n\n【输出要求】只输出与示例同结构的 JSON，不要解释。"
        )
    if prompt_strategy == "cot":
        return (
            "\n\n【输出要求】请在 JSON 的 \"reasoning\" 字段中分步写出你的推理"
            "（数据画像里哪些信号支持你的判断），随后在其余字段给出最终结论。"
            "最终只输出一个 JSON 对象。"
        )
    # 未知策略退化为 zero_shot
    return "\n\n【输出要求】直接输出最终 JSON，不要任何解释。"


def _profile_brief(profile: dict[str, Any]) -> str:
    """把 profile 压成紧凑 JSON 串塞进提示，避免提示过长。"""
    brief = {
        "n_rows": profile.get("n_rows"),
        "n_cols": profile.get("n_cols"),
        "columns": profile.get("columns"),
        "dtypes": profile.get("dtypes"),
        "missing_frac": profile.get("missing_frac"),
        "n_duplicate_rows": profile.get("n_duplicate_rows"),
        "n_duplicate_pairs": len(profile.get("duplicate_pairs", [])),
        "target_corr": profile.get("target_corr"),
        "high_corr_cols": profile.get("high_corr_cols"),
        "class_counts": profile.get("class_counts"),
        "imbalance_ratio": profile.get("imbalance_ratio"),
        "object_cols": profile.get("object_cols"),
        "format_suspect_cols": profile.get("format_suspect_cols"),
    }
    return json.dumps(brief, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #

def planner_prompt(task: str, profile: dict[str, Any], prompt_strategy: str) -> str:
    """生成 Planner 的提示：要求 LLM 基于画像产出一份分步清洗计划。"""
    base = (
        "你是一名数据清洗规划专家。给定数据集的统计画像和任务，"
        "请制定一份**分步**的数据清洗计划，覆盖：疑似数据泄漏列的排查、缺失值处理、"
        "重复样本去除、标签噪声、类别不平衡、列格式不一致的统一。\n\n"
        f"任务：{task}\n"
        f"数据画像（JSON）：{_profile_brief(profile)}\n\n"
        "请输出 JSON：{\"plan\": [\"步骤1\", \"步骤2\", ...]}"
    )
    return base + _strategy_suffix(
        prompt_strategy,
        few_shot_example='输入: {"high_corr_cols":["leak_a"],"missing_frac":{"f3":0.2}}\n'
        '输出: {"plan":["删除疑似泄漏列 leak_a","用中位数填补 f3 缺失","按特征去重"]}',
    )


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

_EXECUTOR_SCHEMA = (
    '{'
    '"leakage_cols": [列名...], '
    '"missing_cols": [列名...], '
    '"duplicate_pairs": ["i-j"...], '
    '"label_noise": true/false, '
    '"class_imbalance": true/false, '
    '"format_inconsistency_cols": [列名...]'
    '}'
)


def executor_prompt(
    task: str,
    profile: dict[str, Any],
    plan: str | None,
    prompt_strategy: str,
    review_feedback: str | None = None,
) -> str:
    """生成 Executor 的判断提示：要求 LLM 基于画像给出结构化缺陷判定。

    注意：只允许依据传入的统计画像判断，严禁假设可以访问测试集或 ground truth。
    """
    plan_part = f"\n参考清洗计划：{plan}" if plan else ""
    feedback_part = (
        f"\n上一轮审阅反馈（请据此修正判断）：{review_feedback}" if review_feedback else ""
    )
    base = (
        "你是一名数据缺陷诊断专家。只依据下面给出的**统计画像**判断脏训练集中存在哪些缺陷"
        "（不得假设你能看到测试集或标准答案）。需要判断的缺陷类型固定为："
        f"{list(DEFECT_TYPES)}。\n\n"
        "判定要点：\n"
        "- leakage（数据泄漏）：与 target 近乎完全相关（|r|≈1）或语义上由标签派生的列。\n"
        "- missing（缺失）：missing_frac 中出现的列。\n"
        "- near_duplicate（近似重复）：画像里的 duplicate_pairs。\n"
        "- label_noise（标签噪声）：标签可能被翻转/污染（画像难直接看出时凭经验判断）。\n"
        "- class_imbalance（类别不平衡）：imbalance_ratio 明显偏大（如 >3）。\n"
        "- format_inconsistency（格式不一致）：format_suspect_cols 中的列。\n\n"
        f"任务：{task}{plan_part}{feedback_part}\n"
        f"数据画像（JSON）：{_profile_brief(profile)}\n\n"
        f"请输出 JSON（仅含这些键）：{_EXECUTOR_SCHEMA}"
    )
    return base + _strategy_suffix(
        prompt_strategy,
        few_shot_example=(
            '输入: {"high_corr_cols":["leak"],"target_corr":{"leak":0.99,"f1":0.3},'
            '"missing_frac":{"f2":0.15},"imbalance_ratio":5.0,"format_suspect_cols":["city"],'
            '"n_duplicate_pairs":2}\n'
            '输出: {"leakage_cols":["leak"],"missing_cols":["f2"],"duplicate_pairs":[],'
            '"label_noise":false,"class_imbalance":true,"format_inconsistency_cols":["city"]}'
        ),
    )


# --------------------------------------------------------------------------- #
# Reviewer
# --------------------------------------------------------------------------- #

def reviewer_prompt(
    profile: dict[str, Any],
    reported: dict[str, Any],
    prompt_strategy: str,
) -> str:
    """生成 Reviewer 的提示：审视 Executor 的判断是否充分/有误，决定是否需要再修一轮。"""
    base = (
        "你是一名严格的数据清洗审阅员。下面是数据画像与执行器报告的缺陷判定。"
        "请检查是否存在**漏报**（画像里明显的缺陷未被报告）或**误报**，"
        "并判断是否需要让执行器再修订一轮。\n\n"
        f"数据画像（JSON）：{_profile_brief(profile)}\n"
        f"执行器的缺陷判定（JSON）：{json.dumps(reported, ensure_ascii=False)}\n\n"
        "请输出 JSON：{\"needs_revision\": true/false, \"feedback\": \"给执行器的具体修订建议\"}"
    )
    return base + _strategy_suffix(
        prompt_strategy,
        few_shot_example=(
            '输入画像有 missing_frac={"f2":0.2} 但报告 missing_cols=[]\n'
            '输出: {"needs_revision":true,"feedback":"f2 有 20% 缺失却未报告，请补上"}'
        ),
    )


SYSTEM_PROMPT = (
    "你是数据清洗基准中的 Data Agent 组件。严格按用户要求返回 JSON，"
    "只依据提供的数据画像作答，不要编造无法从画像中得到的信息。"
)
