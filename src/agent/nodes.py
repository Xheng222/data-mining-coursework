"""Planner / Executor / Reviewer / Reporter 节点实现。

每个节点是一个 ``(state) -> state_update`` 的纯函数（除 Reporter 落盘外）。
节点内的所有 LLM 调用都被包在 ``_invoke_llm`` 中（带一次重试 + JSON 兜底解析），
任何抖动都会被 catch 成日志，最终保证 ``run_agent`` 返回合法 CleaningResult。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent import prompts, tools
from src.agent.state import AgentState
from src.contracts import (
    F_DIRTY_TRAIN,
    TARGET_COL,
    DefectReport,
    empty_report,
    normalize_pair,
)

# --------------------------------------------------------------------------- #
# LLM 客户端（惰性初始化 + 健壮调用）
# --------------------------------------------------------------------------- #

_ENV_LOADED = False


def _ensure_env() -> None:
    """加载 .env：先显式读项目根，再 fallback 默认向上查找。只执行一次。"""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv("F:/doc/数据挖掘/data-mining-coursework/.env")
        load_dotenv()  # fallback：从 cwd 向上找
    except Exception:
        pass
    _ENV_LOADED = True


def make_llm(model: str):
    """构造 DeepSeek（OpenAI 兼容）聊天模型。缺 API key 时抛 RuntimeError。"""
    _ensure_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请在项目根 .env 中配置 "
            "DEEPSEEK_API_KEY（可选 DEEPSEEK_BASE_URL），再运行 Agent。"
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        timeout=60,
        max_retries=0,  # 我们自己控制重试
    )


def _invoke_llm(llm, prompt: str, log: list[str], tag: str) -> dict[str, Any] | None:
    """调用 LLM 并解析 JSON。失败重试一次；仍失败则记录日志并返回 None（交由兜底）。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=prompts.SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = llm.invoke(messages)
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
            last_err = ValueError("LLM 返回中未找到可解析的 JSON")
        except Exception as exc:  # 网络/限流/解析等一律兜底
            last_err = exc
        log.append(f"[{tag}] LLM 调用/解析失败（第 {attempt + 1} 次）：{last_err}")
    log.append(f"[{tag}] LLM 不可用，退化为基于 profile 的启发式判断。")
    return None


def _extract_json(text: str) -> dict[str, Any] | None:
    """从 LLM 文本里稳健抽取 JSON 对象：直接解析失败时尝试截取首个 {...} 块。"""
    if not text:
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # 兜底：从第一个 { 到最后一个 } 截取
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# 启发式兜底：当 LLM 不可用时，直接用 profile 推断缺陷判定
# --------------------------------------------------------------------------- #

def _heuristic_decision(profile: dict[str, Any]) -> dict[str, Any]:
    """完全基于确定性画像的缺陷判定，作为 LLM 失败时的退化路径。"""
    leakage = list(profile.get("high_corr_cols", []))
    for col, r in profile.get("target_corr", {}).items():
        if r >= 0.98 and col not in leakage:
            leakage.append(col)
    missing = list(profile.get("missing_frac", {}).keys())

    # 合并精确重复对与基于距离的近邻重复对
    dup_pairs = list(profile.get("duplicate_pairs", []))
    near_dup = list(profile.get("near_duplicate_pairs", []))
    merged_pairs = dup_pairs + [p for p in near_dup if p not in dup_pairs]

    imbalance = float(profile.get("imbalance_ratio", 1.0)) > 3.0
    fmt = list(profile.get("format_suspect_cols", []))

    # 标签噪声信号：从 CV 残差画像中读取
    noise_signal = profile.get("label_noise_signal", {})
    label_noise = bool(noise_signal.get("suspected", False))

    return {
        "leakage_cols": leakage,
        "missing_cols": missing,
        "duplicate_pairs": merged_pairs,
        "label_noise": label_noise,
        "class_imbalance": imbalance,
        "format_inconsistency_cols": fmt,
    }


def _coerce_decision(raw: dict[str, Any] | None, profile: dict[str, Any]) -> dict[str, Any]:
    """把 LLM 返回（可能字段缺失/类型不对）规整成标准决策 dict，缺啥用启发式补。"""
    base = _heuristic_decision(profile)
    if not raw:
        return base
    out = dict(base)

    def _as_list(v: Any) -> list[str] | None:
        if isinstance(v, list):
            return [str(x) for x in v]
        return None

    for key in ("leakage_cols", "missing_cols", "duplicate_pairs", "format_inconsistency_cols"):
        lst = _as_list(raw.get(key))
        if lst is not None:
            out[key] = lst
    for key in ("label_noise", "class_imbalance"):
        if isinstance(raw.get(key), bool):
            out[key] = raw[key]
    return out


# --------------------------------------------------------------------------- #
# 决策 -> 缺陷报告 + 实际修复
# --------------------------------------------------------------------------- #

def _apply_decision(
    df: pd.DataFrame, decision: dict[str, Any], log: list[str]
) -> tuple[pd.DataFrame, DefectReport]:
    """根据结构化决策，用 tools 里的确定性函数执行修复，并构造缺陷报告。

    items 粒度严格对齐 contracts：
      - leakage / missing / format_inconsistency : 列名
      - near_duplicate                            : "i-j"
      - label_noise / class_imbalance             : ["present"]
    """
    report = empty_report()
    cleaned = df

    cols = set(map(str, df.columns))

    # 格式不一致：先统一文本，再尝试把疑似数值列转回数值
    fmt_cols = [c for c in decision.get("format_inconsistency_cols", []) if c in cols and c != TARGET_COL]
    if fmt_cols:
        cleaned = tools.normalize_text(cleaned, fmt_cols)
        cleaned = tools.coerce_numeric(cleaned, fmt_cols)
        report["format_inconsistency"]["items"] = fmt_cols
        log.append(f"[executor] 统一格式列：{fmt_cols}")

    # 泄漏列：删除
    leak_cols = [c for c in decision.get("leakage_cols", []) if c in cols and c != TARGET_COL]
    if leak_cols:
        cleaned = tools.drop_columns(cleaned, leak_cols)
        report["leakage"]["items"] = leak_cols
        log.append(f"[executor] 删除疑似泄漏列：{leak_cols}")

    # 缺失：填补（报告所有声明的列，修复仍存在的列）
    miss_cols = [c for c in decision.get("missing_cols", []) if c in cleaned.columns and c != TARGET_COL]
    if decision.get("missing_cols"):
        cleaned = tools.impute_missing(cleaned, miss_cols if miss_cols else None)
        report["missing"]["items"] = [c for c in decision["missing_cols"] if c != TARGET_COL]
        log.append(f"[executor] 填补缺失列：{report['missing']['items']}")

    # 近似重复：去重 + 报告样本对（对 LLM 给的对做规范化与有效性过滤）
    raw_pairs = decision.get("duplicate_pairs", [])
    valid_pairs = _normalize_pairs(raw_pairs)
    if valid_pairs:
        report["near_duplicate"]["items"] = valid_pairs
    if valid_pairs or decision.get("duplicate_pairs"):
        before = len(cleaned)
        cleaned = tools.drop_duplicates(cleaned)
        log.append(f"[executor] 去重：{before} -> {len(cleaned)} 行，报告 {len(valid_pairs)} 对。")

    # 标签噪声 / 类别不平衡：只报告 present，不做破坏性修改
    if decision.get("label_noise"):
        report["label_noise"]["items"] = ["present"]
        log.append("[executor] 报告：检测到标签噪声。")
    if decision.get("class_imbalance"):
        report["class_imbalance"]["items"] = ["present"]
        log.append("[executor] 报告：检测到类别不平衡。")

    return cleaned, report


def _normalize_pairs(raw_pairs: Any) -> list[str]:
    """把 LLM/启发式给的重复对规范化为唯一的 "i-j"（i<j）列表。"""
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(raw_pairs, list):
        return out
    for p in raw_pairs:
        try:
            s = str(p)
            if "-" not in s:
                continue
            a, b = s.split("-", 1)
            key = normalize_pair(int(a), int(b))
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


# --------------------------------------------------------------------------- #
# 节点
# --------------------------------------------------------------------------- #

def planner_node(state: AgentState) -> dict[str, Any]:
    """Planner：基于画像让 LLM 产出清洗计划；LLM 不可用则给出默认计划。"""
    log = state.get("log", [])
    profile = state["profile"]
    plan_text = "（默认流程）排查泄漏列 -> 填补缺失 -> 去重 -> 统一格式 -> 标注不平衡/标签噪声。"
    try:
        llm = make_llm(state["model"])
        prompt = prompts.planner_prompt(state["task"], profile, state["prompt_strategy"])
        parsed = _invoke_llm(llm, prompt, log, "planner")
        if parsed and isinstance(parsed.get("plan"), list) and parsed["plan"]:
            plan_text = "; ".join(str(s) for s in parsed["plan"])
            log.append(f"[planner] 生成 {len(parsed['plan'])} 步清洗计划。")
    except RuntimeError:
        raise
    except Exception as exc:
        log.append(f"[planner] 异常，使用默认计划：{exc}")
    return {"plan": plan_text, "log": log}


def executor_node(state: AgentState) -> dict[str, Any]:
    """Executor：LLM 产出结构化缺陷判定 -> 确定性工具执行修复 -> 填充 reported_defects。

    每轮都从原始 dirty_df 重新执行修复，保证幂等、避免在已清洗数据上累积副作用。
    """
    log = state.get("log", [])
    profile = state["profile"]
    plan = state.get("plan") if state.get("use_planner") else None
    feedback = state.get("review_feedback")

    raw_decision: dict[str, Any] | None = None
    try:
        llm = make_llm(state["model"])
        prompt = prompts.executor_prompt(
            state["task"], profile, plan, state["prompt_strategy"], feedback
        )
        raw_decision = _invoke_llm(llm, prompt, log, "executor")
    except RuntimeError:
        raise
    except Exception as exc:
        log.append(f"[executor] LLM 异常，启用启发式：{exc}")

    decision = _coerce_decision(raw_decision, profile)
    cleaned, report = _apply_decision(state["dirty_df"], decision, log)
    return {"cleaned_df": cleaned, "reported_defects": report, "log": log}


def reviewer_node(state: AgentState) -> dict[str, Any]:
    """Reviewer：审视执行器报告，决定是否需要回到 Executor 再修一轮。"""
    log = state.get("log", [])
    round_no = state.get("review_round", 0) + 1
    needs = False
    feedback = ""
    try:
        llm = make_llm(state["model"])
        prompt = prompts.reviewer_prompt(
            state["profile"], state["reported_defects"], state["prompt_strategy"]
        )
        parsed = _invoke_llm(llm, prompt, log, "reviewer")
        if parsed is not None:
            needs = bool(parsed.get("needs_revision", False))
            feedback = str(parsed.get("feedback", ""))
            log.append(f"[reviewer] 第 {round_no} 轮：needs_revision={needs}。")
        else:
            log.append(f"[reviewer] 第 {round_no} 轮：LLM 不可用，默认不再修订。")
    except RuntimeError:
        raise
    except Exception as exc:
        log.append(f"[reviewer] 异常，默认不再修订：{exc}")

    # 触顶强制停止
    if round_no >= state.get("max_review_rounds", 2):
        needs = False
        log.append("[reviewer] 已达最大审视轮数，停止修订。")
    return {"needs_revision": needs, "review_feedback": feedback, "review_round": round_no}


def reporter_node(state: AgentState) -> dict[str, Any]:
    """Reporter：cleaned_df 落盘到 data/processed/<dataset_id>_agent.csv，记录最终日志。"""
    log = state.get("log", [])
    cleaned = state.get("cleaned_df")
    if cleaned is None:
        cleaned = state["dirty_df"]

    out_dir = Path(state["dataset_root"]).resolve()
    # data/processed 相对项目根；从 dataset_root 向上找到含 data 的根
    processed_dir = _processed_dir(out_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{state['dataset_id']}_agent.csv"
    cleaned.to_csv(out_path, index=False)
    log.append(f"[reporter] 清洗后数据落盘：{out_path}（{len(cleaned)} 行 x {cleaned.shape[1]} 列）。")
    return {"cleaned_train_path": str(out_path), "log": log}


def _processed_dir(dataset_root: Path) -> Path:
    """根据 dataset_root 推断 data/processed 目录。

    约定数据集位于 <repo>/data/<tier>/<dataset_id>，故向上找到 data 目录的父级，
    在其下 data/processed 落盘；找不到则退化为 dataset_root 同级的 processed。
    """
    for parent in dataset_root.parents:
        if parent.name == "data":
            return parent / "processed"
    return dataset_root.parent / "processed"
