"""Agent 状态机装配与运行入口。

状态机（LangGraph ``StateGraph``）：

    START -> [planner?] -> executor -> [reviewer?] -> reporter -> END
                              ^             |
                              └──(需修订且未触顶)──┘

消融开关（``AgentConfig``）如何生效：
  - ``use_planner=False``：图里不接入 planner 节点，executor 直接按通用流程跑。
  - ``use_reviewer=False``：图里不接入 reviewer 节点，executor 后直达 reporter（无反思）。
  - ``prompt_strategy``：透传进状态，由各节点的 prompts.* 在三种策略间切换。
  - ``max_review_rounds``：reviewer 的回边在触顶后强制走向 reporter。

稳健性：所有 LLM 不确定性都在 nodes 内被 catch 成日志并退化为启发式；
只有「连 API key 都没有」才会抛 RuntimeError（见 nodes.make_llm）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.agent import nodes, tools
from src.agent.state import AgentState
from src.contracts import F_DIRTY_TRAIN, CleaningResult, empty_report


@dataclass
class AgentConfig:
    """消融开关。"""

    use_planner: bool = True       # 有/无 Planner 节点
    use_reviewer: bool = True      # 有/无 Reviewer 节点（reflection）
    prompt_strategy: str = "cot"   # "zero_shot" | "few_shot" | "cot"
    model: str = "deepseek-chat"
    max_review_rounds: int = 2


def _build_graph(config: AgentConfig):
    """按消融开关装配 StateGraph 并编译。"""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(AgentState)
    g.add_node("executor", nodes.executor_node)
    g.add_node("reporter", nodes.reporter_node)

    # 入口：有无 planner
    if config.use_planner:
        g.add_node("planner", nodes.planner_node)
        g.add_edge(START, "planner")
        g.add_edge("planner", "executor")
    else:
        g.add_edge(START, "executor")

    # executor 之后：有无 reviewer（反思回路）
    if config.use_reviewer:
        g.add_node("reviewer", nodes.reviewer_node)
        g.add_edge("executor", "reviewer")

        def _route(state: AgentState) -> str:
            return "executor" if state.get("needs_revision") else "reporter"

        g.add_conditional_edges(
            "reviewer", _route, {"executor": "executor", "reporter": "reporter"}
        )
    else:
        g.add_edge("executor", "reporter")

    g.add_edge("reporter", END)
    return g.compile()


def run_agent(
    dataset_root: str | Path,
    task: str = "请清洗该数据集并训练一个二分类模型",
    config: AgentConfig | None = None,
) -> CleaningResult:
    """对脏训练集运行 Data Agent，返回清洗结果与报告的缺陷。

    只读 ``dataset_root/dirty_train.csv``；绝不读取 ground_truth.json 或 clean_test.csv
    （评测保留集）。``config`` 为 None 时使用默认 ``AgentConfig``。

    若环境缺少 DEEPSEEK_API_KEY，将抛出 RuntimeError；其余所有 LLM 抖动都会被
    降级为启发式处理，最终始终返回一个合法的 ``CleaningResult``。
    """
    config = config or AgentConfig()
    root = Path(dataset_root).resolve()
    dataset_id = root.name

    dirty_path = root / F_DIRTY_TRAIN
    if not dirty_path.exists():
        raise FileNotFoundError(f"未找到脏训练集：{dirty_path}")
    dirty_df = pd.read_csv(dirty_path, encoding="utf-8")

    log: list[str] = [
        f"[run_agent] 数据集 {dataset_id}：{len(dirty_df)} 行 x {dirty_df.shape[1]} 列；"
        f"config(planner={config.use_planner}, reviewer={config.use_reviewer}, "
        f"prompt={config.prompt_strategy}, model={config.model})。"
    ]
    profile = tools.profile_data(dirty_df)
    log.append(
        f"[run_agent] 画像：缺失列 {len(profile['missing'])}，重复行 {profile['n_duplicate_rows']}，"
        f"疑似泄漏 {profile['high_corr_cols']}，不平衡比 {profile['imbalance_ratio']}。"
    )

    init_state: AgentState = {
        "task": task,
        "dataset_root": str(root),
        "dataset_id": dataset_id,
        "dirty_df": dirty_df,
        "profile": profile,
        "cleaned_df": dirty_df,
        "plan": "",
        "reported_defects": empty_report(),
        "review_feedback": "",
        "needs_revision": False,
        "review_round": 0,
        "use_planner": config.use_planner,
        "use_reviewer": config.use_reviewer,
        "prompt_strategy": config.prompt_strategy,
        "model": config.model,
        "max_review_rounds": config.max_review_rounds,
        "log": log,
    }

    graph = _build_graph(config)
    # recursion_limit 给反思回路留足空间：每轮 executor+reviewer 两步 + 规划/汇报余量
    final_state = graph.invoke(
        init_state,
        config={"recursion_limit": 4 + 2 * (config.max_review_rounds + 1)},
    )

    return CleaningResult(
        cleaned_train_path=Path(final_state["cleaned_train_path"]),
        reported_defects=final_state["reported_defects"],
        log=final_state["log"],
        extra={
            "dataset_id": dataset_id,
            "config": {
                "use_planner": config.use_planner,
                "use_reviewer": config.use_reviewer,
                "prompt_strategy": config.prompt_strategy,
                "model": config.model,
                "max_review_rounds": config.max_review_rounds,
            },
            "profile": profile,
            "review_rounds": final_state.get("review_round", 0),
            "plan": final_state.get("plan", ""),
        },
    )
