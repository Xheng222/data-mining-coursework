"""核心进阶算法的命令行入口（对齐中期模板的模块命名）。

薄封装：解析命令行参数构造 AgentConfig，调用 src.agent.run_agent，把报告的缺陷、
清洗后路径与处理日志序列化成 JSON 写入 --out。

示例：
  uv run python -m src.core_model --dataset data/synthetic/<id> \
      --reviewer on --planner on --prompt cot --model deepseek-chat \
      --out results/agent.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent import AgentConfig, run_agent


def _on_off(value: str) -> bool:
    """把 on/off（及常见同义词）解析为布尔。"""
    v = value.strip().lower()
    if v in ("on", "true", "1", "yes", "y"):
        return True
    if v in ("off", "false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"期望 on/off，得到：{value!r}")


def main() -> None:
    """解析参数、运行 Agent、把结果写成 JSON。"""
    parser = argparse.ArgumentParser(
        description="对脏数据集运行 LangGraph Data Agent，输出缺陷报告与清洗结果。"
    )
    parser.add_argument("--dataset", required=True, help="数据集根目录（含 dirty_train.csv）")
    parser.add_argument("--reviewer", type=_on_off, default=True, help="是否启用 Reviewer 反思节点 (on/off)")
    parser.add_argument("--planner", type=_on_off, default=True, help="是否启用 Planner 规划节点 (on/off)")
    parser.add_argument(
        "--prompt",
        default="cot",
        choices=["zero_shot", "few_shot", "cot"],
        help="提示策略",
    )
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek 模型名")
    parser.add_argument(
        "--max-review-rounds", type=int, default=2, help="Reviewer 最大反思轮数"
    )
    parser.add_argument(
        "--task",
        default="请清洗该数据集并训练一个二分类模型",
        help="自然语言任务描述",
    )
    parser.add_argument("--out", default="results/agent.json", help="结果 JSON 输出路径")
    args = parser.parse_args()

    config = AgentConfig(
        use_planner=args.planner,
        use_reviewer=args.reviewer,
        prompt_strategy=args.prompt,
        model=args.model,
        max_review_rounds=args.max_review_rounds,
    )

    result = run_agent(args.dataset, task=args.task, config=config)

    payload = {
        "dataset": str(Path(args.dataset).resolve()),
        "cleaned_train_path": str(result.cleaned_train_path),
        "reported_defects": result.reported_defects,
        "log": result.log,
        "config": result.extra.get("config", {}),
        "review_rounds": result.extra.get("review_rounds", 0),
        "plan": result.extra.get("plan", ""),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Agent 完成：缺陷报告与清洗结果已写入 {out_path.resolve()}")


if __name__ == "__main__":
    main()
