"""核心进阶算法的命令行入口（对齐中期模板的模块命名）。feature/agent 负责实现。

薄封装：复用 src.agent.run_agent，提供 CLI：
  uv run python -m src.core_model --dataset data/synthetic/<id> \
      --reviewer on --planner on --prompt cot --out results/agent.json
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
