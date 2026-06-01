"""端到端实验编排：生成数据 -> 跑各方法 -> 算指标 -> 落盘结果与图表。

feature/experiments 负责实现。CLI：
  uv run python -m src.run_experiments --config configs/midterm.yaml
产出：results/results.csv、results/detection.csv、results/figures/*.png
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
