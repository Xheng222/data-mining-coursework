"""基线方法与命令行入口。feature/baseline 负责实现。

基线：
  - no_clean   ：脏数据直接训练（下界）
  - rule_based ：调用 preprocess.rule_based_clean 后训练

CLI（中期一键复现）：
  uv run python -m src.baseline --dataset data/synthetic/<id> --out results/baseline.json
"""

from __future__ import annotations

from pathlib import Path

from src.contracts import MethodOutcome


def run_no_clean(dataset_root: str | Path) -> MethodOutcome:
    raise NotImplementedError


def run_rule_based(dataset_root: str | Path) -> MethodOutcome:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
