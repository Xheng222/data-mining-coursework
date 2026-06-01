"""合成数据集生成入口（feature/datagen 负责实现）。"""

from __future__ import annotations

from pathlib import Path

from src.contracts import DatasetBundle


def generate_dataset(
    difficulty: str = "medium",
    seed: int = 42,
    out_dir: str | Path = "data/synthetic",
) -> DatasetBundle:
    """生成一个带已知缺陷的合成分类数据集，落盘并返回 DatasetBundle。

    流程：make_classification 造基座 -> 切 20% 干净测试集冻结 ->
    其余按 difficulty 注入六类缺陷 -> 写 dirty_train / clean_train /
    clean_test / ground_truth.json / meta.json。
    """
    raise NotImplementedError
