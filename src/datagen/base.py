"""基座数据生成：对 ``sklearn.datasets.make_classification`` 的薄封装。

生成一份干净的二分类表格数据，特征列命名为 ``f0..f{n-1}``，
标签列名取自 :data:`src.contracts.TARGET_COL`（值为 0/1）。
后续的缺陷注入与数据集组装都以本函数的输出为起点。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.contracts import TARGET_COL


def make_base(
    n_rows: int = 10000,
    n_features: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """生成一份干净的二分类基座数据集。

    Parameters
    ----------
    n_rows:
        样本行数。
    n_features:
        特征列数（不含标签列）。
    seed:
        随机种子，保证可复现。

    Returns
    -------
    pandas.DataFrame
        含 ``n_features`` 个特征列 ``f0..f{n-1}`` 与一个标签列
        （列名为 :data:`TARGET_COL`，取值 0/1）。
    """
    if n_features < 4:
        raise ValueError("n_features 至少为 4，以便划分 informative/redundant 特征。")

    # 让 informative / redundant / repeated 三类特征占据合理比例，
    # 其余为纯噪声特征，保证数据集既可学习又不过于平凡。
    n_informative = max(2, int(round(n_features * 0.4)))
    n_redundant = max(1, int(round(n_features * 0.2)))
    n_repeated = max(1, int(round(n_features * 0.1)))
    # 防止三类之和超过总特征数。
    while n_informative + n_redundant + n_repeated > n_features:
        if n_repeated > 1:
            n_repeated -= 1
        elif n_redundant > 1:
            n_redundant -= 1
        else:
            n_informative -= 1

    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        flip_y=0.0,
        class_sep=1.0,
        shuffle=True,
        random_state=seed,
    )

    columns = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df[TARGET_COL] = np.asarray(y, dtype=int)
    return df
