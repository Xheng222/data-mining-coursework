"""真实数据集基座加载器：从 sklearn / UCI 等来源加载数据，产出与 make_base 兼容的 DataFrame。

每个加载函数返回一个 ``pd.DataFrame``，包含数值特征列 + 二值 ``target`` 列（0/1），
可直接传入 :func:`src.datagen.generate.generate_dataset` 的 ``base_df`` 参数，
复用同一套 6 类缺陷注入流程。

用法::

    from src.datagen.real_base import load_real_dataset
    df = load_real_dataset("breast_cancer")
    print(df.shape)   # (569, 31) — 30 features + target
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.contracts import TARGET_COL

# --------------------------------------------------------------------------- #
# 内置 sklearn 数据集
# --------------------------------------------------------------------------- #

# fmt: off
_PIMA_URL = (  # noqa: E501
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
_PIMA_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", TARGET_COL,
]
# fmt: on


def _load_breast_cancer() -> pd.DataFrame:
    """Wisconsin Breast Cancer (569×30, 二分类 恶性/良性)。"""
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[TARGET_COL] = data.target.astype(int)
    return df


def _load_digits_binary() -> pd.DataFrame:
    """Digits (1797×64) 二值化：digit >= 5 → 1, 否则 0。"""
    from sklearn.datasets import load_digits

    data = load_digits()
    df = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(data.data.shape[1])])
    df[TARGET_COL] = (data.target >= 5).astype(int)
    return df


def _load_wine_binary() -> pd.DataFrame:
    """Wine (178×13) 二值化：class 0 → 1, 其余 → 0。"""
    from sklearn.datasets import load_wine

    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[TARGET_COL] = (data.target == 0).astype(int)
    return df


def _load_pima_diabetes() -> pd.DataFrame:
    """Pima Indians Diabetes (768×8, 二分类)。"""
    try:
        df = pd.read_csv(_PIMA_URL, header=None, names=_PIMA_COLS)
    except Exception:
        # 离线回退：尝试从本地 data/ 读取
        local = Path("data") / "pima-indians-diabetes.csv"
        if local.exists():
            df = pd.read_csv(local, header=None, names=_PIMA_COLS)
        else:
            raise
    return df


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #

REAL_BASE_REGISTRY: dict[str, tuple[str, callable]] = {
    "breast_cancer": ("Wisconsin Breast Cancer (569×30)", _load_breast_cancer),
    "digits": ("Digits binary (1797×64)", _load_digits_binary),
    "wine": ("Wine binary (178×13)", _load_wine_binary),
    "pima_diabetes": ("Pima Indians Diabetes (768×8)", _load_pima_diabetes),
}


def load_real_dataset(name: str) -> pd.DataFrame:
    """加载一个真实数据集基座。

    参数
    ----------
    name : str
        数据集名称，须在 ``REAL_BASE_REGISTRY`` 中注册。

    返回
    -------
    pd.DataFrame
        含数值特征列与 ``target`` 列（0/1）的 DataFrame。
    """
    if name not in REAL_BASE_REGISTRY:
        msg = f"未知真实数据集：{name!r}，可用：{list(REAL_BASE_REGISTRY)}"
        raise ValueError(msg)
    desc, loader = REAL_BASE_REGISTRY[name]
    df = loader()
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"{name} 加载后缺少 target 列")
    if df[TARGET_COL].nunique() != 2:
        raise RuntimeError(f"{name} 的目标列不是二值（unique={df[TARGET_COL].nunique()}）")
    return df
