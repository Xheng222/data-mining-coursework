"""契约层的基本单元测试（foundation 自带，保证 import 与公式正确）。"""

import pytest

from src.contracts import (
    DEFECT_TYPES,
    empty_report,
    normalize_pair,
    recovery_rate,
)


def test_recovery_rate_basic():
    assert recovery_rate(0.6, 0.8, 1.0) == pytest.approx(0.5)
    assert recovery_rate(0.7, 0.7, 0.7) == 0.0  # 分母为 0 时安全返回


def test_normalize_pair_orders():
    assert normalize_pair(5, 2) == "2-5"
    assert normalize_pair(1, 3) == "1-3"


def test_empty_report_has_all_types():
    r = empty_report()
    assert set(r.keys()) == set(DEFECT_TYPES)
    assert all(r[t]["items"] == [] for t in DEFECT_TYPES)
