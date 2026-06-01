"""agent 模块测试。

默认不触网：真实 LLM 端到端用例通过环境变量门控（默认 skip）；
另有不依赖网络的轻量测试，校验 AgentConfig 默认值与 run_agent 签名/可导入性。
"""

from __future__ import annotations

import inspect
import os

import pytest

from src.agent import AgentConfig, run_agent
from src.contracts import CleaningResult


# --------------------------------------------------------------------------- #
# 不依赖网络的轻量测试
# --------------------------------------------------------------------------- #

def test_agent_config_defaults():
    """AgentConfig 默认值应与消融设计一致。"""
    cfg = AgentConfig()
    assert cfg.use_planner is True
    assert cfg.use_reviewer is True
    assert cfg.prompt_strategy == "cot"
    assert cfg.model == "deepseek-chat"
    assert cfg.max_review_rounds == 2


def test_agent_config_overridable():
    """消融开关可被覆盖。"""
    cfg = AgentConfig(use_reviewer=False, use_planner=False, prompt_strategy="zero_shot")
    assert cfg.use_reviewer is False
    assert cfg.use_planner is False
    assert cfg.prompt_strategy == "zero_shot"


def test_run_agent_importable_and_signature():
    """run_agent 可被 import，且签名含 dataset_root / task / config。"""
    assert callable(run_agent)
    sig = inspect.signature(run_agent)
    params = sig.parameters
    assert "dataset_root" in params
    assert "task" in params
    assert "config" in params
    # config 默认应为 None（运行时回退到 AgentConfig()）
    assert params["config"].default is None


# --------------------------------------------------------------------------- #
# 真实 LLM 端到端（默认 skip，需显式开启）
# --------------------------------------------------------------------------- #

_RUN_LLM = bool(os.environ.get("DEEPSEEK_API_KEY")) and bool(
    os.environ.get("RUN_LLM_TESTS")
)


@pytest.mark.skipif(
    not _RUN_LLM,
    reason="需要 DEEPSEEK_API_KEY 且设置 RUN_LLM_TESTS=1 才运行真实 LLM 端到端测试",
)
def test_run_agent_end_to_end(tmp_path):
    """真实 Agent 跑通：对脏数据集产出 CleaningResult。"""
    from src.datagen import generate_dataset

    bundle = generate_dataset(
        "a",
        difficulty="easy",
        seed=0,
        out_dir=tmp_path,
        n_rows=300,
        n_features=10,
    )
    result = run_agent(bundle.root, config=AgentConfig())
    assert isinstance(result, CleaningResult)
    assert result.cleaned_train_path.exists()
    assert isinstance(result.reported_defects, dict)
