# AGENTS.md — data-mining-coursework

## Project overview

Evaluate LLM Data Agents on **defect detection + data repair** against synthetic dirty data with ground truth. Four methods compared: `clean_upper` (upper bound), `no_clean` (lower bound), `rule_based` (fixed rules), `agent` (LangGraph state machine).

## Commands

```bash
uv sync --extra dev                   # install deps (uses uv, not pip/poetry)
uv run python -m src.run_experiments --config configs/midterm.yaml   # full pipeline
uv run python -m src.run_experiments --config configs/midterm.yaml --quick --max-datasets 1  # smoke test
uv run python -m src.run_experiments --config configs/midterm.yaml --skip-agent              # baselines only
uv run pytest -q                      # all tests
```

## Project structure

- `src/contracts.py` — **read-only** shared schemas (DatasetBundle, DefectReport, DetectionScore, recovery_rate formula). All modules depend on it; do not modify.
- `src/run_experiments.py` — end-to-end orchestration (generate → run methods → evaluate → viz)
- `src/baseline.py` — no_clean / rule_based / clean_upper + CLI entrypoint
- `src/agent/` — LangGraph agent: state/tools/prompts/nodes/graph. Config via `AgentConfig` (use_planner, use_reviewer, prompt_strategy, model)
- `src/core_model.py` — agent CLI entrypoint with ablation switches
- `src/datagen/` — synthetic data generation with 6 defect types injected
- `src/evaluate.py` — evaluation: XGBoost train+auc, detection_scores (P/R/F1 per type + micro avg)
- `src/preprocess.py` — rule_based_clean implementation
- `configs/midterm.yaml` — experiment config (datasets, methods, ablations)
- `tests/` — pytest tests; LLM e2e tests gated by `DEEPSEEK_API_KEY` + `RUN_LLM_TESTS=1`

## Key facts

- **Python 3.13 required** (`.python-version`). Package manager is `uv` — never use pip directly.
- **DeepSeek API key**: copy `.env.example` to `.env`, fill `DEEPSEEK_API_KEY`. Used by agent only; baselines need no key.
- **No CI/CD**, no linter/formatter/typechecker configured — only pytest.
- **`pyproject.toml`** sets `pythonpath = ["."]` for pytest. All imports use `from src.xxx import ...`.
- **Testing convention**: `_RUN_LLM = bool(os.environ.get("DEEPSEEK_API_KEY")) and bool(os.environ.get("RUN_LLM_TESTS"))` gates real LLM tests (`test_agent.py`).
- **AUC evaluation**: uses XGBClassifier with `tree_method="hist"`. Numeric coercion via `pd.to_numeric(errors="coerce")`. NaN is handled natively by XGBoost.
- **rule_based_clean** returns DataFrame in-memory via `CleaningResult.extra["cleaned_df"]`; `baseline.py` does actual persistence.
- **Results** written to `results/`: `results.csv`, `detection_by_type.csv`, `summary.json`, `figures/*.png`.
- **Generated data** in `data/synthetic/*/` and `data/processed/*.csv` are gitignored (can be regenerated).
- **LangGraph agent** uses state machines with Planner → Executor → Reviewer → Reporter nodes. Ablations disable Reviewer or Planner to isolate their contribution.
- **Detection scoring**: micro-averaged P/R/F1 across 6 defect types. Empty ground-truth + empty report → 1.0.
