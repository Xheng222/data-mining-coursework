# 脏数据基准：LLM Data Agent 评估框架

评估 LLM 驱动的 Data Agent 在**缺陷检测**与**数据修复**上的能力。框架在自带
ground-truth 的合成脏数据上，对比四类方法在同一固定干净测试集上的下游性能：

- `clean_upper`：干净数据训练（性能上界）
- `no_clean`：脏数据直接训练（下界）
- `rule_based`：固定规则清洗（`ydata-profiling` 思路的简化版）
- `agent`：LangGraph 状态机 Data Agent（Planner -> Executor -> Reviewer -> Reporter）

## 报告

- 开题报告：[reports/开题报告.md](./reports/开题报告.md)
- 中期进展报告：[reports/中期进展报告.md](./reports/中期进展报告.md)

## 环境

- Python 3.13，包管理用 [uv](https://github.com/astral-sh/uv)
- DeepSeek API key：复制 `.env.example` 为 `.env` 并填入 `DEEPSEEK_API_KEY`（仅 Agent 需要；基线不需要）

```bash
uv sync --extra dev          # 安装依赖
cp .env.example .env         # 填入 DEEPSEEK_API_KEY
```

## 复现

```bash
# 全量中期实验：生成数据 -> 运行四类方法 -> 计算指标 -> 作图
uv run python -m src.run_experiments --config configs/midterm.yaml

# 快速测试
uv run python -m src.run_experiments --config configs/midterm.yaml --quick --max-datasets 1

# 只跑基线、不调用 LLM（无需 API key）
uv run python -m src.run_experiments --config configs/midterm.yaml --skip-agent
```

产出写到 `results/`：`results.csv`、`detection_by_type.csv`、`summary.json`、`figures/*.png`。

单独运行某个组件：

```bash
# 生成一个合成数据集
uv run python -c "from src.datagen import generate_dataset; generate_dataset('demo', 'medium', 42)"

# 基线（无需 API key）
uv run python -m src.baseline --dataset data/synthetic/demo --out results/baseline.json

# Agent（需要 API key），可切换消融开关
uv run python -m src.core_model --dataset data/synthetic/demo \
    --reviewer on --planner on --prompt cot --out results/agent.json

# 测试
uv run pytest -q
```

## 目录结构

```text
  ├── README.md              # 环境配置 + 一键复现命令
  ├── pyproject.toml         # 依赖（uv 管理）
  ├── configs/midterm.yaml   # 实验配置：数据集 / 方法 / 消融
  ├── src/
  │   ├── contracts.py       # 跨模块契约：数据类、缺陷与检测 schema、recovery_rate 公式
  │   ├── datagen/           # 合成数据 + 六类缺陷注入（base/defects/generate）
  │   ├── preprocess.py      # 规则化清洗步骤（缺失填充/去重/泄漏列剔除/格式统一）
  │   ├── baseline.py        # no_clean / rule_based / clean_upper 三基线 + CLI
  │   ├── agent/             # LangGraph Agent：state/tools/prompts/nodes/graph
  │   ├── core_model.py      # Agent 命令行入口（含消融开关）
  │   ├── evaluate.py        # Detection(P/R/F1) 与 Recovery Rate
  │   ├── run_experiments.py # 端到端实验编排
  │   └── viz.py             # 结果可视化
  ├── tests/                 # pytest：契约/数据/评测/基线/Agent
  ├── data/                  # 合成数据（可由代码重新生成，未入库）
  └── results/               # 指标 CSV、汇总 JSON、图表 PNG
```
