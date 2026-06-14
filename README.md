# 脏数据基准：LLM Data Agent 评估框架

评估 LLM 驱动的 Data Agent 在**缺陷检测**与**数据修复**上的能力。框架在自带 ground-truth 的脏数据上，对比若干方法在同一固定干净测试集上的下游性能：

- `clean_upper`：干净数据训练（性能上界）
- `no_clean`：脏数据直接训练（下界）
- `rule_based`：固定规则清洗（`ydata-profiling` 思路的简化版）
- `agent`：LangGraph 状态机 Data Agent（Planner -> Executor -> Reviewer -> Reporter）

Agent 通过消融开关派生出 5 个变体：`agent_full`、`agent_no_reviewer`、`agent_no_planner`（隔离 Reviewer / Planner 节点），以及 `agent_zero_shot`、`agent_few_shot`（对比 Prompt 策略，默认是 CoT）。

## 数据集

框架支持四类数据来源，统一为 `dirty_train.csv / clean_train.csv / clean_test.csv / ground_truth.json` 的目录结构：

- **合成数据**（`source: synthetic`）：用 `sklearn.make_classification` 生成干净的基础数据，注入六类缺陷并记录 ground-truth，缺陷强度由 `difficulty`（easy/medium/hard）决定。
- **CleanML 原生**（`source: cleanml`）：CleanML 基准的真实脏/净配对（Credit / EEG / Marketing）。这类数据没有逐缺陷的 ground-truth，只写入空报告，因此只对 Recovery Rate 有意义，检测 P/R/F1 在这类数据上没有参考价值。
- **CleanML 干净基座 + 受控注入**（`source: cleanml_base`）：以 CleanML 的干净版本作为基础数据，复用与合成数据相同的六类缺陷注入，既有真实复杂度又有 ground-truth（如 `Credit_base` / `EEG_base`）。
- **真实数据基座 + 受控注入**（`source: real_base`）：从 sklearn / UCI 加载真实数据（`breast_cancer` / `digits` / `wine` / `pima_diabetes`）作为基础数据，同样注入六类缺陷。

### 六类缺陷

数据泄漏、缺失值、标签噪声、近似重复、类别不平衡、格式不一致。注入按固定顺序执行，互不干扰；每类都带 ground-truth，供检测指标比对。

## 评估指标

- **Recovery Rate**：在固定干净测试集上训练下游分类器并计算 AUC-ROC，归一化为 `(AUC_method − AUC_dirty) / (AUC_clean − AUC_dirty)`。RR=1 表示完全恢复到干净水平，RR<0 表示比不清洗还差。
- **Detection P/R/F1**：把报告的缺陷与 ground-truth 按缺陷类型逐一比对，计算每类指标与 micro 平均。
- **下游模型**：默认 XGBoost；可在配置里切换或叠加 RandomForest(`rf`)、LogisticRegression(`lr`)，用于检验评估结论在不同下游模型下是否稳定（见 `eval_models` 与每个数据集的 `models` 覆盖）。

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

### CleanML 数据（仅 `cleanml` / `cleanml_base` 数据集需要）

CleanML 原始数据没有包含在仓库里，需自行放到 `data/CleanML/data/` 下，目录形如 `data/CleanML/data/{Credit_major,EEG_major,Marketing}/raw/`（加载器读取 `raw.csv` / `mislabel_clean_raw.csv` / `orgin.csv`，见 `src/datagen/cleanml.py`）。若只运行合成数据与真实基座（`real_base`），则不需要 CleanML 数据。

`pima_diabetes` 默认从 GitHub 在线拉取 CSV；离线时把文件放到 `data/pima-indians-diabetes.csv` 作为回退。

## 复现

```bash
# 最终实验：合成 + CleanML + 真实基座，共 11 个数据集
uv run python -m src.run_experiments --config configs/final.yaml

# 中期实验
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

# 基线（无需 API key），可切换下游模型
uv run python -m src.baseline --dataset data/synthetic/demo --out results/baseline.json

# Agent（需要 API key），可切换消融开关
uv run python -m src.core_model --dataset data/synthetic/demo --reviewer on --planner on --prompt cot --out results/agent.json

# 测试
uv run pytest -q
```

## 目录结构

```text
  ├── README.md              # 环境配置 + 一键复现命令
  ├── pyproject.toml         # 依赖（uv 管理）
  ├── configs/
  │   ├── midterm.yaml       # 中期实验配置
  │   └── final.yaml         # 最终实验配置：合成 + CleanML + 真实基座 + 多模型/消融
  ├── src/
  │   ├── contracts.py       # 跨模块契约：数据类、缺陷与检测 schema、recovery_rate 公式
  │   ├── datagen/           # 数据来源
  │   │   ├── generate.py    # 合成数据 + 六类缺陷注入（支持外部 base_df）
  │   │   ├── cleanml.py     # CleanML 脏/净配对加载
  │   │   └── real_base.py   # 真实数据基座（sklearn/UCI）
  │   ├── preprocess.py      # 规则化清洗步骤（缺失填充/去重/泄漏列剔除/格式统一）
  │   ├── baseline.py        # no_clean / rule_based / clean_upper 三基线 + CLI
  │   ├── agent/             # LangGraph Agent：state/tools/prompts/nodes/graph
  │   │                      #   tools.py 画像含 CV 残差标签噪声信号与基于距离的近似重复检测
  │   ├── core_model.py      # Agent 命令行入口（含消融开关）
  │   ├── evaluate.py        # Detection(P/R/F1) 与多模型 AUC（xgb/rf/lr）
  │   ├── run_experiments.py # 端到端实验编排（四类数据来源 + 每数据集模型覆盖）
  │   └── viz.py             # 结果可视化
  ├── tests/                 # pytest：契约/数据/评测/基线/Agent
  ├── data/                  # 数据（合成数据可重生成、CleanML 需自备，均未入库）
  └── results/               # 指标 CSV、汇总 JSON、图表 PNG（含 model_comparison.png）
```
